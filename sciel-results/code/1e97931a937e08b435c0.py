import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def track_call(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return best_f
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != dim:
                return best_f
            val = func(x)
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = np.array(x)
            return val

        # ---------- Phase 0: cheap random scan to get a decent start ----------
        np.random.seed(0)
        for _ in range(20):
            if calls_used >= budget:
                break
            x = np.random.uniform(-4, 4, dim)
            track_call(x)

        x_mean = np.array(best_x, dtype=float)
        sigma = 1.5

        # ---------- Phase 1: rotation discovery (fit covariance anisotropy) ----------
        # We estimate a rotation R by maximizing variance of f along axes.
        # Represent R via 5x5 parameters, optimize with small Nelder-Mead.
        def rotation_from_params(p):
            # Build a random orthogonal-ish matrix from params (use SVD to orthonormalize)
            A = np.zeros((dim, dim))
            A[0,0] = 1.0
            A[1,1] = 1.0
            A[2,2] = 1.0
            A[3,3] = 1.0
            A[4,4] = 1.0
            # Inject perturbations from p
            for i in range(dim):
                for j in range(dim):
                    if i != j:
                        A[i,j] = p[i*dim+j] * 0.5
            # Orthonormalize
            try:
                U, _, Vt = np.linalg.svd(A)
                R = U @ Vt
            except np.linalg.LinAlgError:
                R = np.eye(dim)
            return R

        def variance_objective(p):
            R = rotation_from_params(p)
            # Sample a few points along each axis direction (cheap probe)
            total_var = 0.0
            for k in range(dim):
                d = R[:, k]
                vals = []
                for s in (-1.0, 1.0):
                    y = x_mean + s * 0.8 * d
                    y = np.clip(y, -5, 5)
                    vals.append(func(y))
                total_var += (vals[1] - vals[0]) ** 2
            # We want to MAXIMIZE spread; return negative for minimization
            return -total_var

        # Small Nelder-Mead to find rotation
        p0 = np.zeros(dim*dim)
        simplex = [p0.copy()]
        for i in range(dim*dim):
            p = p0.copy()
            p[i] = 0.3
            simplex.append(p)

        # Evaluate simplex
        fvals = [variance_objective(s) for s in simplex]
        nm_iter = 0
        max_nm = 40
        while nm_iter < max_nm and calls_used < budget - 100 and time.time() < deadline - 30.0:
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)
            x_refl = centroid + 1.0 * (centroid - simplex[-1])
            f_refl = variance_objective(x_refl)

            if f_refl < fvals[0]:
                # expansion
                x_exp = centroid + 2.0 * (centroid - simplex[-1])
                f_exp = variance_objective(x_exp)
                if f_exp < f_refl:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_refl
                    fvals[-1] = f_refl
            elif f_refl < fvals[-2]:
                simplex[-1] = x_refl
                fvals[-1] = f_refl
            else:
                # contraction
                x_con = centroid + 0.5 * (simplex[-1] - centroid)
                f_con = variance_objective(x_con)
                if f_con < fvals[-1]:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = variance_objective(simplex[i])
            nm_iter += 1

        # Extract best rotation
        order = np.argsort(fvals)
        best_R = rotation_from_params(simplex[order[0]])

        # Build initial covariance from R (elongated along anisotropic directions)
        # Give larger variance along axes where function varies most
        eigvals = np.array([2.0, 1.5, 1.0, 0.8, 0.6])
        C = best_R @ np.diag(eigvals) @ best_R.T
        C = (C + C.T) / 2
        sigma = 1.0

        # ---------- Phase 2: CMA-ES with discovered covariance ----------
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)

        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim+1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim+2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff-1)/(dim+1)) - 1) + cs
        chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        ps = np.zeros(dim)
        pc = np.zeros(dim)
        hsig = 0

        last_improvement_call = calls_used
        best_f_history = [best_f]

        while calls_used < budget and time.time() < deadline - 2.0:
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-6 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(dim)

            pop = np.zeros((lam, dim))
            for i in range(lam):
                if calls_used >= budget:
                    break
                z = np.random.randn(dim)
                pop[i] = x_mean + sigma * (L @ z)
                pop[i] = np.clip(pop[i], -5, 5)

            fit = np.zeros(lam)
            for i in range(lam):
                if calls_used >= budget:
                    break
                fit[i] = track_call(pop[i])

            idx = np.argsort(fit)
            pop = pop[idx]
            fit = fit[idx]

            w = weights[:mu]
            x_new = np.sum(w[:, None] * pop[:mu], axis=0)

            step = np.linalg.norm(x_new - x_mean)
            if step > 0:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / step * (x_mean - x_new)
            ps_norm = np.linalg.norm(ps)
            hsig = (ps_norm / np.sqrt(1 - (1 - cs)**2) / chiN) < (1.4 + 2/(dim+1))
            sigma = sigma * np.exp(cs/damps * (ps_norm/chiN - 1))

            if hsig:
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) / step * (x_mean - x_new)
            else:
                pc = (1 - cc) * pc

            y = (x_mean - x_new) / sigma
            C = (1 - c1 - cmu) * C + c1 * np.outer(y, y)
            rank_mu_term = np.zeros((dim, dim))
            for i in range(mu):
                v_i = (pop[i] - x_mean) / sigma
                rank_mu_term += w[i] * np.outer(v_i, v_i)
            C = C + cmu * rank_mu_term

            x_mean = x_new
            C = (C + C.T) / 2
            try:
                np.linalg.cholesky(C)
            except:
                C += 1e-5 * np.eye(dim)

            if fit[0] < best_f_history[-1]:
                last_improvement_call = calls_used
                best_f_history.append(fit[0])
            else:
                if calls_used - last_improvement_call > 5 * lam:
                    sigma *= 1.5
                    C = 0.5 * C + 0.5 * np.eye(dim)
                    last_improvement_call = calls_used
                    best_f_history = [fit[0]]

        # ---------- Phase 3: final polish with tight local search ----------
        if calls_used < budget - 5 and time.time() < deadline - 3.0:
            # Nelder-Mead polish around best point
            x_cur = np.array(best_x, dtype=float)
            f_cur = best_f
            step_size = 0.2
            simplex = [x_cur.copy()]
            for i in range(dim):
                x = x_cur.copy()
                x[i] += step_size
                simplex.append(x)
            fvals = [func(np.clip(s, -5, 5)) for s in simplex]

            for _ in range(30):
                if calls_used >= budget or time.time() > deadline - 3.0:
                    break
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                centroid = np.mean(simplex[:-1], axis=0)
                x_refl = centroid + 1.0 * (centroid - simplex[-1])
                x_refl = np.clip(x_refl, -5, 5)
                f_refl = func(x_refl)
                if f_refl < fvals[0]:
                    x_exp = centroid + 2.0 * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    if f_exp < f_refl:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_refl
                        fvals[-1] = f_refl
                elif f_refl < fvals[-2]:
                    simplex[-1] = x_refl
                    fvals[-1] = f_refl
                else:
                    x_con = centroid + 0.5 * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = np.array(simplex[0], dtype=float)

        return best_x, best_f
