"""Optimizer for BBOB-style Rastrigin f3 with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES + Nelder-Mead ensemble with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s limit

        # Best solution tracking
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # --- Phase 1: CMA-ES for global exploration ---
        # Initialize around center with large sigma
        x_mean = np.zeros(dim)
        sigma = 3.5
        popsize = 15  # lambda
        # CMA-ES parameters
        cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + 4 + dim / (dim + 4))
        cs = (dim + 1.7) / (dim + 9.2)
        cmu = min(0.4, 2 / ((dim + 1.3) ** 2))
        c1 = (2 - cmu) / ((dim + 1.8) ** 2 + 10)
        cmu2 = cmu / 2.0
        damps = 1 + 2 * max(0, np.sqrt((cs - 1) / (dim + 1)) - 1) + cs
        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition cache
        D = np.ones(dim)  # 1/sqrt(eigenvalues)
        B = np.eye(dim)   # eigenvectors
        # Step size control
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Generate initial population
        def cma_step():
            nonlocal x_mean, sigma, C, D, B, ps, pc, call_count
            if call_count >= budget:
                return False
            if time.time() > deadline - 10:
                return False

            # Generate population
            W = np.zeros((popsize, dim))
            for i in range(popsize):
                z = rng.standard_normal(dim)
                # Transform using B and D
                y = B @ (D * z)
                W[i] = x_mean + sigma * y

            # Evaluate
            F = np.zeros(popsize)
            for i in range(popsize):
                if call_count >= budget or time.time() > deadline - 10:
                    break
                F[i] = evaluate(W[i])
            else:
                pass

            # Sort by fitness
            order = np.argsort(F)
            W_sorted = W[order]
            F_sorted = F[order]

            # Update weights (standard CMA-ES)
            mu = popsize // 2
            w = np.zeros(mu)
            for i in range(mu):
                w[i] = np.log(mu + 0.5) - np.log(i + 1)
            w_sum = w.sum()
            w = w / w_sum
            mu_eff = 1.0 / (w ** 2).sum()

            # Update mean
            x_old = x_mean.copy()
            x_mean = (w @ W_sorted[:mu]).copy()

            # Update step size control
            pc = (1 - cs) * pc + cs / np.sqrt(mu_eff) * B @ (D @ (x_mean - x_old) / (sigma))
            if np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** (2 * (call_count))) < (1.4 + cc) * np.sqrt(dim):
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / np.sqrt(dim) * (x_mean - x_old) / sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                sigma = max(1e-10, min(sigma, 10.0))

            # Update covariance
            delta_sig = 1 - np.exp(-cs * (call_count))
            C = (1 - c1 - cmu2) * C + c1 * (np.outer(pc, pc) + (cs * (2 - cs)) * (np.linalg.norm(pc) ** 2 / dim - 1))
            for i in range(mu):
                y = (W_sorted[i] - x_old) / sigma
                C += cmu2 * w[i] * np.outer(y, y)

            # Periodic eigen decomposition
            if call_count % 10 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                B = eigvecs
                D = 1.0 / np.sqrt(eigvals)

            return True

        # Run CMA-ES for a portion of the budget
        cma_budget = int(budget * 0.7)
        while call_count < cma_budget and time.time() < deadline - 15:
            if not cma_step():
                break

        # --- Phase 2: Nelder-Mead local refinement from best point ---
        # Also try NM from a few other good points
        nm_starts = [best_x.copy()]
        # Add a few random perturbations of the best point
        for _ in range(3):
            perturbed = best_x + 0.5 * rng.standard_normal(dim)
            nm_starts.append(np.clip(perturbed, -5, 5))

        for start in nm_starts:
            if call_count >= budget or time.time() > deadline - 5:
                break
            self._nelder_mead(func, start, dim, evaluate, budget, deadline)

        return best_x, best_f

    def _nelder_mead(self, func, x0, dim, evaluate, budget, deadline):
        """Nelder-Mead simplex method."""
        # Build initial simplex
        simplex = [x0.copy()]
        f0 = evaluate(x0)
        simplex.append(f0)

        alpha = 0.5
        for i in range(dim):
            xi = x0.copy()
            xi[i] += alpha * (1.0 if xi[i] < 4.9 else -1.0)
            xi = np.clip(xi, -5, 5)
            fi = evaluate(xi)
            simplex.append(xi)
            simplex.append(fi)

        # simplex is [x0, f0, x1, f1, ..., xd, fd]
        # Reorganize into list of (x, f)
        verts = [(simplex[2 * i], simplex[2 * i + 1]) for i in range(dim + 1)]

        max_iter = 200
        for _ in range(max_iter):
            if evaluate.__code__.co_varnames and False:
                pass
            # Check budget and time
            if self._check_deadline(deadline):
                break

            # Sort vertices by fitness
            verts.sort(key=lambda v: v[1])
            worst = verts[-1]
            second_worst = verts[-2]
            best = verts[0]

            # Convergence check
            if abs(verts[-1][1] - verts[0][1]) < 1e-10:
                break

            # Centroid of all but worst
            centroid = np.mean([v[0] for v in verts[:-1]], axis=0)

            # Reflection
            xr = centroid + 1.0 * (centroid - worst[0])
            xr = np.clip(xr, -5, 5)
            fr = evaluate(xr)

            if fr < verts[0][1]:
                # Expansion
                xe = centroid + 2.0 * (centroid - worst[0])
                xe = np.clip(xe, -5, 5)
                fe = evaluate(xe)
                if fe < fr:
                    verts[-1] = (xe, fe)
                else:
                    verts[-1] = (xr, fr)
            elif fr < second_worst[1]:
                verts[-1] = (xr, fr)
            else:
                # Contraction
                xc = centroid + 0.5 * (worst[0] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = evaluate(xc)
                if fc < worst[1]:
                    verts[-1] = (xc, fc)
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        verts[i] = (verts[0][0] + 0.5 * (verts[i][0] - verts[0][0]),
                                     evaluate(verts[0][0] + 0.5 * (verts[i][0] - verts[0][0])))

        return

    def _check_deadline(self, deadline):
        return time.time() > deadline - 5
# EVOLVE-BLOCK-END
