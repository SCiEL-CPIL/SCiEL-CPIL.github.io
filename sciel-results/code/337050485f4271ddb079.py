# EVOLVE-BLOCK-START
import numpy as np
import time


def _non_dominated_mask(F):
    """Boolean mask: True for points not dominated by any other point."""
    n = F.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    # F[i] <= F[j] elementwise AND F[i] < F[j] in at least one coord => i dominates j
    le = F[:, None, :] <= F[None, :, :]          # (n,n,m)
    lt = F[:, None, :] < F[None, :, :]           # (n,n,m)
    dom = le & lt.any(axis=2)                    # (n,n): dom[i,j] = i dominates j
    dominated = dom.any(axis=0)                  # j dominated if any i dominates j
    return ~dominated


def _hv_contribution_sort(F, ref):
    """Return points sorted so that later points are 'worse' (for front thinning)."""
    order = np.lexsort((F[:, -1], F[:, -2], F[:, -1] if F.shape[1] > 1 else F[:, 0]))
    return order


class Optimizer:
    """Multi-objective CMA-ES with manifold decomposition for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # 20s safety margin under 300s
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(0)

        # Archive of evaluated points (x, f)
        X_arch = []
        F_arch = []

        def eval_point(x):
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            X_arch.append(x)
            F_arch.append(f)
            return f

        # ---- Phase 0: initial random spread ----
        n_init = min(20, budget)
        X0 = lo + span * rng.random((n_init, dim))
        F0 = np.array([eval_point(x) for x in X0])
        used = n_init

        # ---- CMA-ES distributions ----
        # Each distribution: mean (dim), sigma (scalar), C (dim x dim)
        n_dist = 4
        dists = []
        for i in range(n_dist):
            m = lo + span * rng.random(dim)
            sigma = 0.2 * np.mean(span)
            C = np.eye(dim)
            dists.append({'m': m, 'sigma': sigma, 'C': C, 'step': 0.5})

        # ---- Manifold decomposition: 2D simplex -> 12D via affine basis ----
        # Basis: center + sum_k alpha_k * v_k, alpha on 2-simplex (alpha>=0, sum=1)
        # v_k are search directions, learned from the archive.
        basis = {'center': None, 'V': None, 'alpha': None, 'sigma': 0.1}

        def update_basis():
            if len(F_arch) < 6:
                return
            F = np.array(F_arch)
            X = np.array(X_arch)
            # Use the non-dominated points to build the manifold basis
            nd = _non_dominated_mask(F)
            if nd.sum() < 2:
                nd = np.ones(len(F), dtype=bool)
            Xnd = X[nd]
            center = Xnd.mean(axis=0)
            # PCA-like directions from the non-dominated set
            Xc = Xnd - center
            try:
                U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            except np.linalg.LinAlgError:
                Vt = np.eye(dim)
            # Take top 2 directions (the manifold is 2D)
            V = Vt[:min(2, dim)].T  # (dim, 2)
            basis['center'] = center
            basis['V'] = V
            basis['sigma'] = max(0.05, 0.1 * np.mean(S[:2]) / (np.mean(span) + 1e-12))

        def simplex_to_x(alpha):
            # alpha: (2,) on simplex -> x in R^dim
            x = basis['center'] + basis['V'] @ alpha
            return x

        # Small population of simplex parameters for manifold search
        n_man = 4
        alphas = []
        for _ in range(n_man):
            a = rng.random(2)
            a = a / (a.sum() + 1e-12)
            alphas.append(a)

        # ---- Main loop ----
        # Alternate between CMA-ES steps and manifold steps.
        # Each iteration consumes ~1-2 evaluations.
        cma_counter = 0
        man_counter = 0
        lam = max(4, dim // 2)  # CMA-ES lambda

        while used < budget:
            if time.time() > deadline:
                break

            # Decide mode: 60% CMA-ES, 40% manifold
            if (cma_counter % 5) < 3 and len(dists) > 0:
                # ---- CMA-ES step ----
                d = dists[cma_counter % len(dists)]
                cma_counter += 1
                # Generate lambda samples
                Z = rng.standard_normal((lam, dim))
                L = np.linalg.cholesky(d['C'] + 1e-8 * np.eye(dim))
                samples = d['m'] + d['sigma'] * (Z @ L.T)
                samples = np.clip(samples, lo, hi)
                fs = np.array([eval_point(s) for s in samples])
                used += lam
                # Update mean toward weighted best (by scalarized objective)
                # Use a weighted scalarization: sum of normalized objectives
                fmin = fs.min(axis=0)
                fmax = fs.max(axis=0)
                fspan = (fmax - fmin) + 1e-12
                norm = (fs - fmin) / fspan
                scalar = norm.sum(axis=1)
                # Weighted mean: top half weighted by rank
                order = np.argsort(scalar)
                w = np.zeros(lam)
                top = lam // 2
                w[order[:top]] = np.arange(top, 0, -1)
                w = w / (w.sum() + 1e-12)
                new_m = (samples * w[:, None]).sum(axis=0)
                # Update CMA (simplified: move mean, adapt sigma)
                d['m'] = np.clip(d['m'] + d['step'] * (new_m - d['m']), lo, hi)
                # Adapt sigma: if mean moved, keep; else shrink slightly
                move = np.linalg.norm(new_m - d['m'])
                d['sigma'] = max(0.02 * np.mean(span), d['sigma'] * (0.95 + 0.1 * (move > 1e-8)))
            else:
                # ---- Manifold step ----
                man_counter += 1
                update_basis()
                if basis['center'] is None:
                    # Fallback: random
                    x = lo + span * rng.random(dim)
                    f = eval_point(x)
                    used += 1
                    continue
                # Mutate a random simplex parameter
                i = man_counter % n_man
                a = alphas[i] + basis['sigma'] * rng.standard_normal(2)
                a = np.clip(a, 0, None)
                s = a.sum()
                if s < 1e-12:
                    a = np.ones(2) / 2
                else:
                    a = a / s
                alphas[i] = a
                x = simplex_to_x(a)
                x = np.clip(x, lo, hi)
                f = eval_point(x)
                used += 1

        # Final: ensure we have a valid return
        if len(X_arch) > 0:
            return np.array(X_arch)
        return np.array([lo + span * 0.5])
# EVOLVE-BLOCK-END
