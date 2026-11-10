# EVOLVE-BLOCK-START
import time
import numpy as np


def _union_length_1d(a, s, n, beta):
    """
    Smooth surrogate for the 1-D union length of n sheared intervals.
    Interval r: [a[r] + s[r]/n, a[r] + (s[r]+1)/n] for s in {0..n-1}.
    Uses a soft-min over the left endpoints and soft-max over the right endpoints,
    with a per-interval length floor to keep the objective smooth and bounded.
    """
    left = a + s / n
    right = a + (s + 1.0) / n
    # Soft-min of left endpoints
    Lm = np.max(left)
    soft_min = Lm - np.log(np.sum(np.exp((Lm - left) / beta)) * beta)
    # Soft-max of right endpoints
    Rm = np.min(right)
    soft_max = Rm + np.log(np.sum(np.exp((right - Rm) / beta)) * beta)
    union = soft_max - soft_min
    # Floor: each interval contributes at least 1/n of length (union >= 1/n * n? no)
    # The true union is >= max interval length = 1/n, but can be up to (max s + 1)/n = 1.
    # Add a gentle floor to avoid degenerate collapse: union >= 1/n is automatic.
    return union


def _objective_1d(a, s, n, beta):
    """Objective to minimize: the smooth 1-D union length."""
    return _union_length_1d(a, s, n, beta)


def _optimize_1d(n, budget, seed):
    """
    Optimize n base positions a[0..n-1] to minimize the 1-D union length of
    the sheared intervals [a[r] + s/n, a[r] + (s+1)/n].
    Uses multi-start L-BFGS-B with a smooth soft-min/soft-max surrogate.
    """
    s = np.arange(n, dtype=np.float64)
    rng = np.random.default_rng(seed)
    best_a = None
    best_val = np.inf

    # Multi-start with several initializations
    n_starts = 6
    for start in range(n_starts):
        # Start near Keich-like or random compact positions
        if start == 0:
            # Keich-like binary construction
            pos = int(np.log2(n))
            j_terms = np.arange(1, pos + 1)
            keich = np.empty(n, dtype=np.float64)
            for idx in range(n):
                eps = np.array(list(str(bin(idx))[2:]), dtype=int)
                eps = np.pad(eps, (pos - len(eps), 0), "constant")
                keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
            a0 = keich.copy()
        elif start == 1:
            # Compact cluster
            a0 = rng.uniform(0, 0.1, n)
        elif start == 2:
            # Spread
            a0 = np.linspace(0, 1, n)
        else:
            # Random
            a0 = rng.uniform(0, 1, n)

        # L-BFGS-B optimization with smooth surrogate
        # Use a moderately smooth beta, then refine with smaller beta
        for beta in [0.1, 0.05, 0.02, 0.01]:
            def obj(a):
                return _objective_1d(a, s, n, beta)

            def grad(a):
                # Numerical gradient (central differences)
                h = 1e-6
                g = np.zeros_like(a)
                for k in range(n):
                    a[k] += h
                    f_plus = obj(a)
                    a[k] -= 2 * h
                    f_minus = obj(a)
                    a[k] += h
                    g[k] = (f_plus - f_minus) / (2 * h)
                return g

            # Use scipy if available, else simple gradient descent
            try:
                from scipy.optimize import minimize
                res = minimize(
                    obj, a0, jac=grad, method="L-BFGS-B",
                    options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10}
                )
                a_opt = res.x
                val = res.fun
            except Exception:
                # Fallback: simple gradient descent
                a_opt = a0.copy()
                lr = 0.01
                for it in range(500):
                    g = grad(a_opt)
                    a_opt = a_opt - lr * g
                    val = obj(a_opt)

            if val < best_val:
                best_val = val
                best_a = a_opt.copy()
            a0 = a_opt  # warm start for next beta

    return best_a, best_val


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.

    The 3D union volume factorizes as (1D x-union length) * (1D y-union length),
    because each tube is a product of an x-interval (function of i) and a y-interval
    (function of j). We optimize the 1D base positions to minimize the union length
    of the sheared intervals, using a smooth soft-min/soft-max surrogate with
    L-BFGS-B, then reuse the optimal 1D profile for both axes.
    """
    N = 8
    t0 = time.time()
    deadline = 600.0 - 30.0  # safety margin

    # Optimize the 1D profile
    a_opt, val = _optimize_1d(N, budget=deadline - (time.time() - t0), seed=42)

    if a_opt is None:
        # Fallback to Keich
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        a_opt = keich

    # Build the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = a_opt[i_flat]
    base_y = a_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    # Validate
    if not np.all(np.isfinite(positions)):
        # Fallback
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)

    return positions


# EVOLVE-BLOCK-END
