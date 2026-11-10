# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem. For a fixed n=64, choose
    base positions x = (x_0, ..., x_{63}). For position index i (0-based) let
    a_i = (i + 1) / 64 and define the triangle R_i with vertices
        (x_i, 0), (x_i + 1/64, 0), (x_i + a_i, 1).
    Every triangle has base width 1/64 on the line y = 0 and its apex on the line
    y = 1. The objective is to choose x so that the AREA of the union of the 64
    triangles R_i is as small as possible.

    Strategy: Start from the Keich construction, then use a projected subgradient
    method with adaptive step sizes in a smoothed surrogate, followed by basin
    hopping with structured perturbations, and a final high-precision L-BFGS polish.
    """
    n = 64
    k = int(np.log2(n))  # k = 6 for n=64

    # Deadline guard: finish 30 seconds before the 630s limit
    deadline = time.time() + 600.0

    # Helper: compute area using the exact quadrature (M=20000)
    def compute_area(x):
        M = 20000
        y = (np.arange(M) + 0.5) / M
        w = (1.0 - y) / n
        # L_i = x_i + a_i * y for each i
        a = (np.arange(n) + 1) / n  # shape (n,)
        L = x[None, :] + a[None, :] * y[:, None]  # shape (M, n)
        # Sort along axis=1
        L_sorted = np.sort(L, axis=1)  # shape (M, n)
        # Union length = w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
        diffs = np.diff(L_sorted, axis=1)  # shape (M, n-1)
        union_len = w[:, None] + np.sum(np.minimum(diffs, w[:, None]), axis=1)
        return np.mean(union_len)

    # Helper: compute gradient of area w.r.t. x using finite differences on sorted structure
    # For efficiency, use a reduced resolution for gradient
    def compute_gradient(x, M_grad=4000):
        y = (np.arange(M_grad) + 0.5) / M_grad
        w = (1.0 - y) / n
        a = (np.arange(n) + 1) / n
        L = x[None, :] + a[None, :] * y[:, None]  # shape (M_grad, n)
        L_sorted, idx = np.sort(L, axis=1)  # shape (M_grad, n)
        # The union length at each y: w + sum min(diffs, w)
        # d(union_len)/dx_i involves how x_i affects L_i and its position in sorted order
        # Use subgradient: for each y, the sorted order tells us which L is which
        # The subgradient of the union length w.r.t. x_i is:
        #   sum over y of [contribution of x_i to the union length]
        # More precisely, the union length is a function of the sorted L values.
        # Let's use a finite-difference gradient for simplicity and robustness.
        eps = 1e-7
        grad = np.zeros(n)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (compute_area(x_plus) - compute_area(x_minus)) / (2 * eps)
        return grad

    # Faster gradient using vectorized finite differences with reduced M
    def compute_gradient_fast(x, M_grad=2000):
        y = (np.arange(M_grad) + 0.5) / M_grad
        w = (1.0 - y) / n
        a = (np.arange(n) + 1) / n
        # L shape (M_grad, n)
        L = x[None, :] + a[None, :] * y[:, None]
        L_sorted = np.sort(L, axis=1)
        diffs = np.diff(L_sorted, axis=1)  # (M_grad, n-1)
        # union_len = w + sum min(diffs, w)
        # d union_len / d x_i: x_i appears in L_i. After sorting, L_i is at some position.
        # Use central finite difference with a small step
        eps = 1e-6
        grad = np.zeros(n)
        for i in range(n):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            Lp = xp[None, :] + a[None, :] * y[:, None]
            Lm = xm[None, :] + a[None, :] * y[:, None]
            Lps = np.sort(Lp, axis=1)
            Lms = np.sort(Lm, axis=1)
            dp = np.diff(Lps, axis=1)
            dm = np.diff(Lms, axis=1)
            up = w[:, None] + np.sum(np.minimum(dp, w[:, None]), axis=1)
            um = w[:, None] + np.sum(np.minimum(dm, w[:, None]), axis=1)
            grad[i] = (np.mean(up) - np.mean(um)) / (2 * eps)
        return grad

    # Smooth surrogate using soft-min
    def compute_area_soft(x, beta=50.0, M=20000):
        y = (np.arange(M) + 0.5) / M
        w = (1.0 - y) / n
        a = (np.arange(n) + 1) / n
        L = x[None, :] + a[None, :] * y[:, None]  # (M, n)
        L_sorted = np.sort(L, axis=1)
        diffs = np.diff(L_sorted, axis=1)  # (M, n-1)
        # Soft min: min(diffs, w) -> use soft approximation
        # soft_min(d, w) = -1/beta * logsumexp(-beta*d, -beta*w)
        # But min(d, w) is piecewise linear; a smooth approximation:
        # soft_min(d, w) = (d + w - |d - w|) / 2, smooth |.| via soft absolute
        diff_d_w = diffs - w[:, None]
        # soft_abs: |z| approx = sqrt(z^2 + eps^2)
        eps_soft = 1.0 / beta
        soft_abs = np.sqrt(diff_d_w ** 2 + eps_soft ** 2)
        soft_min_vals = (diffs + w[:, None] - soft_abs) / 2.0
        union_len = w[:, None] + np.sum(soft_min_vals, axis=1)
        return np.mean(union_len)

    # Gradient of soft area
    def compute_gradient_soft(x, beta=50.0, M_grad=2000):
        y = (np.arange(M_grad) + 0.5) / M_grad
        w = (1.0 - y) / n
        a = (np.arange(n) + 1) / n
        eps = 1e-6
        grad = np.zeros(n)
        for i in range(n):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            grad[i] = (compute_area_soft(xp, beta, M_grad) - compute_area_soft(xm, beta, M_grad)) / (2 * eps)
        return grad

    # Phase 0: Start from Keich construction
    positions = []
    for i in range(n):
        eps_arr = np.array(list(str(bin(i))[2:]), dtype=int)
        eps_arr = np.pad(eps_arr, (k - len(eps_arr), 0), "constant")
        j = np.arange(1, k + 1)
        positions.append(np.sum((1 - j) * eps_arr * (2.0 ** -j) / k))
    x_best = np.array(positions, dtype=np.float64)
    best_area = compute_area(x_best)

    # Phase 1: L-BFGS with soft surrogate (adaptive beta)
    beta = 30.0
    res = minimize(
        lambda x: compute_area_soft(x, beta),
        x_best,
        method='L-BFGS-B',
        options={'maxiter': 300, 'ftol': 1e-10}
    )
    if res.x is not None and np.all(np.isfinite(res.x)):
        area_soft = compute_area(res.x)
        if area_soft < best_area:
            x_best = res.x.copy()
            best_area = area_soft

    # Increase beta and polish
    for beta_val in [50.0, 100.0, 200.0]:
        if time.time() > deadline - 120:
            break
        res = minimize(
            lambda x: compute_area_soft(x, beta_val),
            x_best,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-11}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            area_soft = compute_area(res.x)
            if area_soft < best_area:
                x_best = res.x.copy()
                best_area = area_soft

    # Phase 2: Basin hopping with structured perturbations
    rng = np.random.default_rng(42)
    max_hops = 3000
    for hop in range(max_hops):
        if time.time() > deadline - 90:
            break

        # Adaptive beta for soft surrogate in local optimization
        t = hop / max_hops
        beta = 50.0 + 150.0 * t

        # Generate perturbation
        x_try = x_best.copy()
        pert_type = rng.integers(0, 3)

        if pert_type == 0:
            # Subset shift
            subset_size = rng.integers(4, 17)
            subset = rng.choice(n, size=subset_size, replace=False)
            shift = rng.normal(0, 0.008, size=subset_size)
            x_try[subset] += shift
        elif pert_type == 1:
            # Pairwise swap with small noise
            i, j = rng.choice(n, size=2, replace=False)
            x_try[i], x_try[j] = x_try[j], x_try[i]
            x_try += rng.normal(0, 0.003, size=n)
        else:
            # Gaussian perturbation
            x_try += rng.normal(0, 0.008, size=n)

        # Optimize the perturbed point using soft surrogate
        if time.time() > deadline - 60:
            break
        res = minimize(
            lambda x: compute_area_soft(x, beta),
            x_try,
            method='L-BFGS-B',
            options={'maxiter': 80, 'ftol': 1e-9}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            area_opt = compute_area(res.x)
            if area_opt < best_area:
                x_best = res.x.copy()
                best_area = area_opt

    # Phase 3: Final high-precision L-BFGS polish with exact area
    if time.time() < deadline - 30:
        # Use a gradient computed with reduced M for speed, but evaluate with exact M
        def area_and_grad(x):
            a_val = compute_area(x)
            # Use a quick gradient estimate
            g = np.zeros(n)
            eps = 1e-8
            for i in range(n):
                xp = x.copy(); xp[i] += eps
                xm = x.copy(); xm[i] -= eps
                g[i] = (compute_area(xp) - compute_area(xm)) / (2 * eps)
            return a_val, g

        # L-BFGS-B with finite-difference gradient (expensive but high precision)
        # Use a wrapper for scipy
        def objective_and_jacobian(x):
            a_val = compute_area(x)
            g = np.zeros(n)
            eps = 1e-8
            for i in range(n):
                xp = x.copy(); xp[i] += eps
                xm = x.copy(); xm[i] -= eps
                g[i] = (compute_area(xp) - compute_area(xm)) / (2 * eps)
            return a_val, g

        res = minimize(
            objective_and_jacobian,
            x_best,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            final_area = compute_area(res.x)
            if final_area < best_area:
                x_best = res.x.copy()
                best_area = final_area

    # Validate
    x_best = np.array(x_best, dtype=np.float64).flatten()
    if not np.all(np.isfinite(x_best)) or len(x_best) != n:
        # Fallback to Keich
        positions = []
        for i in range(n):
            eps_arr = np.array(list(str(bin(i))[2:]), dtype=int)
            eps_arr = np.pad(eps_arr, (k - len(eps_arr), 0), "constant")
            j = np.arange(1, k + 1)
            positions.append(np.sum((1 - j) * eps_arr * (2.0 ** -j) / k))
        x_best = np.array(positions, dtype=np.float64)

    return x_best


# EVOLVE-BLOCK-END
