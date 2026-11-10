# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(x, n=64, y_count=20000):
    """Compute area of union of n triangles using quadrature."""
    a = (np.arange(n) + 1) / n
    w = 1.0 / n
    ys = (np.arange(y_count) + 0.5) / y_count  # (M,)
    # L_i(y) = x_i + a_i * y  -> shape (n, M)
    L = x[:, None] + a[:, None] * ys[None, :]
    # Sort L along axis 0
    L_sorted = np.sort(L, axis=0)
    # Consecutive gaps
    gaps = L_sorted[1:] - L_sorted[:-1]  # (n-1, M)
    # Union length = w + sum(min(gap, w))
    union_lengths = w + np.sum(np.minimum(gaps, w), axis=0)  # (M,)
    area = np.mean(union_lengths)
    return area

def _compute_area_smooth(x, beta, n=64, y_count=20000):
    """Compute smooth area using log-sum-exp approximation of min."""
    a = (np.arange(n) + 1) / n
    w = 1.0 / n
    ys = (np.arange(y_count) + 0.5) / y_count
    L = x[:, None] + a[:, None] * ys[None, :]
    L_sorted = np.sort(L, axis=0)
    gaps = L_sorted[1:] - L_sorted[:-1]
    # soft_min(gap, w) = -logaddexp(-beta*gap, -beta*w) / beta
    # = w - softplus(beta*(w - gap)) / beta
    # For numerical stability: soft_min = w - (1/beta)*log(1 + exp(beta*(w-gap)))
    # Use the stable form
    d = beta * (w - gaps)  # (n-1, M)
    # log(1+exp(d)) computed stably
    log1pexp = np.where(d > 0, d + np.log1p(np.exp(-d)), np.log1p(np.exp(d)))
    soft_min = w - log1pexp / beta
    union_lengths = w + np.sum(soft_min, axis=0)
    area = np.mean(union_lengths)
    return area

def _area_gradient_smooth(x, beta, n=64, y_count=20000):
    """Compute gradient of smooth area w.r.t. x."""
    a = (np.arange(n) + 1) / n
    w = 1.0 / n
    ys = (np.arange(y_count) + 0.5) / y_count
    L = x[:, None] + a[:, None] * ys[None, :]
    L_sorted = np.sort(L, axis=0)
    gaps = L_sorted[1:] - L_sorted[:-1]
    d = beta * (w - gaps)
    log1pexp = np.where(d > 0, d + np.log1p(np.exp(-d)), np.log1p(np.exp(d)))
    soft_min = w - log1pexp / beta
    # d(soft_min)/d(gap) = -exp(d)/(1+exp(d)) = -sigmoid(d)
    # sigmoid(d) = 1/(1+exp(-d))
    sig = 1.0 / (1.0 + np.exp(-d))  # (n-1, M)
    # d(soft_min)/d(gap) = -sig
    # gap_j = L_sorted[j+1] - L_sorted[j]
    # d(gap_j)/d(L_sorted[j+1]) = 1, d(gap_j)/d(L_sorted[j]) = -1
    # So d(soft_min_j)/d(L_sorted[j+1]) = -sig_j, d(soft_min_j)/d(L_sorted[j]) = sig_j
    # Union length = w + sum_j soft_min_j
    # d(union)/d(L_sorted[k]) = sum_{j: k=j+1} (-sig_j) + sum_{j: k=j} sig_j
    # = sig_{k-1} - sig_k (with boundaries: sig_0 contributes to L_sorted[1], sig_{n-2} to L_sorted[n-1])
    # Let's build the gradient vector for L_sorted
    grad_Ls = np.zeros_like(L_sorted)  # (n, M)
    # For each gap j (0-indexed in gaps, corresponding to L_sorted[j] and L_sorted[j+1]):
    # contribution to L_sorted[j]: +sig_j
    # contribution to L_sorted[j+1]: -sig_j
    grad_Ls[:-1] += sig  # L_sorted[0..n-2] get +sig
    grad_Ls[1:] -= sig   # L_sorted[1..n-1] get -sig
    # Now we need to map back to original L via the sort permutation
    # L_sorted[i, m] = L[perm[i, m], m]
    # d(area)/d(L[perm[i,m], m]) = grad_Ls[i, m]
    # Use scatter: for each m, scatter grad_Ls[:, m] to positions perm[:, m]
    perm = np.argsort(L, axis=0)  # (n, M)
    grad_L = np.zeros_like(L)
    # For each column m: grad_L[perm[i,m], m] = grad_Ls[i, m]
    # Use advanced indexing
    rows = np.arange(n)[:, None]
    cols = perm
    grad_L[rows, cols] = grad_Ls  # This doesn't work directly; need scatter
    # Better: for each m, grad_L[:, m][perm[:, m]] = grad_Ls[:, m]
    # Vectorized scatter
    grad_L = np.zeros_like(L)
    np.add.at(grad_L, (rows, cols), grad_Ls)
    # d(area)/d(x_i) = mean_m(d(L[i,m])/dx_i) = mean_m(a_i) = a_i
    # Wait: L[i,m] = x_i + a_i * y_m, so dL/dx_i = 1
    # So d(area)/d(x_i) = mean_m(grad_L[i, m])
    grad_x = np.mean(grad_L, axis=1)  # (n,)
    return grad_x

def _area_and_grad_smooth(x, beta, n=64, y_count=20000):
    """Compute smooth area and its gradient."""
    a = (np.arange(n) + 1) / n
    w = 1.0 / n
    ys = (np.arange(y_count) + 0.5) / y_count
    L = x[:, None] + a[:, None] * ys[None, :]
    L_sorted = np.sort(L, axis=0)
    gaps = L_sorted[1:] - L_sorted[:-1]
    d = beta * (w - gaps)
    # Stable sigmoid
    sig = np.where(d >= 0, 1.0 / (1.0 + np.exp(-d)), np.exp(d) / (1.0 + np.exp(d)))
    log1pexp = np.where(d > 0, d + np.log1p(np.exp(-d)), np.log1p(np.exp(d)))
    soft_min = w - log1pexp / beta
    union_lengths = w + np.sum(soft_min, axis=0)
    area = np.mean(union_lengths)
    # Gradient
    grad_Ls = np.zeros_like(L_sorted)
    grad_Ls[:-1] += sig
    grad_Ls[1:] -= sig
    perm = np.argsort(L, axis=0)
    grad_L = np.zeros_like(L)
    rows = np.arange(n)[:, None]
    cols = perm
    np.add.at(grad_L, (rows, cols), grad_Ls)
    grad_x = np.mean(grad_L, axis=1)
    return area, grad_x

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem. For a fixed n, choose
    base positions x = (x_0, ..., x_{n-1}). For position index i (0-based) let
    a_i = (i + 1) / n and define the triangle R_i with vertices
        (x_i, 0), (x_i + 1/n, 0), (x_i + a_i, 1).
    Every triangle has base width 1/n on the line y = 0 and its apex on the line
    y = 1. The objective is to choose x so that the AREA of the union of the n
    triangles R_i is as small as possible. Here n = 64.

    Uses L-BFGS-B with a smooth log-sum-exp surrogate for the non-smooth min,
    with annealing on the soft-min parameter beta.

    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    y_count = 20000
    deadline = time.time() + 620  # Safety margin before 630s limit

    # Initial guess: Keich construction
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)

    # Also try a simple spread as alternative start
    x0_alt = np.linspace(-0.5, 0.5, n)

    best_x = x0.copy()
    best_area = _compute_area(x0, n, y_count)

    # Annealing schedule: start with low beta (smooth), increase to high beta (sharp)
    beta_schedule = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0]

    current_x = x0.copy()
    for beta in beta_schedule:
        if time.time() > deadline:
            break

        def obj_grad(x):
            area, grad = _area_and_grad_smooth(x, beta, n, y_count)
            return area, grad

        # L-BFGS-B optimization
        try:
            res = minimize(
                obj_grad,
                current_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            if res.success or res.fun < best_area:
                # Evaluate true (non-smooth) area for this candidate
                true_area = _compute_area(res.x, n, y_count)
                if true_area < best_area:
                    best_area = true_area
                    best_x = res.x.copy()
                current_x = res.x.copy()
        except Exception:
            pass

        # Check deadline
        if time.time() > deadline:
            break

    # Final polish: run a few more iterations with high beta if time permits
    if time.time() < deadline - 10:
        try:
            def obj_grad_final(x):
                area, grad = _area_and_grad_smooth(x, 1000.0, n, y_count)
                return area, grad

            res = minimize(
                obj_grad_final,
                best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 2000}
            )
            true_area = _compute_area(res.x, n, y_count)
            if true_area < best_area:
                best_area = true_area
                best_x = res.x.copy()
        except Exception:
            pass

    # Ensure finite values
    if not np.all(np.isfinite(best_x)):
        best_x = x0.copy()

    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
