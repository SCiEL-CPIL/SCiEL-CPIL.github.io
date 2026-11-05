"""Combined approach: smooth surrogate optimization with diverse initialization
and local refinement, plus a convex QP-style refinement via SLSQP.

Key structure:
    value = abs(2*N*max(convolve(h,h)) / sum(h)**2)
is invariant under h -> c*h, so we fix sum(h)=1 and minimize 2*N*max(conv).
Since conv[0] = sum(h^2) >= (sum h)^2 / N = 1/N (Cauchy-Schwarz), value >= 2.

We use a smooth log-sum-exp surrogate for the max, with diverse initializations
(Gaussian mixture, block structures, filtered randoms) and Nelder-Mead refinement.
We also attempt an SLSQP-based convex refinement to squeeze out further gains.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 150.0
_MARGIN = 8.0


def _make_grid():
    return np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)


def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _smooth_objective(h, beta=80.0):
    """Smooth surrogate using log-sum-exp for max in autocorrelation."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + log_sum / beta
    val = 2.0 * N * soft_max / (s * s)
    return abs(val)


def _smooth_grad(h, beta=80.0):
    """Gradient of the smooth objective w.r.t. h."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.zeros(N), 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    soft_max = m + np.log(np.sum(weights)) / beta
    obj = abs(2.0 * N * soft_max / (s * s))
    
    # d(conv[k])/d(h[j]) = 2*h[k-j] if 0 <= k-j < N, else 0
    # d(soft_max)/d(h[j]) = sum_k weights[k] * 2*h[k-j] (valid k)
    # = 2 * sum_{k: 0<=k-j<2N-1} weights[k] * h[k-j]
    
    # Efficient computation via correlation:
    # For each j, we need sum over m where m = k-j, so k = m+j
    # d(soft_max)/d(h[j]) = 2 * sum_{m=0}^{N-1} weights[m+j] * h[m] (where m+j < 2N-1)
    
    grad_soft = np.zeros(N)
    for j in range(N):
        # k ranges from j to j+N-1, but must be < 2N-1
        k_start = j
        k_end = min(j + N, 2 * N - 1)
        if k_start < k_end:
            grad_soft[j] = 2.0 * np.dot(weights[k_start:k_end], h[:k_end - k_start])
    
    # d(obj)/d(h[j]) = 2N * (grad_soft[j]/s^2 - 2*soft_max/s^3)
    grad = 2.0 * N * (grad_soft / (s * s) - 2.0 * soft_max / (s * s * s))
    return grad, obj


def _gaussian_mixture(x, mu, sig):
    """Sum of Gaussian bumps, normalized so that sum == 1."""
    g = np.zeros(N)
    for m, s in zip(mu, sig):
        g += np.exp(-0.5 * ((x - m) / s) ** 2)
    s = g.sum()
    if s <= 0 or not np.isfinite(s):
        return None
    return g / s


def _block_structure(x, n_blocks=4, width=0.06):
    """Block-like structure with smooth transitions."""
    g = np.zeros(N)
    block_centers = np.linspace(-0.2, 0.2, n_blocks)
    for c in block_centers:
        g += np.exp(-0.5 * ((x - c) / width) ** 2)
    s = g.sum()
    if s <= 0 or not np.isfinite(s):
        return None
    return g / s


def _filtered_random(rng, seed_offset=0):
    """Random profile with Gaussian filtering for smoothness."""
    r = np.random.default_rng(42 + seed_offset)
    raw = r.normal(0, 1, N)
    # Apply a simple Gaussian filter
    kernel_size = 15
    kernel = np.exp(-0.5 * (np.arange(kernel_size) - kernel_size // 2) ** 2 / 4.0)
    kernel /= kernel.sum()
    filtered = np.convolve(raw, kernel, mode='same')
    # Make positive and normalize
    filtered = filtered - filtered.min() + 0.1
    s = filtered.sum()
    if s <= 0 or not np.isfinite(s):
        return None
    return filtered / s


def _construct_baseline():
    """A smooth, strictly-positive, centrally-peaked profile (Gaussian mixture)."""
    x = _make_grid()
    mu = np.array([0.0, -0.09, 0.09])
    sig = np.array([0.055, 0.07, 0.07])
    g = _gaussian_mixture(x, mu, sig)
    if g is None or not np.all(np.isfinite(g)):
        g = np.ones(N) / N
    return g


def _diverse_initializations():
    """Generate diverse initial points for optimization."""
    x = _make_grid()
    inits = []
    
    # Gaussian mixture variants
    mu_sets = [
        np.array([0.0, -0.09, 0.09]),
        np.array([0.0, -0.12, 0.12]),
        np.array([0.0, -0.06, 0.06]),
        np.array([0.0]),
        np.array([0.0, -0.15, 0.15]),
    ]
    sig_sets = [
        np.array([0.055, 0.07, 0.07]),
        np.array([0.06, 0.08, 0.08]),
        np.array([0.05, 0.06, 0.06]),
        np.array([0.06]),
        np.array([0.05, 0.09, 0.09]),
    ]
    for mu, sig in zip(mu_sets, sig_sets):
        g = _gaussian_mixture(x, mu, sig)
        if g is not None and np.all(np.isfinite(g)):
            inits.append(g)
    
    # Block structures
    for n_blocks in [3, 4, 5]:
        for width in [0.05, 0.07, 0.09]:
            g = _block_structure(x, n_blocks, width)
            if g is not None and np.all(np.isfinite(g)):
                inits.append(g)
    
    # Filtered randoms
    for i in range(5):
        g = _filtered_random(None, seed_offset=i)
        if g is not None and np.all(np.isfinite(g)):
            inits.append(g)
    
    # Uniform
    inits.append(np.ones(N) / N)
    
    return inits


def construct():
    t0 = time.time()
    best = _construct_baseline()
    best_v = _value(best)
    
    deadline = t0 + _DEADLINE - _MARGIN
    
    # Phase 1: Nelder-Mead on Gaussian mixture parameters with diverse restarts
    try:
        x = _make_grid()
        
        def params_to_h(p):
            mu = p[:3]
            sig = np.exp(p[3:6])
            return _gaussian_mixture(x, mu, sig)
        
        def objective(p):
            h = params_to_h(p)
            if h is None:
                return 1e6
            return _smooth_objective(h, beta=80.0)
        
        p0 = np.array([0.0, -0.09, 0.09,
                       np.log(0.055), np.log(0.07), np.log(0.07)])
        
        rng = np.random.default_rng(12345)
        restarts = [p0]
        for _ in range(10):
            jitter = rng.normal(0.0, 0.025, size=3)
            sjitter = rng.normal(0.0, 0.25, size=3)
            restarts.append(np.array(
                [p0[0] + jitter[0], p0[1] + jitter[1], p0[2] + jitter[2],
                 np.log(0.055) + sjitter[0], np.log(0.07) + sjitter[1],
                 np.log(0.07) + sjitter[2]]))
        
        for pstart in restarts:
            if time.time() > deadline:
                break
            res = minimize(objective, pstart, method="Nelder-Mead",
                           options={"maxiter": 600, "xatol": 1e-8,
                                    "fatol": 1e-10, "adaptive": True})
            if res.x is not None:
                h = params_to_h(res.x)
                if h is not None and np.all(np.isfinite(h)):
                    v = _value(h)
                    if v < best_v - 1e-12:
                        best_v = v
                        best = h
            if time.time() > deadline:
                break
    except Exception:
        pass
    
    # Phase 2: Direct gradient-based refinement on diverse initializations
    if time.time() < deadline:
        inits = _diverse_initializations()
        for h0 in inits:
            if time.time() > deadline:
                break
            try:
                res = minimize(_smooth_objective, h0, jac=lambda h: _smooth_grad(h, beta=80.0)[0],
                               method="L-BFGS-B",
                               options={"maxiter": 300, "gtol": 1e-8, "maxfun": 500})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    # Normalize to sum=1
                    h_norm = res.x / res.x.sum() if abs(res.x.sum()) > 1e-12 else res.x
                    v = _value(h_norm)
                    if v < best_v - 1e-12:
                        best_v = v
                        best = h_norm
            except Exception:
                pass
    
    # Phase 3: SLSQP convex refinement (minimize ||h||^2 subject to constraints)
    if time.time() < deadline:
        try:
            h0 = best / best.sum() if abs(best.sum()) > 1e-12 else best
            
            def slsqp_objective(h):
                return float(np.dot(h, h))
            
            def slsqp_grad(h):
                return 2.0 * h
            
            # Constraint: sum(h) = 1
            cons = [{'type': 'eq', 'fun': lambda h: h.sum() - 1.0,
                     'jac': lambda h: np.ones(N)}]
            
            # Inequality constraints: conv(h,h)[k] <= t for some t
            # We'll use a soft constraint approach
            def conv_constraint(h):
                conv = np.convolve(h, h)
                return float(np.max(conv)) - best_v / (2.0 * N) * h.sum()**2
            
            cons.append({'type': 'ineq', 'fun': conv_constraint})
            
            res = minimize(slsqp_objective, h0, jac=slsqp_grad,
                           method="SLSQP", constraints=cons,
                           options={"maxiter": 200, "ftol": 1e-10})
            if res.x is not None and np.all(np.isfinite(res.x)):
                h_norm = res.x / res.x.sum() if abs(res.x.sum()) > 1e-12 else res.x
                v = _value(h_norm)
                if v < best_v - 1e-12:
                    best_v = v
                    best = h_norm
        except Exception:
            pass
    
    # Final safety checks
    best = np.asarray(best, dtype=float)
    if not np.all(np.isfinite(best)) or best.sum() == 0:
        best = np.ones(N)
    
    # Numerical guard
    if _value(best) > 2.0 + 1e-9:
        best = np.ones(N)
    
    return best


# Self-check
if __name__ == "__main__":
    _h = construct()
    _s = _h.sum()
    _conv = np.convolve(_h, _h, mode="full")
    print("value =", abs(2.0 * N * np.max(_conv) / (_s * _s)),
          "sum =", _s, "finite =", bool(np.all(np.isfinite(_h))))
