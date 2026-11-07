"""Second autocorrelation inequality: optimize 50 nonnegative step heights."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def compute_value(h):
    """Compute the value for given heights h (50 nonnegative values)."""
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h)
    m = len(g)
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:m+1] = g
    # Spacing w: points are np.linspace(-0.5, 0.5, m+2), so w = 1.0/(m+1)
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # Vectorized computation
    yi = y[:m+1]
    yip1 = y[1:m+2]
    l2 = (w / 3.0) * np.sum(yi**2 + yi * yip1 + yip1**2)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _time_left():
    return TIME_LIMIT - SAFETY_MARGIN - time.time()

def _neg_value(params):
    """Negative value for minimization. params are 50 heights."""
    if time.time() > TIME_LIMIT - SAFETY_MARGIN:
        return np.inf
    h = np.maximum(params, 0.0)  # enforce nonnegativity
    if not np.all(np.isfinite(h)):
        return np.inf
    val = compute_value(h)
    return -val

def _initial_candidates(n_candidates=20, seed=42):
    """Generate diverse initial candidates."""
    rng = np.random.RandomState(seed)
    candidates = []
    
    # Uniform (baseline)
    candidates.append(np.ones(N))
    
    # Gaussian-like shapes with various widths and centers
    for sigma in [2, 4, 6, 8, 12, 16]:
        for center in [10, 25, 40]:
            x = np.arange(N)
            g = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            candidates.append(g)
    
    # Triangular / tent shapes
    for width in [10, 20, 30, 40]:
        for center in [10, 25, 40]:
            x = np.arange(N)
            t = np.maximum(0, 1 - np.abs(x - center) / width)
            candidates.append(t)
    
    # Cosine bumps
    for width in [10, 20, 30]:
        for center in [10, 25, 40]:
            x = np.arange(N)
            c = np.maximum(0, np.cos(np.pi * (x - center) / width))
            candidates.append(c)
    
    # Power-law decay from center
    for p in [1, 2, 3]:
        for center in [10, 25, 40]:
            x = np.arange(N)
            pl = np.maximum(0, 1 - (np.abs(x - center) / 25.0) ** p)
            candidates.append(pl)
    
    # Random smooth shapes
    for _ in range(10):
        x = np.arange(N)
        # Sum of a few random Gaussians
        shape = np.zeros(N)
        for _ in range(3):
            mu = rng.uniform(5, 45)
            sig = rng.uniform(3, 15)
            shape += np.exp(-0.5 * ((x - mu) / sig) ** 2)
        candidates.append(shape)
    
    # Bimodal
    for d in [8, 12, 16]:
        x = np.arange(N)
        bm = np.exp(-0.5 * ((x - 15) / 5) ** 2) + np.exp(-0.5 * ((x - 35) / 5) ** 2)
        candidates.append(bm)
    
    # Plateau with smooth edges
    for half_width in [5, 10, 15]:
        for center in [15, 25, 35]:
            x = np.arange(N)
            # Smooth step: 0 outside, 1 inside, smooth transition
            left = center - half_width
            right = center + half_width
            plateau = np.zeros(N)
            plateau[(x >= left + 2) & (x <= right - 2)] = 1.0
            # Smooth the edges
            for i in range(max(0, int(left) - 2), min(N, int(right) + 3)):
                if i >= int(left) - 2 and i < int(left) + 2:
                    t = (i - (left - 2)) / 4.0
                    plateau[i] = 0.5 * (1 - np.cos(np.pi * np.clip(t, 0, 1)))
                elif i > int(right) - 2 and i < int(right) + 2:
                    t = (i - (right - 2)) / 4.0
                    plateau[i] = 0.5 * (1 + np.cos(np.pi * np.clip(t, 0, 1)))
            candidates.append(plateau)
    
    return candidates

def _lbfgs_optimize(h0, maxiter=100):
    """Polish a candidate with L-BFGS-B."""
    result = minimize(_neg_value, h0, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
    if result.success or result.fun < np.inf:
        return np.maximum(result.x, 0.0)
    return h0

def construct():
    """Construct 50 nonnegative step heights maximizing the value."""
    start_time = time.time()
    
    # Generate initial candidates
    candidates = _initial_candidates(n_candidates=30, seed=42)
    
    # Evaluate all candidates
    best_h = None
    best_val = -np.inf
    
    for h in candidates:
        h = np.maximum(h, 0.0)
        val = compute_value(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
        
        # Check time
        if time.time() > TIME_LIMIT - SAFETY_MARGIN:
            break
    
    # Polish top candidates with L-BFGS-B
    # Sort candidates by value
    scored = []
    for h in candidates:
        h = np.maximum(h, 0.0)
        val = compute_value(h)
        scored.append((val, h))
    scored.sort(key=lambda x: -x[0])
    
    # Polish top 10
    for val, h in scored[:10]:
        if time.time() > TIME_LIMIT - SAFETY_MARGIN:
            break
        polished = _lbfgs_optimize(h, maxiter=80)
        pval = compute_value(polished)
        if pval > best_val:
            best_val = pval
            best_h = polished.copy()
    
    # Try differential evolution on a subset of the parameter space
    # Use a reduced set of parameters: model h as a sum of a few Gaussians
    # h_i = sum_k a_k * exp(-0.5 * ((i - mu_k)/sigma_k)^2)
    # This reduces dimensionality and may find better solutions
    
    if time.time() < TIME_LIMIT - 20:
        def _gauss_model(params):
            """Model h as sum of up to 4 Gaussians."""
            n_gauss = 4
            h = np.zeros(N)
            for k in range(n_gauss):
                a = params[3*k]
                mu = params[3*k+1]
                sigma = params[3*k+2]
                sigma = max(sigma, 0.5)
                x = np.arange(N)
                h += a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            return np.maximum(h, 0.0)
        
        def _neg_value_gauss(params):
            if time.time() > TIME_LIMIT - SAFETY_MARGIN:
                return np.inf
            h = _gauss_model(params)
            if not np.all(np.isfinite(h)):
                return np.inf
            val = compute_value(h)
            return -val
        
        # Bounds for 4 Gaussians: amplitude [0, 5], mu [0, 49], sigma [0.5, 30]
        bounds = []
        for _ in range(4):
            bounds.append((0, 5))
            bounds.append((0, 49))
            bounds.append((0.5, 30))
        
        try:
            de_result = differential_evolution(
                _neg_value_gauss, bounds,
                seed=42, maxiter=50, popsize=20,
                tol=1e-8, polish=False,
                updating='deferred', init='sobol'
            )
            if de_result.fun < np.inf:
                h_de = _gauss_model(de_result.x)
                val_de = compute_value(h_de)
                if val_de > best_val:
                    best_val = val_de
                    best_h = h_de.copy()
                
                # Polish with L-BFGS-B in full space
                if time.time() < TIME_LIMIT - SAFETY_MARGIN:
                    polished = _lbfgs_optimize(h_de, maxiter=60)
                    pval = compute_value(polished)
                    if pval > best_val:
                        best_val = pval
                        best_h = polished.copy()
        except Exception:
            pass
    
    # Final check: ensure we have a valid result
    if best_h is None:
        best_h = np.ones(N)
    
    # Final polish
    if time.time() < TIME_LIMIT - SAFETY_MARGIN:
        final = _lbfgs_optimize(best_h, maxiter=50)
        fval = compute_value(final)
        if fval > best_val:
            best_h = final
    
    return list(np.maximum(best_h, 0.0))
