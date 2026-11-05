"""Combined approach: smooth surrogate optimization with diverse initialization and local refinement."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from scipy.special import eval_chebyt
from scipy.optimize import brent

N = 400
DEADLINE = 148.0  # Safety margin before 150s kill

# Precompute Chebyshev basis
_x = np.linspace(-1, 1, N)
_basis_cache = {}

def _get_basis(degree):
    if degree in _basis_cache:
        return _basis_cache[degree]
    B = np.zeros((degree + 1, N))
    for k in range(degree + 1):
        B[k] = eval_chebyt(k, _x)
    _basis_cache[degree] = B
    return B

def _eval_value(h):
    """Compute the objective value (lower is better)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(~np.isfinite(h)):
        return np.inf
    sh = float(np.sum(h))
    if abs(sh) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (sh * sh))
    return val

def _smooth_objective(x, beta=80.0):
    """Smooth surrogate using log-sum-exp for the max."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    max_c = np.max(conv)
    shifted = conv - max_c
    # Numerically stable log-sum-exp
    log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
    val = abs(2.0 * N * log_sum_exp / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val

def _smooth_grad(x, beta=80.0):
    """Gradient of the smooth surrogate."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = np.convolve(h, h)
    max_c = np.max(conv)
    shifted = conv - max_c
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    # Vectorized gradient computation
    # grad_lse[i] = 2 * sum_{j=0}^{N-1} weights[i+j] * h[j]
    # Create matrix W where W[i, j] = weights[i+j]
    idx_i = np.arange(N)[:, None]
    idx_j = np.arange(N)[None, :]
    W = weights[idx_i + idx_j]
    grad_lse = 2.0 * (W @ h)
    
    lse = max_c + np.log(np.sum(weights)) / beta
    f = 2.0 * N * lse / (s * s)
    
    df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
    
    if f >= 0:
        grad = df_dh
    else:
        grad = -df_dh
    
    return grad

def _logistic_map_h(params, N=400):
    """Generate step heights from logistic map with tunable parameters.
    params: (r, x0, offset, scale) where r is bifurcation param, x0 is initial condition,
    offset shifts the binary threshold, scale scales the output.
    """
    r, x0, offset, scale = params
    x = np.zeros(N)
    x[0] = x0
    for i in range(1, N):
        x[i] = r * x[i-1] * (1.0 - x[i-1])
    # Map to [-1, 1] via thresholding with offset
    h = np.where(x > offset, 1.0, -1.0) * scale
    return h

def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # 2.0
    
    def elapsed():
        return time.time() - t0
    
    rng = np.random.RandomState(42)
    
    # --- Generate diverse initial points ---
    initial_points = []
    
    # Seed
    initial_points.append(np.ones(N).astype(float))
    
    # Random uniform
    for _ in range(15):
        initial_points.append(rng.uniform(-1, 1, N))
    
    # Bimodal: mostly zero with random spikes
    for _ in range(15):
        h = np.zeros(N)
        k = rng.randint(50, 350)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)
    
    # Alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    initial_points.append(h_alt)
    
    # Low-frequency cosines
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.sin(2 * np.pi * freq * t))
    
    # Parameterized bump profiles
    i_arr = np.arange(N)
    for center in [N//8, N//4, N//2, 3*N//4, 7*N//8]:
        for width in [20, 50, 100, 150]:
            h_bump = np.exp(-((i_arr - center)**2) / (2 * width**2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)
            # Double bump with opposite signs
            h_double = np.exp(-((i_arr - center)**2) / (2 * width**2)) - np.exp(-((i_arr - (N-1-center)**2) / (2 * width**2)))
            initial_points.append(h_double)
            # Bump with offset
            h_off = np.exp(-((i_arr - center)**2) / (2 * width**2)) + 0.5 * np.ones(N)
            initial_points.append(h_off)
    
    # Random Fourier synthesis
    for n_comp in [3, 5, 8]:
        for _ in range(5):
            freqs = rng.randint(1, 40, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)
    
    # Structured patterns: blocks of +1 and -1
    for block_size in [20, 50, 100]:
        n_blocks = N // block_size
        for _ in range(3):
            h_blocks = np.zeros(N)
            for b in range(n_blocks):
                sign = rng.choice([-1, 1])
                h_blocks[b*block_size:(b+1)*block_size] = sign
            initial_points.append(h_blocks)
    
    # Smooth random via Gaussian filter
    for _ in range(10):
        h_smooth = rng.uniform(-1, 1, N)
        h_smooth = gaussian_filter1d(h_smooth, sigma=rng.uniform(5, 20))
        initial_points.append(h_smooth)
        
    # Chebyshev-based initializations with random coefficients
    for degree in [5, 10, 15, 20]:
        B = _get_basis(degree)
        for _ in range(5):
            coeffs = rng.uniform(-1, 1, degree + 1)
            h_cheb = B.T @ coeffs
            initial_points.append(h_cheb)
    
    # --- Logistic map based initializations (chaotic binary sequences) ---
    # These exploit aperiodicity to suppress peak autocorrelation
    chaotic_rs = [3.5, 3.7, 3.8, 3.9, 3.95, 3.99]
    for r in chaotic_rs:
        for x0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for offset in [0.3, 0.5, 0.7]:
                for scale in [0.5, 1.0]:
                    if elapsed() > 2.0:
                        break
                    h_chaos = _logistic_map_h((r, x0, offset, scale), N)
                    s = np.sum(h_chaos)
                    if abs(s) > 1e-10:
                        initial_points.append(h_chaos)

    # --- Evaluate and sort initial points ---
    init_values = []
    for init_h in initial_points:
        val_init = _eval_value(init_h)
        init_values.append((val_init, init_h))
    init_values.sort(key=lambda x: x[0])
    
    # Take top 50 initial points for optimization
    top_inits = [h for _, h in init_values[:50]]
    
    # --- Optimize from each initial point using L-BFGS-B with smooth surrogate ---
    for init_h in top_inits:
        if elapsed() > DEADLINE - 15:
            break
        
        # Normalize to have non-zero sum
        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1
            s = np.sum(init_h)
        
        # Try multiple betas for the smooth surrogate
        for beta in [30.0, 60.0, 100.0, 200.0]:
            if elapsed() > DEADLINE - 5:
                break
            
            try:
                result = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    init_h,
                    jac=lambda x, b=beta: _smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 600, 'gtol': 1e-8}
                )
                
                if elapsed() > DEADLINE - 3:
                    break
                
                h_candidate = result.x
                v = _eval_value(h_candidate)
                if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                    best_val = v
                    best_h = h_candidate.copy()
            except Exception:
                pass
    
    # --- Final polish: high-precision smooth refinement from best solution ---
    if elapsed() < DEADLINE - 5:
        try:
            h0 = best_h.copy()
            
            # Use L-BFGS-B on the smooth objective with high beta for final polish
            # This approximates the exact objective more closely while remaining differentiable
            for beta_polish in [300.0, 600.0, 1000.0, 2000.0]:
                if elapsed() > DEADLINE - 2:
                    break
                res = minimize(
                    lambda x, b=beta_polish: _smooth_objective(x, b),
                    h0,
                    jac=lambda x, b=beta_polish: _smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxfun': 200, 'gtol': 1e-12}
                )
                h_candidate = res.x
                v = _eval_value(h_candidate)
                if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                    best_val = v
                    best_h = h_candidate.copy()
                    h0 = h_candidate.copy()
        except Exception:
            pass
    
    # --- Exact local refinement: try small perturbations around best solution ---
    if elapsed() < DEADLINE - 1:
        try:
            h_best = best_h.copy()
            s_best = np.sum(h_best)
            
            # Try coordinate-wise perturbations
            for _ in range(200):
                if elapsed() > DEADLINE - 0.5:
                    break
                h_try = h_best.copy()
                idx = rng.randint(0, N)
                delta = rng.uniform(-0.5, 0.5)
                h_try[idx] += delta
                s_try = np.sum(h_try)
                if abs(s_try) > 1e-10:
                    v_try = _eval_value(h_try)
                    if v_try < best_val:
                        best_val = v_try
                        h_best = h_try.copy()
        except Exception:
            pass
    
    # Final safety: ensure finite and non-zero sum
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
