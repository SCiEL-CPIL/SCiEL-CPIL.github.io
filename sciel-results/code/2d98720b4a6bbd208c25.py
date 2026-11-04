import numpy as np
import time
import scipy.optimize as opt

N = 600
# Interval width for each step
W = 1.0 / 1200.0
# Center points of the intervals
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for accurate convolution evaluation
# We need to evaluate the ratio accurately. 
# The evaluator likely uses a specific method. 
# The current program uses a fine grid FFT approach. 
# The inspiration uses np.convolve on the coarse grid.
# Let's stick to the fine grid FFT from the current program for consistency and accuracy, 
# but optimize the inner loop.

FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
# Map fine grid points to coarse bin indices
# x in [-0.25, 0.25)
# bin index = floor((x + 0.25) / W)
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def compute_ratio(h):
    """
    Computes the score: max(autoconv) / ||h||_2^2
    using the fine grid FFT method for accuracy.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    l2_coarse_sq = np.sum(h * h)
    if l2_coarse_sq < 1e-15:
        return float('inf')
        
    # Upsample to fine grid
    f = h[CELL]
    l2_fine = np.sum(f * f) * DX
    if l2_fine < 1e-15:
        return float('inf')
        
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    max_conv = np.max(conv)
    
    # The ratio is scale invariant. 
    # max(conv) scales as h^2. ||h||_2^2 scales as h^2.
    # So we can just use the fine grid values.
    return float(max_conv / l2_fine)

def smooth_objective(params, tau=0.01):
    """
    Smooth objective for L-BFGS-B.
    Normalizes h to unit L2 norm, then minimizes soft-max of autoconvolution.
    """
    # Unpack Fourier params
    K = len(params) // 2 - 1 # Assuming params are [a0, a1, b1, a2, b2, ...]
    # Actually, let's define the structure clearly in construct()
    
    # For now, this function assumes `params` is the full vector for the Fourier series
    # Structure: [a0, a1, b1, a2, b2, ..., aK, bK]
    n_params = len(params)
    K = (n_params - 1) // 2
    
    a0 = params[0]
    a = params[1::2]
    b = params[2::2]
    
    # Construct h
    h = np.full(N, a0)
    for k in range(1, K + 1):
        h += a[k-1] * np.cos(2.0 * np.pi * k * X / 0.5) # Period is 0.5
        h += b[k-1] * np.sin(2.0 * np.pi * k * X / 0.5)
    
    h = np.clip(h, 0.0, None)
    
    # Normalize to unit L2 norm (on the coarse grid, for stability)
    norm = np.sqrt(np.sum(h * h))
    if norm < 1e-15:
        return 1e10
    h_norm = h / norm
    
    # Upsample and convolve
    f = h_norm[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    
    # Soft max
    m = np.max(conv)
    if m < 0:
        # If max is negative (unlikely for autoconv of non-negative), just return m
        return float(m)
    
    # tau controls smoothness. Smaller tau -> closer to exact max.
    # We need to handle numerical stability.
    exp_vals = np.exp((conv - m) / tau)
    soft_max = m + tau * np.log(np.sum(exp_vals))
    
    return float(soft_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Define Fourier parameterization
    # h(x) = a0 + sum_{k=1}^K (a_k cos(2*pi*k*x/0.5) + b_k sin(2*pi*k*x/0.5))
    K = 20  # Number of modes. Total params: 1 + 2*K = 41
    
    # Initial guess: constant function
    # For constant h, a0 = 1, others 0.
    # Score for constant is 1.0.
    init_params = np.zeros(1 + 2 * K)
    init_params[0] = 1.0
    
    best_params = init_params.copy()
    best_score = compute_ratio(np.ones(N)) # Should be 1.0
    
    def update_best(params, score):
        nonlocal best_params, best_score
        if score < best_score:
            best_score = score
            best_params = params.copy()
    
    # Helper to convert params to h for final evaluation
    def params_to_h(params):
        a0 = params[0]
        a = params[1::2]
        b = params[2::2]
        h = np.full(N, a0)
        for k in range(1, K + 1):
            h += a[k-1] * np.cos(2.0 * np.pi * k * X / 0.5)
            h += b[k-1] * np.sin(2.0 * np.pi * k * X / 0.5)
        return np.clip(h, 0.0, None)
    
    rng = np.random.default_rng(42)
    
    # === Phase 1: Differential Evolution for Global Search ===
    # Use a moderate tau for DE to make it smooth enough for DE's population updates
    tau_de = 0.05
    
    def de_objective(params):
        return smooth_objective(params, tau=tau_de)
    
    bounds = [(-10.0, 10.0)] * (1 + 2 * K)
    # Ensure a0 is likely positive to start, but DE will explore.
    
    if time.time() < deadline - 60.0:
        try:
            result_de = opt.differential_evolution(
                de_objective,
                bounds=bounds,
                seed=42,
                maxiter=100,
                tol=1e-5,
                popsize=20,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False
            )
            
            score_de = compute_ratio(params_to_h(result_de.x))
            update_best(result_de.x, score_de)
            
        except Exception as e:
            pass
    
    # === Phase 2: L-BFGS-B Refinement ===
    # Use a smaller tau for higher precision
    tau_lbfgs = 0.01
    
    if time.time() < deadline - 30.0:
        cur_params = best_params.copy()
        
        try:
            result_lbfgs = opt.minimize(
                lambda p: smooth_objective(p, tau=tau_lbfgs),
                cur_params,
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            
            score_lbfgs = compute_ratio(params_to_h(result_lbfgs.x))
            update_best(result_lbfgs.x, score_lbfgs)
            
        except Exception:
            pass
    
    # === Phase 3: High-Precision Polish with Exact Max ===
    # Switch to exact max for the objective, but use a very small perturbation for gradients?
    # Or just do local search on the exact objective.
    
    if time.time() < deadline - 10.0:
        cur_params = best_params.copy()
        cur_score = best_score
        
        # Local search with small steps in parameter space
        step_size = 0.01
        min_step = 1e-5
        stagnation = 0
        
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
            
            # Random small perturbation
            noise = rng.normal(0, step_size, size=len(cur_params))
            new_params = cur_params + noise
            
            new_h = params_to_h(new_params)
            new_score = compute_ratio(new_h)
            
            if new_score < cur_score:
                cur_score = new_score
                cur_params = new_params
                update_best(cur_params, cur_score)
                stagnation = 0
            else:
                stagnation += 1
                if stagnation > 20:
                    step_size *= 0.8
                    if step_size < min_step:
                        break
                    stagnation = 0
    
    # === Phase 4: Final Verification ===
    final_h = params_to_h(best_params)
    final_score = compute_ratio(final_h)
    
    return list(final_h)
