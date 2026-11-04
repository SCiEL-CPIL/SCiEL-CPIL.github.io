"""
Strategy: Spectral Parametric Optimization via Neural Network/Fourier Representation
Instead of optimizing time-domain heights directly, we optimize a small set of 
spectral parameters (amplitudes and phases) that define a smooth signal.
The signal is generated as a sum of sines/cosines (or via a small MLP if needed, 
but Fourier is more direct for autocorrelation).
We minimize the soft-max of the absolute autoconvolution side-lobes.
Finally, we project to signs and refine.
"""
import numpy as np
import time
import scipy.optimize

N = 400

# EVOLVE-BLOCK-START

def _autoconv_full(x):
    """Full linear autoconvolution of x (length 2N-1)."""
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, n=size)
    conv = np.fft.irfft(F * F, n=size)
    return conv[:2 * n - 1]

def _score_offdiag(x):
    """
    Compute the metric: max(|autoconv|) for k != 0 / (sum(x))^2
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    if len(conv) <= 1:
        return 0.0
    max_abs_offdiag = np.max(np.abs(conv[1:]))
    return max_abs_offdiag / (integral * integral)

def _generate_signal_from_spectral(params, M):
    """
    Generate a time-domain signal from spectral parameters.
    params: [a_1, p_1, a_2, p_2, ..., a_M, p_M, bias]
    x[n] = bias + sum_{k=1}^M a_k * cos(2*pi*k*n/N + p_k)
    """
    n = np.arange(N)
    x = np.full(N, params[-1])  # bias
    
    # M frequencies
    for k in range(1, M + 1):
        a = params[2*(k-1)]
        p = params[2*(k-1)+1]
        x += a * np.cos(2 * np.pi * k * n / N + p)
        
    return x

def _spectral_objective(params, M):
    """
    Objective function for spectral parameter optimization.
    Minimizes the soft-max of the absolute autoconvolution side-lobes,
    normalized by the square of the integral.
    """
    x = _generate_signal_from_spectral(params, M)
    
    # Ensure the signal has a non-zero mean to avoid division by zero issues
    # and to align with the "integral" normalization.
    integral = np.sum(x)
    if abs(integral) < 1e-6:
        # Penalty for zero integral
        return 1e12
        
    # Normalize x so that sum(x) = 1? 
    # The score is scale-invariant: score(c*x) = score(x).
    # However, the spectral representation is linear.
    # If we fix the scale, we remove the flat valley in the scaling direction.
    # Let's normalize x to have unit sum during the objective calculation.
    x_norm = x / integral
    
    conv = _autoconv_full(x_norm)
    if len(conv) <= 1:
        return 0.0
    
    abs_c = np.abs(conv[1:])
    
    # Soft max approximation
    beta = 100.0
    m = np.max(abs_c)
    if m < 1e-15:
        soft_max = 0.0
    else:
        # Numerically stable log-sum-exp
        soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
        
    return soft_max

def _optimize_spectral(M, deadline, rng):
    """
    Optimize spectral parameters using L-BFGS-B.
    """
    # Parameters: 2*M (amplitudes and phases) + 1 (bias)
    n_params = 2 * M + 1
    
    # Initialize with small random amplitudes and random phases
    params0 = np.zeros(n_params)
    for k in range(M):
        params0[2*k] = rng.uniform(0.1, 1.0)  # amplitude
        params0[2*k+1] = rng.uniform(0, 2*np.pi) # phase
    params0[-1] = rng.uniform(-0.5, 0.5) # bias
    
    t_start = time.time()
    
    # Use L-BFGS-B
    try:
        result = scipy.optimize.minimize(
            _spectral_objective,
            x0=params0,
            args=(M,),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
        )
        
        if result.success or result.fun < 1e10:
            x_opt = _generate_signal_from_spectral(result.x, M)
            # Normalize to sum=1
            s = np.sum(x_opt)
            if abs(s) > 1e-12:
                x_opt = x_opt / s
            return x_opt, result.fun
    except Exception:
        pass
        
    # Fallback: return the initial guess
    x_init = _generate_signal_from_spectral(params0, M)
    s = np.sum(x_init)
    if abs(s) > 1e-12:
        x_init = x_init / s
    return x_init, _score_offdiag(x_init)

def _refine_signed(h, deadline, rng):
    """
    Refine the signed sequence using local search (bit flips).
    """
    n = len(h)
    current = h.copy()
    current_score = _score_offdiag(current)
    best_score = current_score
    best_h = current.copy()
    
    size = 1 << (2 * n - 1).bit_length()
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Try flipping a random bit
        i = int(rng.integers(0, n))
        old_val = current[i]
        current[i] = -old_val
        
        s = np.sum(current)
        if abs(s) < 1e-12:
            new_score = np.inf
        else:
            H = np.fft.rfft(current, size)
            ac = np.fft.irfft(H * H, size)[:2 * n - 1]
            max_abs = np.max(np.abs(ac[1:]))
            new_score = max_abs / (s * s)
            
        if new_score < current_score:
            current_score = new_score
            no_improve = 0
            if new_score < best_score:
                best_score = new_score
                best_h = current.copy()
        else:
            current[i] = old_val
            no_improve += 1
            
    return best_h, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    rng = np.random.default_rng(42)
    
    best_score = np.inf
    best_h = np.ones(N)
    
    # Strategy: Try different spectral complexities (M)
    # Low M captures broad structure, high M captures details.
    # We will run spectral optimization for a few M values.
    
    M_values = [5, 10, 20, 40]
    
    for M in M_values:
        if time.time() > deadline - 20:
            break
            
        # Allocate time for optimization and refinement
        # Optimization: ~10s
        # Refinement: ~10s
        opt_deadline = time.time() + 15
        ref_deadline = opt_deadline + 10
        
        x_cont, score_cont = _optimize_spectral(M, opt_deadline, rng)
        
        # Project to signs
        # Thresholding at 0 is standard, but we can also try thresholding at mean
        # Since we normalized sum(x)=1, mean is 1/N.
        # Signs of x_cont
        h_proj = np.sign(x_cont)
        
        # If all signs are the same, it's a bad projection
        if np.all(h_proj == 1) or np.all(h_proj == -1):
             # Try random signs? Or just skip?
             # Let's try to force some variation by adding small noise before sign
             h_proj = np.sign(x_cont + rng.normal(0, 0.1, N))
        
        # Refine
        h_ref, score_ref = _refine_signed(h_proj, ref_deadline, rng)
        
        if score_ref < best_score:
            best_score = score_ref
            best_h = h_ref
            
    # Final polish with remaining time
    if time.time() < deadline - 5:
        best_h, best_score = _refine_signed(best_h, deadline - 5, rng)
        
    return best_h

# EVOLVE-BLOCK-END
