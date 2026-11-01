import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 115.0  # Safety margin from 120s limit

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 = max_t |autoconv(t)| / (sum |h|)^2
    
    For a step function with heights in {-1, 1}, sum|h| = N.
    So C3 = max_t |autoconv(t)| / N^2.
    
    The autoconvolution is: (f*f)(t) = integral f(s) f(t-s) ds
    For step functions, this is a piecewise linear function.
    We compute it efficiently using FFT on a fine grid.
    """
    # signed_step is an array of N values in {-1, 1}
    # The function f(x) = signed_step[i] for x in [-1/4 + i/N, -1/4 + (i+1)/N)
    # Domain is [-1/4, 1/4], so length is 1/2.
    
    # Let h = 1/N be the interval width.
    # We need to compute the autoconvolution (f * f)(t) = integral f(s) f(t-s) ds
    # This is a convolution of f with itself.
    
    # For efficiency, we can use the fact that for step functions,
    # the autoconvolution at discrete points can be computed via FFT.
    
    # Actually, let's think more carefully.
    # f is a step function with N steps.
    # (f*f)(t) = integral_{-1/4}^{1/4} f(s) f(t-s) ds
    # The support of f is [-1/4, 1/4].
    # The support of f*f is [-1/2, 1/2].
    
    # Let's use a fine grid approach.
    # The step function f(x) = signed_step[i] for x in [i/N - 1/4, (i+1)/N - 1/4)
    # Wait, let's define it properly.
    # Let the domain be [0, 1/2] for simplicity, then shift.
    # Actually, the domain is [-1/4, 1/4].
    # Let's map to [0, N] for the step indices.
    # f(x) = signed_step[i] for x in [-1/4 + i/N, -1/4 + (i+1)/N)
    
    # The autoconvolution (f*f)(t) = integral f(s) f(t-s) ds
    # For t in [-1/2, 1/2].
    
    # Let's use FFT.
    # We need to represent f on a fine grid.
    # Let M be the number of fine grid points.
    # The domain length is 1/2.
    # Let's use M = 4*N or similar.
    
    M = 4 * N
    dx = 0.5 / M  # Grid spacing
    
    # Build f on the fine grid
    # f(x) for x in [-1/4, 1/4]
    # x_i = -1/4 + i * dx for i = 0, ..., M-1
    # We need to determine which step interval each x_i falls into.
    # Step i covers [-1/4 + i/N, -1/4 + (i+1)/N)
    # So for x_i, the step index is floor((x_i + 1/4) / (1/N)) = floor((x_i + 1/4) * N)
    
    x = -0.25 + np.arange(M) * dx
    # Compute step indices
    step_indices = np.floor((x + 0.25) * N).astype(int)
    # Clip to valid range [0, N-1]
    step_indices = np.clip(step_indices, 0, N-1)
    
    # Build f on fine grid
    f_fine = signed_step[step_indices]
    
    # Compute autoconvolution via FFT
    # (f*f)(t) = integral f(s) f(t-s) ds
    # This is the convolution of f with f.
    # In discrete terms: (f*f)[k] = sum_i f[i] f[k-i] * dx
    # Using FFT:
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # The autoconvolution is defined on [-0.5, 0.5] (or [0, 0.5] and symmetric)
    # We need to find the maximum absolute value.
    # The autoconv array has 2*M points, corresponding to t from -0.5 to 0.5.
    # Actually, the FFT-based convolution gives the circular convolution.
    # We need to be careful with the indexing.
    
    # Let's just take the max absolute value of the autoconv array.
    # But we need to make sure we're looking at the correct range.
    # The autoconv should be symmetric around t=0.
    
    # For now, let's just take the max abs value.
    c3 = np.max(np.abs(autoconv))
    
    # Normalize by (sum|h|)^2 = N^2
    c3 = c3 / (N * N)
    
    return c3

def _tanh_relaxation(x):
    """
    Map real-valued x to [-1, 1] using tanh.
    """
    return np.tanh(x)

def _objective_tanh(x):
    """
    Objective function for the tanh relaxation.
    x is a real-valued vector of length N.
    We map it to [-1, 1] using tanh, then compute C3.
    """
    f = _tanh_relaxation(x)
    # Ensure f is close to +/- 1 by scaling? No, tanh is already in [-1, 1].
    # But we want to encourage f to be close to +/- 1.
    # We can add a penalty term.
    # For now, just compute C3.
    return _compute_c3(f)

def _grad_tanh(x):
    """
    Gradient of the tanh relaxation objective.
    This is tricky because _compute_c3 is not differentiable in a simple way.
    We'll use numerical gradient or a different approach.
    
    Actually, let's use a different strategy.
    We'll use the discrete hill-climbing from the current program,
    but with better initialization from the tanh relaxation.
    """
    # For now, return zeros (we'll use L-BFGS-B with numerical gradient)
    return np.zeros(N)

def _local_search(h, max_iterations=10000, time_limit=5.0):
    """
    Fast local search using hill climbing.
    """
    start_time = time.time()
    current_score = _compute_c3(h)
    best_score = current_score
    best_h = h.copy()
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping a random element
        idx = np.random.randint(N)
        old_val = h[idx]
        h[idx] = -old_val
        
        new_score = _compute_c3(h)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            h[idx] = old_val
    
    return best_h, best_score

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Strategy: Multiple restarts of Continuous Relaxation + Discrete Polish
    
    # First, try the tanh relaxation approach
    # We'll use L-BFGS-B with numerical gradient
    
    np.random.seed(42)
    
    # Try several random initializations
    for restart in range(5):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random initialization in [-2, 2] (tanh will map to [-0.96, 0.96])
        x0 = np.random.uniform(-2, 2, N)
        
        # Run L-BFGS-B
        try:
            res = minimize(
                _objective_tanh, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            x_opt = res.x
        except Exception:
            x_opt = x0
        
        # Map to discrete
        f_rounded = np.sign(x_opt)
        f_rounded[f_rounded == 0] = 1.0
        f_rounded = f_rounded.astype(np.int8)
        
        # Discrete local search
        f_local, score_local = _local_search(f_rounded, max_iterations=5000, time_limit=5.0)
        
        if score_local < best_c3:
            best_c3 = score_local
            best_step = f_local.copy()
    
    # If we still have time, do more local search from the best found
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 10:
        f_local, score_local = _local_search(best_step, max_iterations=50000, time_limit=remaining_time - 2)
        if score_local < best_c3:
            best_c3 = score_local
            best_step = f_local.copy()
    
    return best_step
