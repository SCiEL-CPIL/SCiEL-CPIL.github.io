import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 600
TIME_LIMIT = 140  # Safety margin

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fft_autocorr(f)
    return np.max(conv) / (s * s)

def fft_autocorr(f):
    """Compute autoconvolution using FFT for speed."""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)[:size]
    return conv

def symmetrize(f):
    """Enforce symmetry f[i] = f[N-1-i]."""
    half = f[:N//2]
    f_sym = np.empty(N)
    f_sym[:N//2] = half
    f_sym[N//2:] = half[::-1]
    return f_sym

def cosine_basis_sym(x, centers, widths, heights):
    """Generate a symmetric smooth nonnegative function from cosine bump components."""
    # x is the full grid
    result = np.zeros_like(x)
    # To ensure symmetry of the basis functions themselves is not required if we symmetrize the output,
    # but it's better to place bumps symmetrically or just symmetrize the result.
    for c, w, h in zip(centers, widths, heights):
        dist = np.abs(x - c)
        mask = dist <= w / 2.0
        if np.any(mask):
            result[mask] += h * np.cos(np.pi * (x[mask] - c) / w) ** 2
    return result

def parametric_to_step_sym(params, n_components):
    """Convert parameters to a symmetric step function of length N."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    widths = np.abs(widths) + 0.01
    widths = np.minimum(widths, 0.5)
    heights = np.abs(heights)
    
    f = cosine_basis_sym(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    f = symmetrize(f)
    return f

def objective_sym(params, n_components):
    f = parametric_to_step_sym(params, n_components)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fft_autocorr(f)
    return np.max(conv) / (s * s)

def compute_subgradient_sym(f, peak_indices):
    """
    Compute the subgradient of the max autoconvolution for the symmetric half of f.
    f is the full symmetric vector.
    peak_indices are indices in the full conv vector (length 2N-1).
    Returns gradient for the first N//2 elements.
    """
    N_full = len(f)
    half_len = N_full // 2
    g_full = np.zeros(N_full)
    
    # f is symmetric: f[i] = f[N-1-i]
    # C_k = sum_i f_i f_{i+k}
    # dC_k/df_j = f_{j+k} + f_{j-k}
    
    for k in peak_indices:
        # Calculate derivative for each j
        # f_{j+k} is valid if 0 <= j+k < N
        # f_{j-k} is valid if 0 <= j-k < N
        
        # Vectorized calculation for speed
        # Indices for f_{j+k}: j from 0 to N-1-k
        # Indices for f_{j-k}: j from k to N-1
        
        # dC_k/df_j = f[j+k] (if j+k < N) + f[j-k] (if j-k >= 0)
        
        # Let's compute the two parts
        # Part 1: f[j+k]
        # j ranges from 0 to N-1-k
        # Values are f[k], f[k+1], ..., f[N-1]
        # So g_full[0 : N-k] += f[k : N]
        
        # Part 2: f[j-k]
        # j ranges from k to N-1
        # Values are f[0], f[1], ..., f[N-1-k]
        # So g_full[k : N] += f[0 : N-k]
        
        g_full[0:N-k] += f[k:N]
        g_full[k:N] += f[0:N-k]
        
    # Map full gradient to symmetric half
    # g_sym[j] = g_full[j] + g_full[N-1-j] for j in 0..N/2-1
    g_sym = g_full[:half_len] + g_full[N-1-half_len:N][::-1]
    return g_sym

def local_refinement_sym(h, start_time, max_time):
    """
    Local refinement on the symmetric half vector h.
    Uses multiplicative weight updates based on subgradients.
    """
    best_h = h.copy()
    best_val = np.inf
    current_h = h.copy()
    
    # Normalize initial h to sum = N/2 (so full sum is N)
    s = np.sum(current_h)
    if s < 1e-12:
        return best_h, np.inf
    current_h = current_h * (N / (2 * s))
    
    # Adaptive step size
    eta = 0.05
    eta_min = 1e-6
    eta_max = 0.5
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < max_time:
        f = symmetrize(current_h)
        conv = fft_autocorr(f)
        max_c = np.max(conv)
        current_val = max_c / (N * N) # Since sum(f) = N
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                eta *= 0.9
                if eta < eta_min:
                    break
                no_improve_count = 0
            continue
        
        # Find peaks
        # Use a fraction of the max to include near-peaks for smoother gradient
        threshold = max_c * 0.95
        peak_indices = np.where(conv >= threshold)[0]
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])
        
        # Compute subgradient
        g_sym = compute_subgradient_sym(f, peak_indices)
        
        # Normalize gradient to prevent huge steps
        g_norm = np.linalg.norm(g_sym)
        if g_norm < 1e-12:
            break
        g_sym = g_sym / g_norm
        
        # Multiplicative update: h <- h * exp(-eta * g)
        # Clip exponent to avoid underflow/overflow
        update_factor = np.exp(-eta * g_sym)
        update_factor = np.clip(update_factor, 1e-6, 1e6)
        
        h_new = current_h * update_factor
        h_new = np.maximum(h_new, 1e-10)
        
        # Normalize
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / (2 * s_new))
        
        # Check if this step improves the score
        f_new = symmetrize(h_new)
        conv_new = fft_autocorr(f_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            # Increase eta slightly if we are making progress
            eta = min(eta * 1.01, eta_max)
        else:
            # Decrease eta
            eta *= 0.9
            if eta < eta_min:
                break

    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function (symmetric)
    h_const = np.ones(N//2)
    f_const = symmetrize(h_const)
    best_score = compute_score(f_const)
    best_h = h_const.copy()
    
    # Parametric optimization using Differential Evolution
    # This finds a good starting point for the local refinement
    n_components = 4
    bounds = [(-0.2, 0.2)] * n_components + \
             [(0.05, 0.4)] * n_components + \
             [(0.1, 3.0)] * n_components
    
    # DE is stochastic, set seed for reproducibility if needed, but default is fine
    # Limit time for DE
    de_time_limit = 30
    
    def de_objective(params):
        return objective_sym(params, n_components)
    
    try:
        if time.time() - start_time < TIME_LIMIT - de_time_limit - 20:
            result = differential_evolution(
                de_objective, bounds,
                maxiter=20,
                popsize=10,
                tol=1e-5,
                seed=42,
                polish=False # We will do our own polish
            )
            if result.fun < best_score:
                best_score = result.fun
                h_param = parametric_to_step_sym(result.x, n_components)[:N//2]
                best_h = h_param
    except Exception:
        pass
    
    # Local refinement on the symmetric half
    if time.time() - start_time < TIME_LIMIT - 5:
        refined_h, refined_score = local_refinement_sym(
            best_h, start_time, TIME_LIMIT - 2
        )
        if refined_score < best_score:
            best_h = refined_h
            best_score = refined_score
    
    # Final check and return
    f_final = symmetrize(best_h)
    f_final = np.maximum(f_final, 0.0)
    
    return [float(x) for x in f_final]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
