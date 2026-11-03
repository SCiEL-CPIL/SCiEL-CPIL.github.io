import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0
DOMAIN_LEN = 0.5
X_STEP = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute for fast evaluation on a fine grid
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = DOMAIN_LEN / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Fourier basis parameters
# We will use K cosine and K sine terms.
# h(x) = a0 + sum_{k=1}^K (a_k cos(2*pi*k*x/L) + b_k sin(2*pi*k*x/L))
# L = 0.5
K_FOURIER = 20 
NUM_PARAMS = 1 + 2 * K_FOURIER

# Precompute Fourier basis on the step grid
X_STEP_CENTRED = X_STEP # already centered
# Angles: 2 * pi * k * x / L
# L = 0.5, so 2*pi*k*x/0.5 = 4*pi*k*x
angles_step = np.outer(X_STEP_CENTRED, 4 * np.pi * np.arange(1, K_FOURIER + 1))
cos_basis_step = np.cos(angles_step) # shape (N, K)
sin_basis_step = np.sin(angles_step) # shape (N, K)

# Precompute on fine grid for ratio calculation
angles_fine = np.outer(X_FINE, 4 * np.pi * np.arange(1, K_FOURIER + 1))
cos_basis_fine = np.cos(angles_fine) # shape (FINE_N, K)
sin_basis_fine = np.sin(angles_fine) # shape (FINE_N, K)

def params_to_h_fine(params):
    """
    Convert Fourier coefficients to the function values on the fine grid.
    params: [a0, a1, b1, a2, b2, ..., aK, bK]
    """
    a0 = params[0]
    a = params[1::2]  # a1, a2, ..., aK
    b = params[2::2]  # b1, b2, ..., bK
    
    # h(x) = a0 + a @ cos(x) + b @ sin(x)
    h_fine = a0 + a @ cos_basis_fine.T + b @ sin_basis_fine.T
    return h_fine

def params_to_h_step(params):
    """
    Convert Fourier coefficients to step function heights.
    """
    a0 = params[0]
    a = params[1::2]
    b = params[2::2]
    
    h_step = a0 + a @ cos_basis_step.T + b @ sin_basis_step.T
    return h_step

def ratio_from_h_fine(h_fine):
    """
    Compute the normalized maximum of the autoconvolution from fine grid values.
    """
    h_fine = np.clip(h_fine, 0.0, None)
    s = h_fine.sum()
    if s < 1e-15:
        return float('inf')
    
    # Compute autoconvolution via FFT
    F = np.fft.rfft(h_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX
    
    max_conv = np.max(conv)
    l2_sq = np.sum(h_fine ** 2) * DX
    if l2_sq < 1e-15:
        return float('inf')
    
    return max_conv / l2_sq

def objective(params):
    h_fine = params_to_h_fine(params)
    # The problem requires non-negative step function.
    # We evaluate the ratio on the clipped fine grid.
    # This is a smooth surrogate for the discrete step function constraint,
    # assuming the step function is a good approximation of the continuous one.
    # However, the step function is defined by averaging or sampling.
    # Let's use the step function heights for the final score, but optimize on fine grid for smoothness.
    # Actually, the task says "Choose a nonnegative step function...".
    # The score is based on the step function.
    # So we should compute the ratio based on the step function heights.
    # But optimizing on the fine grid gives a smoother gradient.
    # Let's compute the step heights and then expand to fine grid for the ratio calculation?
    # No, the ratio calculation in the seed used the fine grid expansion of the step heights.
    # Let's stick to that: compute step heights, expand to fine, compute ratio.
    
    h_step = params_to_h_step(params)
    h_step_clipped = np.clip(h_step, 0.0, None)
    
    # Expand to fine grid
    h_fine_eval = h_step_clipped[CELL]
    
    val = ratio_from_h_fine(h_fine_eval)
    
    if np.sum(h_step_clipped) < 1e-10:
        return 1e10
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_params = None
    best_val = float('inf')
    
    # Initial guess: small random coefficients
    # a0 should be positive to ensure non-negativity mostly
    # Let's try a few random starts
    
    np.random.seed(42)
    
    n_starts = 30
    
    for i in range(n_starts):
        if time.time() > deadline - 10.0:
            break
        
        # Random initialization
        params = np.zeros(NUM_PARAMS)
        params[0] = np.random.uniform(0.5, 2.0) # a0
        # Small coefficients for higher frequencies to start smooth
        for k in range(1, K_FOURIER + 1):
            scale = 1.0 / (k ** 0.5)
            params[1 + 2*(k-1)] = np.random.uniform(-0.5, 0.5) * scale # a_k
            params[2 + 2*(k-1)] = np.random.uniform(-0.5, 0.5) * scale # b_k
        
        try:
            res = minimize(
                objective, 
                params, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'maxfun': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception as e:
            pass
            
    # Refine the best solution
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            res = minimize(
                objective, 
                best_params, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    if best_params is not None:
        best_h = params_to_h_step(best_params)
    else:
        # Fallback: uniform
        best_h = np.ones(N)
        
    best_h = np.clip(best_h, 0, None)
    
    return list(best_h)
