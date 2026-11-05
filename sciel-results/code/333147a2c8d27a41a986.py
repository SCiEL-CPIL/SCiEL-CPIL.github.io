import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import scipy.fft as fft
import scipy.linalg as sla

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

def compute_value(h):
    """Exact evaluation of the objective function."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return float('inf')
    if not np.all(np.isfinite(h)):
        return float('inf')
    z = np.convolve(h, h)
    return abs(2 * N * np.max(z) / (s * s))

def smooth_objective(h):
    """
    Differentiable surrogate for the objective function.
    Uses log-sum-exp to approximate max(autoconv) smoothly.
    """
    if not np.all(np.isfinite(h)):
        return float('inf')
    
    s = np.sum(h)
    if abs(s) < 1e-9:
        return float('inf')
        
    # Compute autoconvolution via FFT with proper linear convolution length
    # z[k] = sum_j h[j] h[k-j]
    # Linear convolution length is 2N-1. We use FFT size >= 2N-1 to avoid aliasing.
    # Next power of 2 >= 799 is 1024.
    fft_len = 1024
    H_lin = fft.fft(h, n=fft_len)
    Z_lin = fft.ifft(H_lin * H_lin)
    z_lin = Z_lin.real[:2*N-1]
    
    # Smooth max: log-sum-exp(z / tau) * tau
    # To avoid overflow, subtract max
    z_max = np.max(z_lin)
    
    # Temperature tau. Smaller tau -> closer to max. Larger tau -> smoother.
    tau = 1.0 
    
    # If z_max is very large, exp(z/tau) might overflow.
    # Scale z by 1/tau first.
    z_scaled = z_lin / tau
    z_max_scaled = np.max(z_scaled)
    
    # log_sum_exp = z_max_scaled + log(sum(exp(z_scaled - z_max_scaled)))
    exp_terms = np.exp(z_scaled - z_max_scaled)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        log_sum_exp = z_max_scaled
    else:
        log_sum_exp = z_max_scaled + np.log(sum_exp)
        
    max_z_smooth = log_sum_exp * tau
    
    val = abs(2 * N * max_z_smooth / (s * s))
    return val

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - SAFETY_MARGIN)
    
    best_h = None
    best_val = float('inf')
    
    # 1. Initial Guesses
    # We want a function that minimizes the peak of its autocorrelation.
    # This is related to Barker codes / Golay sequences, but for continuous/real variables.
    # Random Gaussian is a decent baseline.
    # All ones is the seed (value 2.0).
    
    candidates = []
    
    # Seed
    candidates.append(np.ones(N))
    
    # Random Gaussian
    rng = np.random.default_rng(42)
    candidates.append(rng.normal(0, 1, N))
    
    # "Box" with negative sides
    h_box = np.zeros(N)
    h_box[N//4:3*N//4] = 1.0
    h_box[:N//4] = -0.5
    h_box[3*N//4:] = -0.5
    candidates.append(h_box)
    
    # Chebyshev-like oscillation
    x = np.linspace(-1, 1, N)
    h_cheb = np.cos(5 * np.arccos(x))
    candidates.append(h_cheb)
    
    # Evaluate initial candidates
    for h_cand in candidates:
        v = compute_value(h_cand)
        if v < best_val:
            best_val = v
            best_h = h_cand.copy()
    
    # 2. Optimization Phase 1: Local refinement of best start
    if best_h is not None:
        try:
            res = minimize(
                smooth_objective, 
                best_h, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                h_check = res.x
                if np.all(np.isfinite(h_check)) and abs(np.sum(h_check)) > 1e-9:
                    val_check = compute_value(h_check)
                    if val_check < best_val:
                        best_val = val_check
                        best_h = h_check.copy()
        except Exception:
            pass
    
    # 3. Optimization Phase 2: Global search using Differential Evolution
    # DE is good for finding a better global basin if the local optima is poor.
    # However, N=400 is high dimension for DE. 
    # The inspiration mentions "convex hull of optima" or "Chebyshev polynomials".
    # Let's try a lower-dimensional parameterization:
    # h[n] = sum_{k=0}^{K} c_k * T_k(n/N - 0.5)  (Chebyshev basis)
    # This enforces smoothness and might help escape local minima of the raw vector search.
    
    if time.time() < deadline - 20:
        # Try optimization in Chebyshev coefficient space
        K = 10  # Number of Chebyshev coefficients
        # Chebyshev polynomials of first kind: T_k(x)
        x = np.linspace(-1, 1, N)
        T = np.zeros((K+1, N))
        T[0] = 1
        if K >= 1: T[1] = x
        for k in range(2, K+1):
            T[k] = 2*x*T[k-1] - T[k-2]
        
        def cheb_obj(coeffs):
            h = coeffs @ T
            return smooth_objective(h)
        
        # Initial coeffs
        # We want h to have some structure.
        # Random coeffs
        init_coeffs = rng.normal(0, 1, K+1)
        # Ensure sum(h) != 0
        # If sum is 0, adjust c0
        h_init = init_coeffs @ T
        if abs(np.sum(h_init)) < 1e-6:
            init_coeffs[0] += 1.0
            
        try:
            # Use L-BFGS-B on coefficients first
            res_c = minimize(
                cheb_obj, 
                init_coeffs, 
                method='L-BFGS-B',
                options={'maxiter': 200}
            )
            h_c = res_c.x @ T
            val_c = compute_value(h_c)
            if val_c < best_val:
                best_val = val_c
                best_h = h_c.copy()
                
        except Exception:
            pass
            
        # If time permits, try a few random restarts in Cheb space
        if time.time() < deadline - 10:
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                rand_coeffs = rng.normal(0, 1, K+1)
                h_rand_c = rand_coeffs @ T
                if abs(np.sum(h_rand_c)) < 1e-6:
                    rand_coeffs[0] += 1.0
                    h_rand_c = rand_coeffs @ T
                try:
                    res_c2 = minimize(
                        cheb_obj, 
                        rand_coeffs, 
                        method='L-BFGS-B',
                        options={'maxiter': 100}
                    )
                    h_c2 = res_c2.x @ T
                    val_c2 = compute_value(h_c2)
                    if val_c2 < best_val:
                        best_val = val_c2
                        best_h = h_c2.copy()
                except Exception:
                    pass

    # 4. Final Polish
    if best_h is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                smooth_objective, 
                best_h, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_final.fun < best_val:
                h_check = res_final.x
                if np.all(np.isfinite(h_check)) and abs(np.sum(h_check)) > 1e-9:
                    val_check = compute_value(h_check)
                    if val_check < best_val:
                        best_val = val_check
                        best_h = h_check.copy()
        except Exception:
            pass

    # Ensure validity
    if best_h is None or not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print(len(h))
    print(np.sum(h))
    z = np.convolve(h, h)
    print(abs(2 * N * np.max(z) / np.sum(h)**2))
