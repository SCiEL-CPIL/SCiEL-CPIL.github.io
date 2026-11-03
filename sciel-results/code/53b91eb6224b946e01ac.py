import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    # Continuous autoconvolution of step function f(t) = sum h_i 1_{[x_i, x_{i+1})}(t)
    # The peak is proportional to the max of the discrete convolution of h with itself.
    # Specifically, max(f*f) = DELTA * max(conv(h,h))
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Pre-compute indices for convolution
    # We will use FFT for speed in the optimization loop
    # np.convolve is O(N^2), FFT is O(N log N)
    # But N=50 is small. O(N^2) = 2500. FFT overhead might be higher.
    # However, we are doing many iterations. Let's stick to np.convolve for simplicity and correctness first,
    # or use scipy.signal.fftconvolve if available and faster.
    # Actually, for N=50, np.convolve is extremely fast.
    
    def objective_and_grad(z):
        # z is unconstrained vector of length N
        # h = softmax(z)
        # To avoid overflow, subtract max
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        h = exp_z / np.sum(exp_z)
        
        l1 = 1.0
        l2_sq = np.sum(h**2)
        
        # Discrete convolution
        h_conv = np.convolve(h, h, mode='full')
        autoconv_max = DELTA * np.max(h_conv)
        
        if autoconv_max == 0 or l2_sq == 0:
            return 1e10, np.zeros(N)
        
        score = l2_sq / autoconv_max
        
        # We want to maximize score.
        # Minimize -log(score) = log(autoconv_max) - log(l2_sq)
        # Let L = log(B) - log(A) where B = autoconv_max, A = l2_sq
        # Grad L wrt z
        
        # A = sum h^2
        # dA/dz_j = 2 h_j (h_j - A)
        # d(log A)/dz_j = (2 h_j (h_j - A)) / A
        
        # B = max_k C_k. Let k* be argmax.
        # C_k = sum_l h_l h_{k-l}
        # dC_k/dz_j = 2 h_j (h_{k-j} - C_k)  (with proper boundary handling for h_{k-j})
        # d(log B)/dz_j = (2 h_j (h_{k*-j} - B)) / B
        
        k_star = np.argmax(h_conv)
        B = DELTA * h_conv[k_star]
        A = l2_sq
        
        # Compute gradient of L = log(B) - log(A)
        # grad_L = grad_log_B - grad_log_A
        
        # grad_log_A_j = 2 * h_j * (h_j - A) / A
        grad_log_A = 2.0 * h * (h - A) / A
        
        # grad_log_B_j = 2 * h_j * (h_{k*-j} - B) / B
        # Note: h_{k*-j} corresponds to the term in the convolution sum.
        # C_{k*} = sum_l h_l h_{k*-l}
        # The term involving h_j appears when l=j (h_j h_{k*-j}) and when k*-l=j => l=k*-j (h_{k*-j} h_j).
        # So derivative is h_j * (dh_{k*-j}/dz_j + h_{k*-j} * delta_{j, k*-j} ... wait)
        # Let's re-verify: C_k = sum_l h_l h_{k-l}.
        # dC_k/dz_j = sum_l (dh_l/dz_j h_{k-l} + h_l dh_{k-l}/dz_j)
        # dh_m/dz_j = h_m (delta_{mj} - h_j)
        # Term 1: sum_l h_l (delta_{lj} - h_j) h_{k-l} = h_j h_{k-j} - h_j C_k
        # Term 2: sum_l h_l h_{k-l} (delta_{k-l, j} - h_j) = h_j h_{k-j} - h_j C_k
        # Total: 2 h_j (h_{k-j} - C_k)
        
        # We need h_{k*-j}.
        # h is indexed 0..N-1.
        # k* is in 0..2N-2.
        # j is in 0..N-1.
        # h_{k*-j} is valid if 0 <= k*-j < N.
        # If not, it is 0.
        
        h_shifted = np.zeros(N)
        for j in range(N):
            idx = k_star - j
            if 0 <= idx < N:
                h_shifted[j] = h[idx]
            else:
                h_shifted[j] = 0.0
                
        grad_log_B = 2.0 * h * (h_shifted - B) / B
        
        grad_L = grad_log_B - grad_log_A
        
        # We minimize L, so return -score (for value) and grad_L (for gradient)
        # Wait, scipy minimize expects gradient of the function to be minimized.
        # Function to minimize: f(z) = -log(score) = L
        # So gradient is grad_L.
        
        return -np.log(score), grad_L

    # Inits
    inits_z = []
    
    # Uniform
    inits_z.append(np.zeros(N))
    
    # Random
    for _ in range(20):
        inits_z.append(np.random.randn(N) * 0.5)
        
    # Peaked
    for c in np.linspace(5, N-5, 10):
        z = -np.abs(np.arange(N) - c) * 0.5
        inits_z.append(z)
        
    for z_init in inits_z:
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(objective_and_grad, z_init, method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-8})
            
            z_opt = res.x
            z_max = np.max(z_opt)
            exp_z = np.exp(z_opt - z_max)
            h_opt = exp_z / np.sum(exp_z)
            
            # Clip small negatives if any (should be positive due to softmax)
            h_opt = np.maximum(h_opt, 0)
            h_opt /= np.sum(h_opt)
            
            s_opt = _score_raw(h_opt)
            
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
                
        except Exception:
            pass

    # Final refinement with COBYLA if time permits? 
    # L-BFGS-B should be sufficient.
    
    return list(best_h)
