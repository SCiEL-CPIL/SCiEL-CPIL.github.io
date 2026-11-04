import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0

# Fine grid for accurate autoconvolution evaluation
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Precompute bin centers for parametric model evaluation
BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * W

def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Normalization: max((f*f)(t)) / (integral f)^2
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    # Map step heights to fine grid
    f_fine = h[CELL]
    
    # FFT-based autoconvolution
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX  # scale by dx for continuous integral
    
    max_conv = np.max(conv)
    
    # Normalization: (integral f)^2 = (sum(h) * W)^2
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')
    
    return max_conv / integral_sq

def fourier_to_h(params):
    """
    Convert Fourier coefficients to nonnegative step function heights.
    params: [c0, c1, c2, ..., cK] where h(x) = c0 + sum_{k=1}^{K} c_k * cos(2*pi*k*x/0.5)
    We use |sum a_k exp(i*k*omega*x)|^2 form to guarantee nonnegativity.
    params: [a0, b1, a1, b2, a2, ..., bK, aK] where
    h(x) = (a0 + sum_{k=1}^{K} (a_k*cos(k*omega*x) + b_k*sin(k*omega*x)))^2
    omega = 2*pi/0.5 = 4*pi
    """
    K = (len(params) - 1) // 2
    omega = 4.0 * np.pi  # 2*pi / domain_length(0.5)
    
    # Build the signal: s(x) = a0 + sum (a_k cos + b_k sin)
    s = np.zeros(N)
    s += params[0]
    
    idx = 1
    for k in range(1, K + 1):
        a_k = params[idx]
        b_k = params[idx + 1]
        s += a_k * np.cos(k * omega * BIN_CENTERS) + b_k * np.sin(k * omega * BIN_CENTERS)
        idx += 2
    
    # h = s^2 ensures nonnegativity
    h = s * s
    return h

def objective_fourier(params):
    h = fourier_to_h(params)
    val = ratio_from_h(h)
    if not np.isfinite(val):
        return 1e10
    return val

def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)
    
    np.random.seed(42)
    
    # Fourier parameterization with varying K
    best_params = None
    
    # Try different numbers of Fourier modes
    for K in [3, 5, 7, 10]:
        if time.time() > deadline - 40.0:
            break
        
        n_params = 1 + 2 * K  # a0, then (a_k, b_k) pairs
        
        # Generate diverse initializations
        inits = []
        
        # 1. Near-uniform (small oscillations)
        p0 = np.zeros(n_params)
        p0[0] = 1.0
        for k in range(1, K + 1):
            p0[2*k - 1] = np.random.uniform(-0.1, 0.1)
            p0[2*k] = np.random.uniform(-0.1, 0.1)
        inits.append(p0)
        
        # 2. Dominant low-frequency modes
        p0 = np.zeros(n_params)
        p0[0] = np.random.uniform(0.5, 1.5)
        p0[1] = np.random.uniform(-0.5, 0.5)  # a1
        p0[2] = np.random.uniform(-0.5, 0.5)  # b1
        for k in range(2, K + 1):
            p0[2*k - 1] = np.random.uniform(-0.1, 0.1)
            p0[2*k] = np.random.uniform(-0.1, 0.1)
        inits.append(p0)
        
        # 3. Random smooth
        for _ in range(8):
            p0 = np.zeros(n_params)
            p0[0] = np.random.uniform(0.5, 2.0)
            for k in range(1, K + 1):
                amp = np.random.uniform(0.0, 0.5) / np.sqrt(k)
                p0[2*k - 1] = amp * np.random.choice([-1, 1])
                p0[2*k] = amp * np.random.choice([-1, 1])
            inits.append(p0)
        
        # 4. Cosine-dominant (even symmetry)
        for _ in range(5):
            p0 = np.zeros(n_params)
            p0[0] = np.random.uniform(0.5, 1.5)
            for k in range(1, K + 1):
                p0[2*k - 1] = np.random.uniform(-0.3, 0.3) / k
                p0[2*k] = 0.0  # No sine terms for even symmetry
            inits.append(p0)
        
        # Optimize from each initialization
        for p0 in inits:
            if time.time() > deadline - 35.0:
                break
                
            try:
                res = minimize(
                    objective_fourier,
                    p0,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                h_candidate = fourier_to_h(res.x)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Direct optimization with soft-max surrogate on h
    if time.time() < deadline - 15.0:
        beta = 200.0
        
        def obj_direct_smooth(h):
            h = np.clip(h, 0.0, None)
            s = h.sum()
            if s < 1e-15:
                return 1e10
            integral_sq = (s * W) ** 2
            
            f_fine = h[CELL]
            F = np.fft.rfft(f_fine, N_FFT)
            conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
            conv *= DX
            
            m = np.max(conv)
            if m < 1e-15:
                return 1e10
            soft_max = m + np.log(np.sum(np.exp(conv - m))) / beta
            return soft_max / integral_sq
            
        bounds = [(0.0, None)] * N
        try:
            res_direct = minimize(
                obj_direct_smooth,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            h_candidate = np.clip(res_direct.x, 0.0, None)
            val = ratio_from_h(h_candidate)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_h = h_candidate.copy()
        except Exception:
            pass
        
        # Second pass with higher beta for sharper max approximation
        if time.time() < deadline - 8.0:
            beta2 = 500.0
            
            def obj_direct_smooth2(h):
                h = np.clip(h, 0.0, None)
                s = h.sum()
                if s < 1e-15:
                    return 1e10
                integral_sq = (s * W) ** 2
                
                f_fine = h[CELL]
                F = np.fft.rfft(f_fine, N_FFT)
                conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
                conv *= DX
                
                m = np.max(conv)
                if m < 1e-15:
                    return 1e10
                soft_max = m + np.log(np.sum(np.exp(conv - m))) / beta2
                return soft_max / integral_sq
                
            try:
                res_direct2 = minimize(
                    obj_direct_smooth2,
                    best_h,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                h_candidate = np.clip(res_direct2.x, 0.0, None)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
            except Exception:
                pass
    
    # Final local search polish
    if time.time() < deadline - 3.0:
        # Adaptive perturbation size
        perturb_scale = 0.01
        for it in range(3000):
            if time.time() > deadline - 2.0:
                break
            
            n_perturb = np.random.randint(3, 15)
            idx = np.random.choice(N, n_perturb, replace=False)
            h_try = best_h.copy()
            h_try[idx] += np.random.randn(n_perturb) * perturb_scale
            h_try = np.clip(h_try, 0, None)
            
            val = ratio_from_h(h_try)
            if val < best_val:
                best_val = val
                best_h = h_try.copy()
                perturb_scale *= 0.95  # Reduce perturbation on improvement
            else:
                perturb_scale = min(perturb_scale * 1.05, 0.05)
    
    return list(best_h)
