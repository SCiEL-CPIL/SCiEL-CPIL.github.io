import numpy as np
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

N = 600
DELTA = (1.0 / 2.0) / N  # interval width = 1/1200
BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * DELTA

# Fine grid for accurate autoconvolution evaluation
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / DELTA).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def compute_autoconv(h):
    """Compute autoconvolution of step function with heights h on N equal intervals using fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX
    return conv

def compute_score(h, beta=1000.0):
    """Compute the normalized max autoconvolution score using soft-max."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    if np.any(h < -1e-12):
        return np.inf
    h = np.maximum(h, 0.0)
    
    integral = DELTA * np.sum(h)
    if integral <= 1e-12:
        return np.inf
    
    autoconv = compute_autoconv(h)
    
    # Numerically stable soft-max: (1/beta) * log(sum(exp(beta * x_i)))
    scaled = autoconv * beta
    m_scaled = np.max(scaled)
    if m_scaled < 1e-15:
        return np.inf
    log_sum = m_scaled + np.log(np.sum(np.exp(scaled - m_scaled)))
    soft_max = log_sum / beta
    
    score = soft_max / (integral ** 2)
    return score

def params_to_h_gaussian(params):
    """Convert Gaussian superposition parameters to step heights."""
    K = len(params) // 3
    h = np.zeros(N)
    for i in range(K):
        a_raw = params[3 * i]
        x_c = params[3 * i + 1]
        s_raw = params[3 * i + 2]
        
        # Softplus for positive amplitude
        a = np.log1p(np.exp(np.clip(a_raw, -50, 50)))
        # Exp for positive width
        s = np.exp(np.clip(s_raw, -10, 10))
        s = max(s, 1e-4)
        # Clip center to domain
        x_c = np.clip(x_c, -0.25, 0.25)
        
        h += a * np.exp(-0.5 * ((BIN_CENTERS - x_c) / s) ** 2)
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = compute_score(best_h, beta=1000.0)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()
            
    rng = np.random.default_rng(42)
    
    # === Phase 1: Gaussian superposition with L-BFGS-B using Soft-Max ===
    K_options = [3, 5, 7]
    
    for K in K_options:
        if time.time() > deadline - 25.0:
            break
        
        n_starts = 8
        for start in range(n_starts):
            if time.time() > deadline - 15.0:
                break
            
            # Generate diverse initializations
            p0 = np.zeros(K * 3)
            if start == 0:
                for i in range(K):
                    p0[3 * i] = np.log(1.0 / K)
                    p0[3 * i + 1] = float(rng.uniform(-0.2, 0.2))
                    p0[3 * i + 2] = np.log(0.1 + float(rng.uniform(0, 0.2)))
            elif start == 1:
                # Single dominant bump
                p0[0] = np.log(1.0)
                p0[1] = 0.0
                p0[2] = np.log(0.15)
                for i in range(1, K):
                    p0[3 * i] = np.log(0.01)
                    p0[3 * i + 1] = float(rng.uniform(-0.2, 0.2))
                    p0[3 * i + 2] = np.log(0.05)
            else:
                # Random smooth shapes
                for i in range(K):
                    p0[3 * i] = float(rng.uniform(-2, 1))
                    p0[3 * i + 1] = float(rng.uniform(-0.25, 0.25))
                    p0[3 * i + 2] = float(rng.uniform(np.log(0.05), np.log(0.3)))
            
            # Annealing loop for this start
            betas = [10.0, 50.0, 200.0, 1000.0]
            current_params = p0.copy()
            
            for beta in betas:
                if time.time() > deadline - 10.0:
                    break
                
                def objective_gauss(params):
                    h = params_to_h_gaussian(params)
                    val = compute_score(h, beta=beta)
                    if not np.isfinite(val):
                        return 1e10
                    return val
                
                try:
                    res = minimize(
                        objective_gauss,
                        current_params,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
                    )
                    if res.success or res.fun < 1e9:
                        current_params = res.x
                        h_candidate = params_to_h_gaussian(current_params)
                        val = compute_score(h_candidate, beta=1000.0)
                        update_best(h_candidate, val)
                except Exception:
                    pass
    
    # === Phase 2: Direct bin-level optimization with L-BFGS-B ===
    if time.time() < deadline - 8.0:
        bounds = [(0.0, None)] * N
        
        def make_smooth_objective(beta):
            def smooth_objective(h):
                h = np.clip(h, 0.0, None)
                s = h.sum()
                if s < 1e-15:
                    return 1e10
                integral_sq = (s * DELTA) ** 2
                
                f_fine = h[CELL]
                F = np.fft.rfft(f_fine, N_FFT)
                conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
                conv *= DX
                
                # Soft max
                scaled = conv * beta
                m_scaled = np.max(scaled)
                log_sum = m_scaled + np.log(np.sum(np.exp(scaled - m_scaled)))
                soft_max = log_sum / beta
                
                if soft_max < 1e-15:
                    return 1e10
                return soft_max / integral_sq
            return smooth_objective
        
        # Anneal beta
        betas = [10.0, 50.0, 200.0, 1000.0, 5000.0]
        current_h = best_h.copy()
        
        for beta in betas:
            if time.time() > deadline - 3.0:
                break
            
            try:
                res_direct = minimize(
                    make_smooth_objective(beta),
                    current_h,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res_direct.success or res_direct.fun < 1e9:
                    current_h = np.clip(res_direct.x, 0.0, None)
                    val = compute_score(current_h, beta=10000.0)
                    update_best(current_h, val)
            except Exception:
                pass
    
    # === Phase 3: Final Polish with very high beta ===
    if time.time() < deadline - 1.0:
        try:
            res_final = minimize(
                make_smooth_objective(10000.0),
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_final.success:
                h_final = np.clip(res_final.x, 0.0, None)
                val = compute_score(h_final, beta=100000.0)
                update_best(h_final, val)
        except Exception:
            pass

    return list(best_h)
