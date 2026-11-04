import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import BSpline
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

# Pre-compute basis functions for B-Splines
# We will use a small number of control points, say K=10 to 20
# B-Splines are non-negative if control points are non-negative? No, not necessarily.
# But we can constrain coefficients to be non-negative if we use a positive basis.
# B-Splines with non-negative coefficients are non-negative functions.
# Let's define the knot vector.
KNOTS = np.linspace(-0.25, 0.25, 20) # 20 knots, degree 3 -> 16 basis functions?
# Actually, let's just use a fixed number of basis functions, e.g., 15.
# We can define them on the fine grid or the bin centers.
# Let's define them on bin centers for simplicity in mapping to h.

NUM_SPLINES = 15
# Create a knot vector for B-Splines
# Degree 3, K control points.
# Knots: [0]*4 + ... + [1]*4
# Let's use a uniform knot vector over [0, 1] mapped to [-0.25, 0.25]
# Or just define Gaussians? The prompt says "Keep this design" (Gaussian/Block/Spline family).
# Inspiration 3 used splines. Let's stick to Splines as they are more flexible than Gaussians.

# Define B-Spline basis on the fine grid for accurate convolution
# We need the basis functions sampled on X_FINE.
# Let's use scipy.interpolate.BSpline
# Knots: uniform
knots = np.linspace(0, 1, NUM_SPLINES + 4)
# Extend knots for boundary conditions?
# Standard B-Splines on [0,1]
# Let's just use a set of well-spaced Gaussians as a robust baseline, 
# but allow "flat top" via block-like splines?
# Actually, the "probabilistic mixture" inspiration suggests Bernoulli variables.
# But Splines are smoother.

# Let's try a hybrid: 
# 1. Optimize a set of "peaks" (Gaussians) with L-BFGS-B using soft-max.
# 2. Then optimize the raw bins.
# Why? Because Gaussians are easy to initialize and optimize.
# The previous program did this but used raw max for search.
# We will use soft-max for the *entire* search.

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
    
    # Soft max
    # m = np.max(autoconv)
    # soft_max = m + np.log(np.sum(np.exp(autoconv - m))) / beta
    # To avoid overflow in exp, we subtract max
    m = np.max(autoconv)
    if m < 1e-15:
        return np.inf
    # If beta is very large, this approximates max
    # If beta is small, this is more "average"
    soft_max = m + np.log(np.sum(np.exp((autoconv - m) * beta)) / beta) / beta
    # Wait, the formula is: (1/beta) * log(sum(exp(beta * x_i)))
    # Let's implement it correctly:
    # log_sum_exp = m + (1/beta) * log(sum(exp(beta * (x_i - m))))
    # soft_max = log_sum_exp
    
    # Actually, simpler:
    # soft_max = np.logaddexp.reduce(autoconv) / beta  # This is for beta=1? No.
    # Standard soft max: 
    # S = sum(exp(x_i / T))
    # log(S) / T
    # Let T = 1/beta
    # log(sum(exp(beta * x_i))) / beta
    
    # Numerically stable:
    scaled = autoconv * beta
    m_scaled = np.max(scaled)
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
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    # Normalize initial best to integral 1? 
    # Score is scale invariant. Let's just keep it.
    best_val = compute_score(best_h, beta=1000.0)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()
            
    rng = np.random.default_rng(42)
    
    # === Phase 1: Gaussian superposition with L-BFGS-B using Soft-Max ===
    # We will anneal the beta parameter.
    # Start with low beta (smooth) to find basin, increase beta (sharp) to minimize peak.
    
    K_options = [3, 5, 7]
    
    for K in K_options:
        if time.time() > deadline - 30.0:
            break
        
        n_starts = 10
        for start in range(n_starts):
            if time.time() > deadline - 20.0:
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
            
            def objective_gauss(params, beta):
                h = params_to_h_gaussian(params)
                val = compute_score(h, beta=beta)
                if not np.isfinite(val):
                    return 1e10
                return val
            
            # Annealing loop for this start
            betas = [10.0, 50.0, 200.0, 1000.0]
            current_params = p0.copy()
            
            for beta in betas:
                if time.time() > deadline - 15.0:
                    break
                
                try:
                    res = minimize(
                        lambda p: objective_gauss(p, beta),
                        current_params,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
                    )
                    if res.success or res.fun < 1e9:
                        current_params = res.x
                        h_candidate = params_to_h_gaussian(current_params)
                        val = compute_score(h_candidate, beta=1000.0) # Always evaluate with high beta for comparison
                        update_best(h_candidate, val)
                except Exception:
                    pass
    
    # === Phase 2: Direct bin-level optimization with L-BFGS-B ===
    # This is the most powerful step. It removes the Gaussian constraint.
    # We start from the best Gaussian shape found.
    if time.time() < deadline - 10.0:
        # Normalize best_h to have integral 1 to help optimization?
        # The objective is scale invariant, but fixing scale can help convergence.
        # Let's fix the sum of h to be 1/DELTA (integral = 1).
        # Actually, L-BFGS-B with bounds [0, None] is fine.
        
        # We will use a soft-max objective with beta annealing.
        # Start with moderate beta.
        
        bounds = [(0.0, None)] * N
        
        def smooth_objective(h, beta):
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
        
        # Anneal beta
        betas = [10.0, 50.0, 200.0, 1000.0, 5000.0]
        current_h = best_h.copy()
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
            
            try:
                res_direct = minimize(
                    lambda h: smooth_objective(h, beta),
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
    if time.time() < deadline - 2.0:
        try:
            res_final = minimize(
                lambda h: smooth_objective(h, 10000.0),
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
