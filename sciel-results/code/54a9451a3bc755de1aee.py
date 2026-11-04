import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute max(autoconv) / ||h||_2^2 using FFT on fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def complex_exp_to_h(params):
    """
    Convert complex exponential parameters to step heights.
    params = [theta_1, r_1, phi_1, theta_2, r_2, phi_2, ..., r_base]
    h(x) = r_base^2 + |sum_k r_k * exp(i*(theta_k*x + phi_k))|^2
    
    This guarantees non-negativity by construction.
    """
    n_terms = (len(params) - 1) // 3
    r_base = params[-1]
    
    # Build the complex sum
    z = np.zeros(N, dtype=np.complex128)
    for k in range(n_terms):
        theta = params[3*k]
        r = params[3*k + 1]
        phi = params[3*k + 2]
        z += r * np.exp(1j * (theta * X + phi))
    
    # h(x) = r_base^2 + |z(x)|^2
    h = r_base**2 + np.real(z * np.conj(z))
    return h

def fourier_to_h(params):
    """
    Convert Fourier coefficients to step heights.
    params = [a0, a1, b1, a2, b2, ...]
    h(x) = a0 + sum_{k=1}^{K} [a_k cos(2*pi*k*x) + b_k sin(2*pi*k*x)]
    """
    K = (len(params) - 1) // 2
    h = np.full(N, params[0])
    for k in range(1, K + 1):
        ak = params[1 + 2*(k-1)]
        bk = params[2 + 2*(k-1)]
        arg = 2.0 * np.pi * k * X
        h += ak * np.cos(arg) + bk * np.sin(arg)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)

    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()

    rng = np.random.default_rng(42)

    # === Phase 1: Complex exponential superposition with L-BFGS-B ===
    # h(x) = r_base^2 + |sum_k r_k * exp(i*(theta_k*x + phi_k))|^2
    # This guarantees non-negativity by construction.
    
    for n_terms in [2, 3, 4, 5]:
        if time.time() > deadline - 15.0:
            break
        
        n_params = 3 * n_terms + 1  # +1 for r_base
        
        # Bounds
        bounds = []
        for _ in range(n_terms):
            bounds.append((-50.0, 50.0))   # theta (frequency)
            bounds.append((0.0, 5.0))       # r (amplitude)
            bounds.append((-np.pi, np.pi))  # phi (phase)
        bounds.append((0.0, 5.0))           # r_base
        
        # Try multiple restarts
        n_restarts = 25
        for i in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Random initial point
            x0 = np.array([rng.uniform(b[0], b[1]) for b in bounds])
            
            try:
                def objective_complex(params):
                    h = complex_exp_to_h(params)
                    # h is always non-negative by construction, but clip for safety
                    h = np.clip(h, 0.0, None)
                    r = ratio_from_h(h)
                    return r
                
                result = opt.minimize(
                    objective_complex,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                h_opt = complex_exp_to_h(result.x)
                h_clipped = np.clip(h_opt, 0, None)
                true_val = ratio_from_h(h_clipped)
                update_best(h_clipped, true_val)
                
            except Exception:
                pass
    
    # === Phase 2: Complex exponential with differential evolution ===
    # Use a fixed number of complex exponential terms (e.g., 4 terms = 13 params)
    # This provides a smooth, low-dimensional parameterization with guaranteed non-negativity
    
    if time.time() < deadline - 15.0:
        n_terms = 4
        n_params = 3 * n_terms + 1  # theta, r, phi for each term + r_base
        
        # Bounds
        bounds = []
        for _ in range(n_terms):
            bounds.append((-30.0, 30.0))   # theta
            bounds.append((0.0, 3.0))       # r
            bounds.append((-np.pi, np.pi))  # phi
        bounds.append((0.0, 3.0))           # r_base
        
        def complex_objective(params):
            h = complex_exp_to_h(params)
            h = np.clip(h, 0.0, None)
            return ratio_from_h(h)
        
        try:
            result = opt.differential_evolution(
                complex_objective,
                bounds,
                seed=42,
                maxiter=150,
                popsize=15,
                tol=1e-6,
                update_interval=1,
                polish=True
            )
            if result.fun < best_val:
                h_opt = complex_exp_to_h(result.x)
                h_clipped = np.clip(h_opt, 0, None)
                true_val = ratio_from_h(h_clipped)
                update_best(h_clipped, true_val)
        except Exception:
            pass

    # === Phase 3: Local refinement in spatial domain ===
    # From the best solution, do local spatial optimization
    # using coordinate descent on random subsets
    
    if time.time() < deadline - 5.0:
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break
            
            h_cur = best_h.copy()
            val_cur = best_val
            
            # Add small noise to escape local minima
            noise = rng.normal(0, 0.05, N)
            h_cur = np.clip(h_cur + noise, 0, None)
            val_cur = ratio_from_h(h_cur)
            
            # Coordinate descent: iterate over random coordinates
            for it in range(300):
                if time.time() > deadline - 2.0:
                    break
                
                # Pick a random coordinate
                idx = rng.integers(0, N)
                
                # Try to optimize this coordinate via a small 1D search
                # Sample a few values around the current value
                current_val = h_cur[idx]
                # Try values in a range around current
                candidates = [
                    current_val * 0.5,
                    current_val * 0.8,
                    current_val * 1.2,
                    current_val * 1.5,
                    max(0.0, current_val - 0.1),
                    max(0.0, current_val + 0.1),
                    0.0,  # Try zeroing out
                    1.0,  # Try setting to 1
                ]
                
                best_candidate = h_cur[idx]
                best_candidate_val = val_cur
                
                for cand in candidates:
                    h_try = h_cur.copy()
                    h_try[idx] = cand
                    v_try = ratio_from_h(h_try)
                    if v_try < best_candidate_val:
                        best_candidate_val = v_try
                        best_candidate = cand
                
                if best_candidate != h_cur[idx]:
                    h_cur[idx] = best_candidate
                    val_cur = best_candidate_val
            
            if val_cur < best_val:
                update_best(h_cur, val_cur)
    
    # === Phase 4: Try specific analytical shapes ===
    if time.time() < deadline - 2.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        
        # Gaussian with different widths
        for width in [3.0, 5.0, 7.0, 10.0]:
            h_g = np.exp(-width * x_norm**2)
            update_best(h_g, ratio_from_h(h_g))
        
        # Cosine
        h_c = 0.5 + 0.5 * np.cos(np.pi * x_norm)
        update_best(h_c, ratio_from_h(h_c))
        
        # Raised Cosine (Blackman-ish)
        h_b = 0.42 + 0.5 * np.cos(np.pi * x_norm) + 0.08 * np.cos(2 * np.pi * x_norm)
        h_b = np.clip(h_b, 0, None)
        update_best(h_b, ratio_from_h(h_b))
        
        # Triangular
        h_t = 1.0 - np.abs(x_norm)
        update_best(h_t, ratio_from_h(h_t))
        
        # Flat top (constant)
        h_f = np.ones(N)
        update_best(h_f, ratio_from_h(h_f))
        
        # Complex exponential examples: |1 + e^{i*theta*x}|^2 = 2 + 2*cos(theta*x)
        for theta in [5.0, 10.0, 15.0, 20.0]:
            params = np.array([theta, 1.0, 0.0, 1.0])  # one term + r_base
            h_ce = complex_exp_to_h(params)
            update_best(h_ce, ratio_from_h(h_ce))
        
        # Two-term complex exponential
        for theta1, theta2 in [(5.0, 10.0), (8.0, 16.0), (10.0, 20.0)]:
            params = np.array([theta1, 1.0, 0.0, theta2, 1.0, 0.0, 0.5])
            h_ce = complex_exp_to_h(params)
            update_best(h_ce, ratio_from_h(h_ce))
    
    return list(best_h)
