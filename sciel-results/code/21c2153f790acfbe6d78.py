"""
Third autocorrelation inequality: minimize 2*N*max(conv(h,h))/sum(h)^2 over h in R^N, N=400.

Strategy:
- Use a parameterized model: DC offset + Chirp + Gaussian Window.
- Global search with Differential Evolution.
- Local refinement with Nelder-Mead and L-BFGS-B.
- Final high-dimensional polish using L-BFGS-B with soft-max surrogate.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 400

def compute_value(h):
    """Compute the task value: 2*N*max(conv(h,h))/sum(h)^2"""
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s ** 2))

def soft_max_value(h, beta=50.0):
    """
    Smooth surrogate using soft-max instead of hard max.
    """
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    conv_scaled = conv / beta
    max_conv_scaled = np.max(conv_scaled)
    log_sum_exp = max_conv_scaled + np.log(np.sum(np.exp(conv_scaled - max_conv_scaled)))
    soft_max_conv = beta * log_sum_exp
    return abs(2.0 * N * soft_max_conv / (s ** 2))

def make_h_from_params(params):
    """
    Convert parameter vector to step heights h of length N.
    params: [dc, amp, chirp_rate, win_center, win_width]
    """
    dc = params[0]
    amp = params[1]
    chirp_rate = params[2]
    win_center = params[3]
    win_width = params[4]
    
    n = np.arange(N, dtype=float)
    # Chirp component
    chirp = np.cos(2.0 * np.pi * chirp_rate * (n / N) ** 2)
    # Gaussian window
    win = np.exp(-((n - win_center) ** 2) / (2.0 * win_width ** 2))
    
    h = dc + amp * chirp * win
    return h

def objective_from_params(params):
    h = make_h_from_params(params)
    return compute_value(h)

def soft_max_obj_from_params(params, beta=50.0):
    h = make_h_from_params(params)
    return soft_max_value(h, beta=beta)

def construct():
    t_start = time.time()
    DEADLINE = t_start + 140.0 # Safety margin

    def remaining():
        return DEADLINE - time.time()

    # Initial best: seed
    best_h = np.ones(N)
    best_val = compute_value(best_h) # 2.0

    # --- Phase 1: Global search with Differential Evolution ---
    # Bounds:
    # dc: [-10, 10]
    # amp: [-10, 10]
    # chirp_rate: [-5, 5]
    # win_center: [0, N]
    # win_width: [1, N]
    bounds = [
        (-10.0, 10.0),
        (-10.0, 10.0),
        (-5.0, 5.0),
        (0.0, float(N)),
        (1.0, float(N))
    ]

    if remaining() > 30.0:
        try:
            res_de = differential_evolution(
                objective_from_params,
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                polish=False,
                init='sobol',
                updating='immediate',
                workers=1,
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_h = make_h_from_params(res_de.x)
        except Exception:
            pass

    # --- Phase 2: Multi-restart Local Refinement ---
    rng = np.random.default_rng(123)
    n_restarts = 15
    for i in range(n_restarts):
        if remaining() < 15.0:
            break
        
        # Random start within bounds
        params_rand = np.array([
            rng.uniform(b[0], b[1]) for b in bounds
        ])
        
        # Try Nelder-Mead
        try:
            res_nm = minimize(
                objective_from_params,
                params_rand,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            )
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_h = make_h_from_params(res_nm.x)
        except Exception:
            pass
        
        # Try L-BFGS-B with soft-max surrogate
        if remaining() > 20.0:
            try:
                start_params = params_rand.copy()
                if hasattr(res_nm, 'x') and res_nm.x is not None:
                    start_params = res_nm.x
                    
                res_lbfgs = minimize(
                    lambda p: soft_max_obj_from_params(p, beta=20.0),
                    start_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0},
                )
                if res_lbfgs.fun < np.inf:
                    h_check = make_h_from_params(res_lbfgs.x)
                    v_check = compute_value(h_check)
                    if v_check < best_val:
                        best_val = v_check
                        best_h = h_check
            except Exception:
                pass

    # --- Phase 3: High-Dim Polish ---
    # Use L-BFGS-B directly on h with soft-max surrogate
    if remaining() > 10.0:
        try:
            def h_soft_max_obj(h_vec, beta=10.0):
                return soft_max_value(h_vec, beta=beta)
            
            # Start from best_h
            res_final = minimize(
                lambda hv: h_soft_max_obj(hv, beta=10.0),
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'iprint': 0},
            )
            if res_final.fun < np.inf:
                h_final = res_final.x
                v_final = compute_value(h_final)
                if v_final < best_val and np.all(np.isfinite(h_final)) and abs(np.sum(h_final)) > 1e-12:
                    best_val = v_final
                    best_h = h_final
        except Exception:
            pass

    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    # Final sanity check
    final_val = compute_value(best_h)
    if not np.isfinite(final_val) or final_val > 2.0:
        return np.ones(N)

    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Value: {compute_value(h)}")
    print(f"Score: {2.0 / compute_value(h)}")
