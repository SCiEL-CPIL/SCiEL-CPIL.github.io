"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, Bounds
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def compute_value(h):
        h = np.asarray(h, dtype=float)
        g = np.convolve(h, h)
        m = len(g)
        y = np.concatenate(([0.0], g, [0.0]))
        w = 1.0 / (m + 1)
        # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
        l2 = (w / 3.0) * np.sum(y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2)
        norm_1 = np.sum(np.abs(g)) / (m + 1)
        norm_inf = np.max(np.abs(g))
        if norm_1 == 0 or norm_inf == 0:
            return 0.0
        return l2 / (norm_1 * norm_inf)
    
    def neg_value(h):
        return -compute_value(h)
    
    # Seed value
    seed_h = np.ones(N)
    seed_val = compute_value(seed_h)
    
    best_h = seed_h.copy()
    best_val = seed_val
    
    # Generate diverse starting points
    starts = [seed_h.copy()]
    
    # Add various structured starts
    # Ramp
    starts.append(np.linspace(0.1, 1.0, N))
    starts.append(np.linspace(1.0, 0.1, N))
    
    # Bump in center
    bump1 = np.zeros(N)
    bump1[N//4:3*N//4] = 1.0
    starts.append(bump1)
    
    # Wider bump
    bump2 = np.zeros(N)
    bump2[N//8:7*N//8] = 1.0
    starts.append(bump2)
    
    # Sine wave
    t = np.linspace(0, 2*np.pi, N)
    starts.append(0.5 + 0.5*np.sin(t))
    
    # Cosine wave
    starts.append(0.5 + 0.5*np.cos(t))
    
    # Two bumps
    bump3 = np.zeros(N)
    bump3[N//6:N//3] = 1.0
    bump3[2*N//3:5*N//6] = 1.0
    starts.append(bump3)
    
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    starts.append(np.exp(-x**2 * 2))
    starts.append(np.exp(-x**2 * 5))
    starts.append(np.exp(-x**2 * 0.5))
    
    # Random starts
    rng = np.random.RandomState(42)
    for _ in range(10):
        starts.append(rng.uniform(0, 1, N))
    
    # Also try scaling variations
    for base in starts[:5]:
        starts.append(base * 0.5)
        starts.append(base * 2.0)
    
    bounds = Bounds(np.zeros(N), np.inf)
    
    # Multi-start L-BFGS-B
    for i, h0 in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
        
        try:
            res = minimize(neg_value, h0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < 0:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_h = res.x.copy()
        except Exception:
            continue
    
    # Nelder-Mead polish on best result
    if time.time() < deadline - 10.0:
        try:
            res_nm = minimize(neg_value, best_h, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res_nm.fun < 0:
                val = -res_nm.fun
                if val > best_val:
                    best_val = val
                    best_h = res_nm.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 5.0:
        try:
            res_final = minimize(neg_value, best_h, method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10})
            if res_final.fun < 0:
                val = -res_final.fun
                if val > best_val:
                    best_val = val
                    best_h = res_final.x.copy()
        except Exception:
            pass
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0.0)
    
    # Return as list of floats
    return list(best_h)
# EVOLVE-BLOCK-END
