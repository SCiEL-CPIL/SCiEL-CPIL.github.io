"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s kill
    
    # Parametric model: h[i] = 0.5 + sum_k a_k * sin(2*pi*k*i/N + phi_k)
    # This gives 9 parameters (3 amplitudes + 3 phases + 3 frequencies)
    # Actually let's use fixed frequencies 1,2,3 and optimize amplitudes and phases
    # That's 6 parameters: a1,a2,a3,phi1,phi2,phi3
    
    def param_to_h(params):
        """Convert 6 parameters to 100 heights, projecting onto feasible set."""
        a1, a2, a3, p1, p2, p3 = params
        i = np.arange(N)
        h = 0.5 + a1*np.sin(2*np.pi*1*i/N + p1) + \
                     a2*np.sin(2*np.pi*2*i/N + p2) + \
                     a3*np.sin(2*np.pi*3*i/N + p3)
        # Project to [0,1] and adjust sum
        h = np.clip(h, 0.0, 1.0)
        # Adjust to satisfy sum constraint: sum(h) = N/2
        diff = np.sum(h) - N/2
        h = h - diff/N
        # Re-clip after adjustment
        h = np.clip(h, 0.0, 1.0)
        # Final correction for sum (small residual)
        diff2 = np.sum(h) - N/2
        h = h - diff2/N
        h = np.clip(h, 0.0, 1.0)
        return h
    
    def objective(params):
        if time.time() > deadline:
            return 1e6
        h = param_to_h(params)
        corr = np.correlate(h, 1-h, mode="full")
        return np.max(corr) / N * 2
    
    # Multiple random restarts
    best_h = np.full(N, 0.5)
    best_val = objective([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    rng = np.random.RandomState(42)
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initial parameters
        a1 = rng.uniform(-0.3, 0.3)
        a2 = rng.uniform(-0.3, 0.3)
        a3 = rng.uniform(-0.3, 0.3)
        p1 = rng.uniform(0, 2*np.pi)
        p2 = rng.uniform(0, 2*np.pi)
        p3 = rng.uniform(0, 2*np.pi)
        x0 = [a1, a2, a3, p1, p2, p3]
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if result.fun < best_val:
                best_val = result.fun
                best_h = param_to_h(result.x)
        except Exception:
            continue
    
    # Verify and return
    h_final = np.clip(best_h, 0.0, 1.0)
    sum_check = np.sum(h_final) - N/2
    if abs(sum_check) > 1e-9 * N:
        # Fallback to seed if constraint violated
        return np.full(N, 0.5)
    
    return h_final
# EVOLVE-BLOCK-END
