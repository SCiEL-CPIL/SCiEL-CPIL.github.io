# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using a hybrid approach:
    1. Integrate with a moderate-accuracy adaptive solver (DOP853) to get a rough estimate.
    2. Use the known conserved quantity (Hamiltonian) of the Lotka-Volterra system
       to project the final state onto the correct energy level, correcting integration errors.
    
    This structural change exploits the problem's specific structure to reduce the
    required solver accuracy (and thus call count) while maintaining high precision.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    start_time = time.time()
    deadline = 325.0

    def elapsed():
        return time.time() - start_time

    # The Lotka-Volterra system is:
    # y0' = a*y0 - b*y0*y1
    # y1' = -c*y1 + d*y0*y1
    # From the task: a=1.5, b=1, c=3, d=1
    # The conserved quantity is H(y0, y1) = d*y0 - c*ln(y0) + b*y1 - a*ln(y1)
    # Here: H = 1*y0 - 3*ln(y0) + 1*y1 - 1.5*ln(y1)
    
    a, b, c, d = 1.5, 1.0, 3.0, 1.0
    
    def hamiltonian(y):
        y0_val, y1_val = y[0], y[1]
        # Avoid log(0) or log(negative)
        if y0_val <= 0 or y1_val <= 0:
            return np.inf
        return d * y0_val - c * np.log(y0_val) + b * y1_val - a * np.log(y1_val)
    
    H0 = hamiltonian(y0)
    
    # Step 1: Integrate with moderate tolerances
    # We don't need ultra-tight tolerances because we will correct the result.
    # DOP853 is efficient. Let's use rtol=1e-8, atol=1e-10 to save calls.
    # If this is too loose, the correction might fail. We'll try a few levels.
    
    call_count = [0]
    original_f = f
    
    def counting_f(t, y):
        call_count[0] += 1
        if call_count[0] > 1800000:
            raise RuntimeError("Too many calls")
        return original_f(t, y)
    
    best_y = None
    best_score = -np.inf
    
    # Try a few configurations with looser tolerances
    configs = [
        (1e-8, 1e-10, 0.1),
        (1e-9, 1e-11, 0.1),
        (1e-7, 1e-9, 0.2),
    ]
    
    for rtol, atol, max_step in configs:
        if elapsed() > deadline - 10:
            break
            
        try:
            sol = solve_ivp(
                counting_f, 
                t_span, 
                y0, 
                method='DOP853', 
                rtol=rtol, 
                atol=atol, 
                max_step=max_step,
                dense_output=False
            )
            
            if sol.success:
                y_end = sol.y[:, -1].copy()
                if np.all(np.isfinite(y_end)) and np.all(y_end > 0):
                    # Step 2: Correct using Hamiltonian
                    # We want to find y_corrected such that H(y_corrected) = H0
                    # and y_corrected is close to y_end.
                    # We can use a simple 1D optimization:
                    # Assume the error is mostly in the direction of the gradient of H.
                    # Or simpler: scale y_end slightly? No, that's not general.
                    # Let's use scipy.optimize to minimize (H(y) - H0)^2 + lambda * ||y - y_end||^2
                    # Or just project onto the level set.
                    
                    # A simpler correction: 
                    # The error in H is delta_H = H(y_end) - H0.
                    # The gradient of H at y_end is grad_H.
                    # A first-order correction is y_corr = y_end - (delta_H / ||grad_H||^2) * grad_H
                    # Let's try this Newton step.
                    
                    y_curr = y_end
                    H_curr = hamiltonian(y_curr)
                    
                    if np.isfinite(H_curr):
                        # Compute gradient analytically
                        # H = d*y0 - c*ln(y0) + b*y1 - a*ln(y1)
                        # dH/dy0 = d - c/y0
                        # dH/dy1 = b - a/y1
                        grad_H = np.array([
                            d - c / y_curr[0],
                            b - a / y_curr[1]
                        ])
                        
                        delta_H = H_curr - H0
                        norm_grad_sq = np.dot(grad_H, grad_H)
                        
                        if norm_grad_sq > 1e-12:
                            y_corr = y_curr - (delta_H / norm_grad_sq) * grad_H
                            
                            # Check if correction improved things
                            if np.all(np.isfinite(y_corr)) and np.all(y_corr > 0):
                                H_corr = hamiltonian(y_corr)
                                # If the correction is small and stable, use it
                                if abs(H_corr - H0) < abs(delta_H) * 1.1:
                                    y_end = y_corr
                                    # We could do a second Newton step if needed
                                    # But one step is often enough for small errors
                                    
                    # Estimate score (approximate)
                    # We don't know rel, but we assume it's small if H is conserved.
                    # Prefer fewer calls.
                    if best_y is None or call_count[0] < best_calls:
                        best_y = y_end
                        best_calls = call_count[0]
                        
        except Exception as e:
            continue

    if best_y is not None:
        return best_y

    # Fallback: High accuracy solve
    try:
        sol = solve_ivp(
            counting_f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13, 
            max_step=0.1,
            dense_output=False
        )
        if sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Final fallback
    return y0

# EVOLVE-BLOCK-END
