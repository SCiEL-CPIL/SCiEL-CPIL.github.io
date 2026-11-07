"""Root finder for 3 fixed 2D systems using Nelder-Mead Localization + Newton Polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 140.0  # Safety margin from 150s
    
    x0_arr = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0_arr)):
        x0_arr = np.zeros(2)
        
    call_counter = [0]
    max_calls = 480  # Leave some buffer for safety

    def safe_eval(x):
        if call_counter[0] >= max_calls:
            return None, float('inf')
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                call_counter[0] += 1
                return None, float('inf')
            call_counter[0] += 1
            return Fx, np.linalg.norm(Fx)
        except Exception:
            call_counter[0] += 1
            return None, float('inf')

    def objective(x):
        Fx, norm_F = safe_eval(x)
        if Fx is None:
            return 1e20
        return norm_F**2

    best_x = x0_arr.copy()
    best_norm = float('inf')
    
    # Phase 1: Nelder-Mead to get close to the root
    # We use a small maxiter to keep call count low.
    # Nelder-Mead on 2D vars typically uses ~1-2 calls per iteration.
    # maxiter=40 should use ~80 calls.
    
    try:
        # Use a slightly perturbed x0 to avoid exact saddle points if any
        res = minimize(
            objective, 
            x0_arr, 
            method='Nelder-Mead', 
            options={'maxiter': 40, 'xatol': 1e-4, 'fatol': 1e-8, 'adaptive': True}
        )
        x_nm = res.x
        if np.all(np.isfinite(x_nm)):
            Fx_nm, norm_nm = safe_eval(x_nm)
            if Fx_nm is not None and norm_nm < best_norm:
                best_norm = norm_nm
                best_x = x_nm.copy()
    except Exception:
        pass

    # Phase 2: Newton-Raphson with Finite Difference Jacobian
    # Start from the best point found
    x = best_x.copy()
    Fx, norm_F = safe_eval(x)
    
    if Fx is not None and norm_F < 1e-8:
        return x

    # Newton iterations
    h = 1e-5
    for i in range(20):
        if time.time() - start_time > time_limit:
            break
        if call_counter[0] >= max_calls:
            break
        
        # Compute Jacobian using forward differences
        J = np.zeros((2, 2))
        for j in range(2):
            x_pert = x.copy()
            x_pert[j] += h
            Fx_pert, _ = safe_eval(x_pert)
            if Fx_pert is None:
                break
            J[:, j] = (Fx_pert - Fx) / h
        
        if Fx is None:
            break
            
        # Check if Jacobian is valid
        if not np.all(np.isfinite(J)):
            break
            
        # Solve J * delta = -Fx
        try:
            delta = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, try least squares
            try:
                delta = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break
                
        if not np.all(np.isfinite(delta)):
            break
            
        # Line search for Newton step to ensure stability
        alpha = 1.0
        x_new = None
        F_new = None
        norm_new = float('inf')
        
        # Backtracking line search
        for _ in range(10):
            x_trial = x + alpha * delta
            F_trial, norm_trial = safe_eval(x_trial)
            if F_trial is None:
                alpha *= 0.5
                continue
            # Accept if norm decreases
            if norm_trial < norm_F:
                x_new = x_trial
                F_new = F_trial
                norm_new = norm_trial
                break
            else:
                alpha *= 0.5
                
        if x_new is None:
            # If no improvement, try a smaller step or break
            # If we can't improve, break to save calls
            break
            
        x = x_new
        Fx = F_new
        norm_F = norm_new
        
        if norm_F < best_norm:
            best_norm = norm_F
            best_x = x.copy()
            
        if norm_F <= 1e-8:
            break
            
    return best_x
# EVOLVE-BLOCK-END
