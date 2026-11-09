"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 such that ||F(x)|| <= 1e-8 using a robust
    combination of:
      1. Dogleg trust-region method on ||F(x)||^2 (derivative-free)
      2. Damped Newton with finite-difference Jacobian as fallback
    
    Includes deadline guard for 120s wall-clock limit.
    """
    deadline = time.time() + 115.0  # safety margin before 120s limit
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf
    
    def update_best(x_candidate, Fx_candidate):
        nonlocal best_x, best_norm
        nrm = np.linalg.norm(Fx_candidate)
        if nrm < best_norm:
            best_norm = nrm
            best_x = x_candidate.copy()
    
    # Check initial point
    Fx0 = F(x)
    update_best(x, Fx0)
    if best_norm <= 1e-8:
        return best_x
    
    # --- Phase 1: Dogleg trust-region on ||F||^2 ---
    # We'll implement a simple trust-region dogleg method
    # f(x) = 0.5 * ||F(x)||^2, gradient = J^T F, Hessian approx = J^T J
    
    def objective_and_grad(x_pt):
        Fx_pt = F(x_pt)
        f_val = 0.5 * np.dot(Fx_pt, Fx_pt)
        # Finite difference Jacobian
        J = np.zeros((2, 2))
        h = 1e-5
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = F(x_pt + e)
            J[:, j] = (Fx_plus - Fx_pt) / h
        grad = J.T @ Fx_pt
        return f_val, grad, J, Fx_pt
    
    # Initialize trust region
    delta = 1.0  # initial trust region radius
    f_val, grad, J, Fx = objective_and_grad(x)
    update_best(x, Fx)
    
    # Gauss-Newton step: solve J^T J dx = -J^T F
    try:
        JTJ = J.T @ J
        JTF = J.T @ Fx
        # Regularize if needed
        reg = 1e-10 * np.trace(JTJ) / 2
        dx_full = np.linalg.solve(JTJ + reg * np.eye(2), -JTF)
    except np.linalg.LinAlgError:
        dx_full = -grad / (np.linalg.norm(grad) + 1e-16)
    
    grad_norm = np.linalg.norm(grad)
    if grad_norm < 1e-15:
        # At a critical point of the objective
        pass
    
    # Steepest descent step
    dx_sd = -grad / (grad_norm + 1e-16)
    dx_sd = dx_sd * min(np.linalg.norm(dx_sd), delta)
    
    # Dogleg step
    dx_full_norm = np.linalg.norm(dx_full)
    if dx_full_norm <= delta:
        dx_dl = dx_full
    else:
        # Find the point on the line from 0 to dx_full with norm = delta
        # Actually, dogleg goes: steepest descent to the point where it meets the
        # line from 0 to dx_full, then along that line to the trust boundary
        # Simplified: just use damped full Newton
        dx_dl = dx_full * (delta / dx_full_norm)
    
    # Evaluate step
    x_new = x + dx_dl
    Fx_new = F(x_new)
    f_new = 0.5 * np.dot(Fx_new, Fx_new)
    update_best(x_new, Fx_new)
    
    actual_reduction = f_val - f_new
    predicted_reduction = -0.5 * np.dot(grad, dx_dl) + 0.5 * np.dot(dx_dl, J.T @ J @ dx_dl)
    
    rho = actual_reduction / max(abs(predicted_reduction), 1e-16)
    
    max_iter = 100
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if best_norm <= 1e-8:
            return best_x
        
        # Adjust trust region
        if rho < 0.25:
            delta = 0.5 * delta
            continue
        elif rho < 0.75:
            delta = delta  # keep same
        else:
            delta = min(2.0 * delta, 10.0)
        
        x = x_new
        f_val = f_new
        
        # Recompute gradient and Jacobian
        f_val, grad, J, Fx = objective_and_grad(x)
        update_best(x, Fx)
        if best_norm <= 1e-8:
            return best_x
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        try:
            JTJ = J.T @ J
            JTF = J.T @ Fx
            reg = 1e-10 * np.trace(JTJ) / 2
            dx_full = np.linalg.solve(JTJ + reg * np.eye(2), -JTF)
        except np.linalg.LinAlgError:
            dx_full = -grad / (grad_norm + 1e-16)
        
        dx_full_norm = np.linalg.norm(dx_full)
        if dx_full_norm <= delta:
            dx_dl = dx_full
        else:
            dx_dl = dx_full * (delta / dx_full_norm)
        
        x_new = x + dx_dl
        Fx_new = F(x_new)
        f_new = 0.5 * np.dot(Fx_new, Fx_new)
        update_best(x_new, Fx_new)
        
        actual_reduction = f_val - f_new
        predicted_reduction = -0.5 * np.dot(grad, dx_dl) + 0.5 * np.dot(dx_dl, J.T @ J @ dx_dl)
        rho = actual_reduction / max(abs(predicted_reduction), 1e-16)
    
    # --- Phase 2: Damped Newton fallback ---
    if best_norm > 1e-8 and time.time() < deadline:
        x = best_x.copy()
        for it in range(80):
            if time.time() > deadline:
                break
            Fx = F(x)
            nrm = np.linalg.norm(Fx)
            update_best(x, Fx)
            if best_norm <= 1e-8:
                return best_x
            
            # Finite difference Jacobian
            J = np.zeros((2, 2))
            h = 1e-6
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                J[:, j] = (F(x + e) - Fx) / h
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            # Damping
            lam = 1.0
            while lam > 1e-12:
                x_trial = x + lam * dx
                Fx_trial = F(x_trial)
                nrm_trial = np.linalg.norm(Fx_trial)
                if nrm_trial < nrm:
                    x = x_trial
                    update_best(x, Fx_trial)
                    if best_norm <= 1e-8:
                        return best_x
                    break
                lam *= 0.5
            else:
                break
            
            if lam < 1e-12:
                break
    
    # --- Phase 3: If still not converged, try from multiple starting points ---
    if best_norm > 1e-8 and time.time() < deadline:
        # Try a few random perturbations
        rng = np.random.RandomState(42)
        for _ in range(10):
            if time.time() > deadline:
                break
            x_try = best_x + rng.normal(0, 0.1, size=2)
            Fx_try = F(x_try)
            nrm_try = np.linalg.norm(Fx_try)
            if nrm_try < best_norm:
                best_x = x_try
                best_norm = nrm_try
                if best_norm <= 1e-8:
                    return best_x
                # Run a few Newton steps from here
                x = x_try.copy()
                for inner in range(30):
                    if time.time() > deadline:
                        break
                    Fx = F(x)
                    nrm = np.linalg.norm(Fx)
                    update_best(x, Fx)
                    if best_norm <= 1e-8:
                        return best_x
                    J = np.zeros((2, 2))
                    h = 1e-6
                    for j in range(2):
                        e = np.zeros(2)
                        e[j] = h
                        J[:, j] = (F(x + e) - Fx) / h
                    try:
                        dx = np.linalg.solve(J, -Fx)
                    except np.linalg.LinAlgError:
                        break
                    lam = 1.0
                    while lam > 1e-12:
                        x_trial = x + lam * dx
                        Fx_trial = F(x_trial)
                        nrm_trial = np.linalg.norm(Fx_trial)
                        if nrm_trial < nrm:
                            x = x_trial
                            update_best(x, Fx_trial)
                            if best_norm <= 1e-8:
                                return best_x
                            break
                        lam *= 0.5
                    if lam < 1e-12:
                        break
    
    return best_x
# EVOLVE-BLOCK-END
