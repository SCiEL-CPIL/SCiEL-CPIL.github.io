"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = float('inf')
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin from 150s limit
    
    # Initial evaluation
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,):
            Fx = Fx.reshape(2)
        if not np.all(np.isfinite(Fx)):
            return best_x
        norm = np.linalg.norm(Fx)
        if norm < best_norm:
            best_norm = norm
            best_x = x.copy()
        if norm <= 1e-8:
            return best_x
    except Exception:
        return best_x

    calls = 1

    # Compute initial Jacobian using finite differences
    h = 1e-6
    J = np.zeros((2, 2))
    jacobian_valid = True
    try:
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_j = F(x + e)
            Fx_j = np.asarray(Fx_j, dtype=float)
            if Fx_j.shape != (2,):
                Fx_j = Fx_j.reshape(2)
            if not np.all(np.isfinite(Fx_j)):
                jacobian_valid = False
                break
            J[:, j] = (Fx_j - Fx) / h
            calls += 1
    except Exception:
        jacobian_valid = False

    if not jacobian_valid:
        J = np.eye(2)

    # Broyden's method
    max_iter = 490  # Use most of the call budget
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if calls >= 495:  # Safety margin for call limit
            break

        try:
            # Solve J * dx = -Fx
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, try adding damping to diagonal or restart
            # Simple fallback: break
            break

        # Line search
        alpha = 1.0
        Fx_old = Fx
        Fx_old_norm = np.linalg.norm(Fx_old)
        step_accepted = False
        
        # Backtracking line search
        for _ in range(20):
            x_try = x + alpha * dx
            try:
                Fx_try = F(x_try)
                Fx_try = np.asarray(Fx_try, dtype=float)
                if Fx_try.shape != (2,):
                    Fx_try = Fx_try.reshape(2)
                if not np.all(np.isfinite(Fx_try)):
                    alpha *= 0.5
                    continue
                norm_try = np.linalg.norm(Fx_try)
                calls += 1
                
                if norm_try < best_norm:
                    best_norm = norm_try
                    best_x = x_try.copy()
                
                if norm_try < Fx_old_norm:
                    # Accept step
                    # Update Jacobian (Broyden)
                    s = alpha * dx
                    y = Fx_try - Fx_old
                    # J_new = J + (y - J*s) * s^T / (s^T * s)
                    J_s = J @ s
                    diff = y - J_s
                    denom = s @ s
                    if denom > 1e-15:
                        J = J + np.outer(diff, s) / denom
                    else:
                        J = J # Fallback, no update
                    x = x_try
                    Fx = Fx_try
                    step_accepted = True
                    break
                else:
                    alpha *= 0.5
            except Exception:
                alpha *= 0.5
                continue
        
        if not step_accepted:
            break

    return best_x
# EVOLVE-BLOCK-END
