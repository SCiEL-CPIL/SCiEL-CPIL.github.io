"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 using Broyden's Good Method.
    This method is efficient as it requires only 1 function call per iteration
    after the initial Jacobian setup.
    """
    x = np.array(x0, dtype=float, copy=True)
    deadline = time.time() + 140.0  # Safety margin
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf
    
    def _call_F(x_val):
        nonlocal call_count
        if time.time() > deadline:
            return None
        if call_count >= 490:  # Leave buffer
            return None
        call_count += 1
        try:
            result = F(x_val)
            result = np.asarray(result, dtype=float)
            if not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            return None

    # 1. Initial Evaluation
    Fx = _call_F(x)
    if Fx is None:
        return best_x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # 2. Initial Jacobian (Forward Difference)
    J = np.zeros((2, 2))
    h = 1e-5
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = _call_F(x + e)
        if Fx_plus is None:
            return best_x
        J[:, j] = (Fx_plus - Fx) / h
    
    if not np.all(np.isfinite(J)) or np.linalg.cond(J) > 1e10:
        J = np.eye(2)

    # 3. Broyden's Good Method Loop
    for it in range(400):
        if time.time() > deadline:
            break
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Fallback to least squares or pseudo-inverse if singular
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                dx = -Fx # Steepest descent fallback
        
        # Limit step size
        norm_dx = np.linalg.norm(dx)
        if norm_dx > 10.0:
            dx = dx / norm_dx * 10.0

        # Line Search (Backtracking)
        alpha = 1.0
        x_new = x
        Fx_new = None
        accepted = False
        
        for ls in range(20):
            x_trial = x + alpha * dx
            if not np.all(np.isfinite(x_trial)):
                alpha *= 0.5
                continue
                
            Fx_trial = _call_F(x_trial)
            if Fx_trial is None:
                break
            
            norm_trial = np.linalg.norm(Fx_trial)
            
            if norm_trial < best_norm:
                best_x = x_trial.copy()
                best_norm = norm_trial
            
            if norm_trial <= 1e-8:
                return x_trial
            
            # Armijo condition: sufficient decrease
            if norm_trial < norm_Fx - 1e-4 * alpha * (dx @ Fx):
                x_new = x_trial
                Fx_new = Fx_trial
                accepted = True
                break
            else:
                alpha *= 0.5
        
        if not accepted:
            # If line search fails, try a smaller step or break
            # If we couldn't improve, we might be stuck.
            # Try a restart strategy: perturb x slightly and recompute J?
            # For simplicity, break and try a different strategy if this fails completely.
            break

        # Broyden Rank-1 Update
        s = x_new - x
        y = Fx_new - Fx
        s_norm_sq = s.T @ s
        if s_norm_sq > 1e-15:
            J = J + np.outer(y - J @ s, s) / s_norm_sq
        
        x = x_new
        Fx = Fx_new
        norm_Fx = np.linalg.norm(Fx)
        
        # Check convergence
        if norm_Fx <= 1e-8:
            return x

    # 4. Fallback: Simple Newton with Central Differences if Broyden failed
    # This uses more calls but is more robust for difficult cases.
    if best_norm > 1e-8:
        x = best_x
        Fx = _call_F(x)
        if Fx is None:
            return best_x
            
        J = np.zeros((2, 2))
        h = 1e-5
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fp = _call_F(x + e)
            Fm = _call_F(x - e)
            if Fp is None or Fm is None:
                return best_x
            J[:, j] = (Fp - Fm) / (2 * h)
            
        if not np.all(np.isfinite(J)):
            J = np.eye(2)

        for it in range(50):
            if time.time() > deadline:
                break
            if call_count >= 495:
                break
                
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                
            alpha = 1.0
            for ls in range(20):
                x_trial = x + alpha * dx
                if not np.all(np.isfinite(x_trial)):
                    alpha *= 0.5
                    continue
                Fx_trial = _call_F(x_trial)
                if Fx_trial is None:
                    break
                norm_trial = np.linalg.norm(Fx_trial)
                if norm_trial < best_norm:
                    best_x = x_trial.copy()
                    best_norm = norm_trial
                if norm_trial <= 1e-8:
                    return x_trial
                if norm_trial < norm_Fx:
                    x = x_trial
                    Fx = Fx_trial
                    break
                alpha *= 0.5
            
            # Update J every few steps
            if (it + 1) % 5 == 0:
                J = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    Fp = _call_F(x + e)
                    Fm = _call_F(x - e)
                    if Fp is None or Fm is None:
                        break
                    J[:, j] = (Fp - Fm) / (2 * h)
                if not np.all(np.isfinite(J)):
                    J = np.eye(2)

    return best_x
# EVOLVE-BLOCK-END
