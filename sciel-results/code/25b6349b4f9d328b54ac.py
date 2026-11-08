"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 using a robust hybrid approach:
    1. Initial Newton iterations with finite-difference Jacobian.
    2. If Newton stalls or diverges, fall back to Broyden's method (rank-1 update).
    3. If still not converged, try a damped Newton with line search.
    4. Time-guarded: return best result before 140s wall-clock (safety margin).
    
    Budget: <500 calls to F.
    """
    x = np.array(x0, dtype=float, copy=True)
    deadline = time.time() + 135.0  # safety margin
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _call_F(x_val):
        nonlocal call_count
        if time.time() > deadline:
            return None  # signal to stop
        if call_count >= 498:  # leave 2 calls for safety
            return None
        call_count += 1
        result = F(x_val)
        result = np.asarray(result, dtype=float)
        if not np.all(np.isfinite(result)):
            return None
        return result

    # --- Phase 1: Newton with finite-difference Jacobian ---
    Fx = _call_F(x)
    if Fx is None:
        return best_x if best_norm < np.inf else x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # Compute initial Jacobian
    J = np.zeros((2, 2))
    h = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_e = _call_F(x + e)
        if Fx_e is None:
            break
        J[:, j] = (Fx_e - Fx) / h

    # Newton iterations
    for it in range(20):
        if time.time() > deadline:
            break
        if call_count >= 480:  # leave room for safety
            break

        Fx = _call_F(x)
        if Fx is None:
            break
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x
        if norm_Fx < best_norm:
            best_x = x.copy()
            best_norm = norm_Fx

        # Recompute Jacobian every 3 steps to avoid stale Jacobian
        if it % 3 == 0 or it == 0:
            h = 1e-6
            Fx_ref = _call_F(x)
            if Fx_ref is None:
                break
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_e = _call_F(x + e)
                if Fx_e is None:
                    break
                J[:, j] = (Fx_e - Fx_ref) / h

        # Try Newton step
        try:
            dx = np.linalg.solve(J, Fx)
            # Damping: if step is too large, scale it
            norm_dx = np.linalg.norm(dx)
            if norm_dx > 1.0:
                dx = dx / norm_dx
            x_new = x - dx
            # Check if x_new is finite
            if not np.all(np.isfinite(x_new)):
                # Dampen more
                dx = dx * 0.5
                x_new = x - dx
                if not np.all(np.isfinite(x_new)):
                    break
            x = x_new
        except np.linalg.LinAlgError:
            # J is singular; use Broyden-style update or break
            break

    # --- Phase 2: Broyden's method (rank-1) ---
    x = x.copy()
    Fx = _call_F(x)
    if Fx is None:
        return best_x if best_norm < np.inf else x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    if norm_Fx < best_norm:
        best_x = x.copy()
        best_norm = norm_Fx

    # Ensure J is valid
    if not np.all(np.isfinite(J)):
        # Recompute J
        h = 1e-6
        Fx_ref = _call_F(x)
        if Fx_ref is None:
            return best_x if best_norm < np.inf else x
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = _call_F(x + e)
            if Fx_e is None:
                return best_x if best_norm < np.inf else x
            J[:, j] = (Fx_e - Fx_ref) / h

    for it in range(150):  # up to 150 Broyden steps, each 1 call = 150 calls total for this phase
        if time.time() > deadline:
            break
        if call_count >= 495:
            break

        Fx = _call_F(x)
        if Fx is None:
            break
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x
        if norm_Fx < best_norm:
            best_x = x.copy()
            best_norm = norm_Fx

        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse
            dx = np.linalg.pinv(J) @ (-Fx)

        # Damp if needed
        norm_dx = np.linalg.norm(dx)
        if norm_dx > 10.0:
            dx = dx / norm_dx * 10.0

        x_new = x + dx
        if not np.all(np.isfinite(x_new)):
            dx = dx * 0.1
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                break

        Fx_new = _call_F(x_new)
        if Fx_new is None:
            break
        norm_Fx_new = np.linalg.norm(Fx_new)

        if norm_Fx_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_Fx_new

        # Broyden rank-1 update
        s = x_new - x
        y = Fx_new - Fx
        if np.linalg.norm(s) > 1e-15:
            J = J + np.outer(y - J @ s, s.T) / (s.T @ s)

        x = x_new
        Fx = Fx_new
        if norm_Fx_new <= 1e-8:
            return x

    # --- Phase 3: Final check and return best ---
    Fx_final = _call_F(x)
    if Fx_final is not None:
        norm_final = np.linalg.norm(Fx_final)
        if norm_final <= 1e-8:
            return x
        if norm_final < best_norm:
            best_x = x.copy()
            best_norm = norm_final

    return best_x
# EVOLVE-BLOCK-END
