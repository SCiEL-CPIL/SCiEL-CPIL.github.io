"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill


def _time_ok():
    return (time.time() - _start_time) < _DEADLINE


def _safe_F(F, x):
    """Call F and return (Fx, fx_norm) or (None, inf) on failure."""
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return None, np.inf
        return Fx, np.linalg.norm(Fx)
    except Exception:
        return None, np.inf


def _compute_jacobian_fd(F, x, Fx, h=1e-6):
    """Compute Jacobian using forward differences. Returns J or None on failure."""
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fp, _ = _safe_F(F, x + e)
        if Fp is None:
            return None
        J[:, j] = (Fp - Fx) / h
    if not np.all(np.isfinite(J)):
        return None
    return J

def _compute_jacobian_cd(F, x, h=1e-5):
    """Compute Jacobian using central differences. Returns J, Fx or (None, None, inf) on failure."""
    J = np.zeros((2, 2))
    Fx, fx_norm = _safe_F(F, x)
    if Fx is None:
        return None, None, np.inf
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fp, _ = _safe_F(F, x + e)
        Fm, _ = _safe_F(F, x - e)
        if Fp is None or Fm is None:
            return None, None, np.inf
        J[:, j] = (Fp - Fm) / (2 * h)
    if not np.all(np.isfinite(J)):
        return None, None, np.inf
    return J, Fx, fx_norm

def _newton_cd_polish(F, x, Fx, fx_norm, max_steps=5):
    """
    Perform Newton steps with Central Difference Jacobian.
    Used for high-precision polishing or robust fallback.
    Returns (x, Fx, fx_norm)
    """
    for _ in range(max_steps):
        if not _time_ok():
            break
        
        J, Fx_new, fx_new_norm = _compute_jacobian_cd(F, x, h=1e-5)
        if J is None:
            break
        
        # Use the fresh Fx from _compute_jacobian_cd
        Fx = Fx_new
        fx_norm = fx_new_norm
        
        if fx_norm <= 1e-10:
            return x, Fx, fx_norm

        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break

        if not np.all(np.isfinite(s)):
            break

        # Line search
        alpha = 1.0
        converged = False
        for _ in range(10):
            x_new = x + alpha * s
            Fx_ls, fx_ls_norm = _safe_F(F, x_new)
            if Fx_ls is None:
                alpha *= 0.5
                continue

            if fx_ls_norm < fx_norm:
                x = x_new
                Fx = Fx_ls
                fx_norm = fx_ls_norm
                if fx_norm <= 1e-10:
                    return x, Fx, fx_norm
                converged = True
                break
            alpha *= 0.5
        
        if not converged:
            break
            
    return x, Fx, fx_norm


def find_root_system(F, x0):
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf

    # Initial evaluation
    Fx, fx_norm = _safe_F(F, x)
    if Fx is None:
        return best_x

    if fx_norm < best_norm:
        best_norm = fx_norm
        best_x = x.copy()

    if fx_norm <= 1e-10:
        return x

    # Initialize Jacobian with forward differences
    J = _compute_jacobian_fd(F, x, Fx, h=1e-6)
    if J is None:
        # Fallback to Newton CD immediately
        x, Fx, fx_norm = _newton_cd_polish(F, x, Fx, fx_norm, max_steps=20)
        return x

    # Broyden's Good Method
    max_iters = 30
    for _ in range(max_iters):
        if not _time_ok():
            break

        # Solve J s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Refresh Jacobian with CD and try Newton
            x, Fx, fx_norm = _newton_cd_polish(F, x, Fx, fx_norm, max_steps=20)
            return x

        if not np.all(np.isfinite(s)):
            break

        # Line search with backtracking
        alpha = 1.0
        converged = False
        for _ in range(20):
            x_new = x + alpha * s
            Fx_new, fx_new_norm = _safe_F(F, x_new)
            if Fx_new is None:
                alpha *= 0.5
                continue

            if fx_new_norm < fx_norm:
                # Broyden update
                y = Fx_new - Fx
                s_actual = x_new - x
                denom = s_actual @ s_actual
                if denom > 1e-15:
                    J = J + np.outer(y - J @ s_actual, s_actual) / denom

                x = x_new
                Fx = Fx_new
                fx_norm = fx_new_norm

                if fx_norm < best_norm:
                    best_norm = fx_norm
                    best_x = x.copy()

                if fx_norm <= 1e-10:
                    return x

                converged = True
                break
            alpha *= 0.5

        if not converged:
            break

    # If we are close but not quite there, polish with Newton-CD
    # This handles cases where Broyden's approximation prevents final convergence
    if best_norm < 1e-4 and best_norm > 1e-8:
         x_polished, Fx_polished, fx_polished = _newton_cd_polish(F, best_x, None, np.inf, max_steps=3)
         if fx_polished < best_norm:
             return x_polished
         return best_x

    return best_x

# EVOLVE-BLOCK-END
