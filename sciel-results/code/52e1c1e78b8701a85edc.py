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
    """Compute Jacobian using central differences. Returns (J, Fx, fx_norm) or (None, None, inf)."""
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


def _nelder_mead(F, x0, max_iters=50):
    """
    Nelder-Mead simplex method for finding a root.
    Minimizes ||F(x)||^2.
    Returns (x, fx_norm).
    """
    # Initialize simplex
    x0 = np.array(x0, dtype=float)
    simplex = [x0.copy()]
    for i in range(2):
        xi = x0.copy()
        xi[i] += 0.1
        simplex.append(xi)
    
    # Evaluate initial simplex
    norms = []
    for i in range(3):
        _, n = _safe_F(F, simplex[i])
        norms.append(n)
    
    best_norm = min(norms)
    best_x = simplex[norms.index(best_norm)].copy()
    
    alpha = 1.0    # Reflection
    gamma = 2.0    # Expansion
    rho = 0.5      # Contraction
    sigma = 0.5    # Shrinkage
    
    for _ in range(max_iters):
        if not _time_ok():
            break
        
        # Sort simplex by norm
        order = np.argsort(norms)
        simplex = [simplex[i] for i in order]
        norms = [norms[i] for i in order]
        
        # Check convergence
        if norms[0] <= 1e-10:
            return simplex[0], norms[0]
        
        if norms[0] < best_norm:
            best_norm = norms[0]
            best_x = simplex[0].copy()
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        _, fr = _safe_F(F, xr)
        
        if fr < norms[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            _, fe = _safe_F(F, xe)
            if fe < fr:
                simplex[-1] = xe
                norms[-1] = fe
            else:
                simplex[-1] = xr
                norms[-1] = fr
        elif fr < norms[-2]:
            # Simple reflection
            simplex[-1] = xr
            norms[-1] = fr
        else:
            # Contraction
            if fr < norms[-1]:
                # Outside contraction
                xc = centroid + 0.5 * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid + 0.5 * (simplex[-1] - centroid)
            _, fc = _safe_F(F, xc)
            if fc < min(fr, norms[-1]):
                simplex[-1] = xc
                norms[-1] = fc
            else:
                # Shrinkage
                for i in range(1, 3):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    _, ni = _safe_F(F, simplex[i])
                    norms[i] = ni
        
        # Re-sort after modification
        order = np.argsort(norms)
        simplex = [simplex[i] for i in order]
        norms = [norms[i] for i in order]
    
    return best_x, best_norm


def _newton_polish(F, x, Fx, fx_norm, max_steps=5):
    """
    Perform Newton steps with Central Difference Jacobian for high-precision polishing.
    Returns (x, Fx, fx_norm).
    """
    for _ in range(max_steps):
        if not _time_ok():
            break
        
        J, Fx_new, fx_new_norm = _compute_jacobian_cd(F, x, h=1e-5)
        if J is None:
            break
        
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


def _broyden_method(F, x, Fx, fx_norm, max_iters=50):
    """
    Broyden's Good Method with forward difference initialization.
    Returns (x, Fx, fx_norm).
    """
    # Initialize Jacobian with forward differences
    J = _compute_jacobian_fd(F, x, Fx, h=1e-6)
    if J is None:
        return x, Fx, fx_norm
    
    best_x = x.copy()
    best_norm = fx_norm
    
    for _ in range(max_iters):
        if not _time_ok():
            break

        # Solve J s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Refresh Jacobian with FD and retry
            J_new = _compute_jacobian_fd(F, x, Fx, h=1e-5)
            if J_new is None:
                break
            J = J_new
            try:
                s = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break

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
                    return x, Fx, fx_norm

                converged = True
                break
            alpha *= 0.5

        if not converged:
            break

    return best_x, None, best_norm


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

    # Phase 1: Try Broyden's method first (efficient for well-behaved systems)
    x_b, Fx_b, norm_b = _broyden_method(F, x, Fx, fx_norm, max_iters=30)
    if norm_b < best_norm:
        best_norm = norm_b
        best_x = x_b.copy()
    
    if best_norm <= 1e-10:
        return best_x

    # Phase 2: If Broyden didn't converge well, try Nelder-Mead for global exploration
    # Use the best point from Broyden as starting point for NM
    x_nm, norm_nm = _nelder_mead(F, best_x, max_iters=50)
    if norm_nm < best_norm:
        best_norm = norm_nm
        best_x = x_nm.copy()
    
    if best_norm <= 1e-10:
        return best_x

    # Phase 3: Polish with Newton (central differences) for high precision
    if best_norm < 0.1:  # Only polish if reasonably close
        x_pol, Fx_pol, norm_pol = _newton_polish(F, best_x, None, best_norm, max_steps=5)
        if norm_pol < best_norm:
            best_norm = norm_pol
            best_x = x_pol.copy()

    return best_x
# EVOLVE-BLOCK-END
