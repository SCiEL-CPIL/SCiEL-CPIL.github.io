# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 130.0  # safety margin before 150s kill

def _safe_norm(F, x):
    """Safely compute ||F(x)||, returning inf on error."""
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return np.inf
        return float(np.linalg.norm(Fx))
    except Exception:
        return np.inf

def _damped_newton(F, x, max_iter=40, tol=1e-10):
    """
    Damped Newton's method with forward-difference Jacobian and backtracking line search.
    Returns (x, norm_F).
    """
    x = np.array(x, dtype=float)
    t0 = time.time()
    
    for _ in range(max_iter):
        if time.time() - t0 > 5.0:  # per-Newton timeout
            break
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if not np.all(np.isfinite(Fx)):
                return x, np.inf
            norm_Fx = float(np.linalg.norm(Fx))
            if norm_Fx <= tol:
                return x, norm_Fx
            
            # Forward-difference Jacobian (2 extra calls)
            h = 1e-6
            J = np.empty((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_p = F(x + e)
                Fx_p = np.asarray(Fx_p, dtype=float)
                if not np.all(np.isfinite(Fx_p)):
                    return x, norm_Fx
                J[:, j] = (Fx_p - Fx) / h
            
            # Solve J * dx = -Fx  =>  x_new = x + dx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # If singular, use least squares
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            
            if not np.all(np.isfinite(dx)):
                return x, norm_Fx
            
            # Backtracking line search
            alpha = 1.0
            for _ls in range(30):
                x_trial = x + alpha * dx
                if not np.all(np.isfinite(x_trial)):
                    alpha *= 0.5
                    continue
                Fx_trial = F(x_trial)
                Fx_trial = np.asarray(Fx_trial, dtype=float)
                if not np.all(np.isfinite(Fx_trial)):
                    alpha *= 0.5
                    continue
                norm_trial = float(np.linalg.norm(Fx_trial))
                if norm_trial < norm_Fx * (1.0 - 1e-4 * alpha):
                    # Sufficient decrease
                    x = x_trial
                    norm_Fx = norm_trial
                    if norm_Fx <= tol:
                        return x, norm_Fx
                    break
                alpha *= 0.5
            else:
                # If no progress after 30 line search steps, accept the trial
                x_trial = x + alpha * dx
                if np.all(np.isfinite(x_trial)):
                    Fx_trial = F(x_trial)
                    Fx_trial = np.asarray(Fx_trial, dtype=float)
                    if np.all(np.isfinite(Fx_trial)):
                        x = x_trial
                        norm_Fx = float(np.linalg.norm(Fx_trial))
                break
        except Exception:
            break
    
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if not np.all(np.isfinite(Fx)):
            return x, np.inf
        return x, float(np.linalg.norm(Fx))
    except Exception:
        return x, np.inf

def find_root_system(F, x0):
    """
    Finds a root x such that ||F(x)|| <= 1e-8.
    """
    start_time = time.time()
    x0 = np.array(x0, dtype=float)
    
    # 1. Check if x0 is already a root
    norm0 = _safe_norm(F, x0)
    if norm0 <= 1e-8:
        return x0
    
    best_x = x0
    best_norm = norm0
    
    # 2. Generate adaptive initial points
    # Scale radii based on initial norm magnitude
    base_scale = max(0.5, min(10.0, np.sqrt(norm0) if norm0 < np.inf else 1.0))
    
    initial_points = [x0]
    
    # Radial sampling: multiple radii, 12 angles each
    radii = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    for r in radii:
        r_scaled = r * base_scale
        for theta in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            x_try = x0 + r_scaled * np.array([np.cos(theta), np.sin(theta)])
            initial_points.append(x_try)
    
    # Add axis-aligned points for better coverage
    for r in [0.5, 1.0, 5.0, 20.0]:
        r_scaled = r * base_scale
        for sign in [1.0, -1.0]:
            initial_points.append(x0 + sign * r_scaled * np.array([1.0, 0.0]))
            initial_points.append(x0 + sign * r_scaled * np.array([0.0, 1.0]))
            initial_points.append(x0 + sign * r_scaled * np.array([1.0, 1.0]) / np.sqrt(2))
    
    # 3. Screen initial points (evaluate F at each)
    screened = []
    for x_try in initial_points:
        if time.time() - start_time > _DEADLINE * 0.25:
            break
        norm_try = _safe_norm(F, x_try)
        screened.append((norm_try, x_try))
        if norm_try < best_norm:
            best_norm = norm_try
            best_x = x_try
        if norm_try <= 1e-8:
            return x_try
    
    # Sort by norm (best first)
    screened.sort(key=lambda t: t[0])
    
    # 4. Try damped Newton from the top candidates
    # Try up to 8 best points, or until time runs out
    max_newton_tries = 8
    for i, (norm_try, x_try) in enumerate(screened[:max_newton_tries]):
        if time.time() - start_time > _DEADLINE * 0.8:
            break
        x_new, norm_new = _damped_newton(F, x_try, max_iter=40, tol=1e-10)
        if norm_new <= 1e-8:
            return x_new
        if norm_new < best_norm:
            best_norm = norm_new
            best_x = x_new
    
    # 5. If Newton didn't fully converge, try a few more with different starts
    # Use points that weren't in the top 8 but are still reasonable
    if time.time() - start_time < _DEADLINE * 0.9:
        for i, (norm_try, x_try) in enumerate(screened[max_newton_tries:max_newton_tries+4]):
            if time.time() - start_time > _DEADLINE * 0.9:
                break
            x_new, norm_new = _damped_newton(F, x_try, max_iter=25, tol=1e-10)
            if norm_new <= 1e-8:
                return x_new
            if norm_new < best_norm:
                best_norm = norm_new
                best_x = x_new
    
    # 6. Final attempt: try a few random-ish points if we have time
    if time.time() - start_time < _DEADLINE * 0.95:
        # Deterministic "random" points using a simple LCG
        seed = 42
        for _ in range(6):
            if time.time() - start_time > _DEADLINE * 0.95:
                break
            seed = (seed * 1103515245 + 12345) & 0x7fffffff
            r1 = (seed >> 16) / 65536.0
            seed = (seed * 1103515245 + 12345) & 0x7fffffff
            r2 = (seed >> 16) / 65536.0
            seed = (seed * 1103515245 + 12345) & 0x7fffffff
            r3 = (seed >> 16) / 65536.0
            
            r_rand = r1 * 100.0 * base_scale
            theta_rand = r2 * 2 * np.pi
            x_rand = x0 + r_rand * np.array([np.cos(theta_rand), np.sin(theta_rand)])
            
            norm_rand = _safe_norm(F, x_rand)
            if norm_rand <= 1e-8:
                return x_rand
            if norm_rand < best_norm:
                best_norm = norm_rand
                best_x = x_rand
            
            x_new, norm_new = _damped_newton(F, x_rand, max_iter=20, tol=1e-10)
            if norm_new <= 1e-8:
                return x_new
            if norm_new < best_norm:
                best_norm = norm_new
                best_x = x_new
    
    # 7. Return the best point found
    return best_x
# EVOLVE-BLOCK-END
