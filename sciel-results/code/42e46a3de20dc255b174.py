"""Root finder for 3 fixed two-dimensional systems using hybrid Newton-Broyden-Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _safe_F(F, x, call_count):
    """Call F safely, returning a finite array of length 2 or None on failure."""
    try:
        y = F(x)
        y = np.asarray(y, dtype=float)
        if y.shape != (2,) or not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None


def _broyden_step(x, Fx, J, F, call_count):
    """One Broyden step: solve J*dx = -Fx, update J, return new x, Fx, J."""
    try:
        dx = np.linalg.solve(J, -Fx)
    except np.linalg.LinAlgError:
        return None, None, None
    
    x_new = x + dx
    F_new = _safe_F(F, x_new, call_count)
    if F_new is None:
        return None, None, None
    
    # Rank-1 Broyden update
    s = x_new - x
    y = F_new - Fx
    denom = np.dot(y, s)
    if abs(denom) > 1e-14:
        J = J + np.outer(y - J @ s, s) / denom
    
    return x_new, F_new, J


def _nelder_mead(x0, F, call_count, max_iter=200):
    """Nelder-Mead on Phi(x) = 0.5*||F(x)||^2."""
    def phi(x):
        Fx = _safe_F(F, x, call_count)
        if Fx is None:
            return np.inf
        return 0.5 * np.dot(Fx, Fx)
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(2):
        offset = x0.copy()
        offset[i] += 0.1 * (abs(x0[i]) + 1.0)
        simplex.append(offset)
    
    vals = [phi(v) for v in simplex]
    best_idx = np.argmin(vals)
    best_x = simplex[best_idx].copy()
    best_val = vals[best_idx]
    
    for _ in range(max_iter):
        if best_val <= 1e-16:  # ||F||^2/2 <= 1e-16 means ||F|| <= 1e-8
            return best_x
        
        # Sort by value
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        best_idx = 0
        best_x = simplex[0].copy()
        best_val = vals[0]
        worst = simplex[-1]
        worst_val = vals[-1]
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_ref = centroid + alpha * (centroid - worst)
        val_ref = phi(x_ref)
        
        if val_ref < vals[0]:
            # Expansion
            gamma = 2.0
            x_exp = centroid + gamma * (centroid - worst)
            val_exp = phi(x_exp)
            if val_exp < val_ref:
                simplex[-1] = x_exp
                vals[-1] = val_exp
            else:
                simplex[-1] = x_ref
                vals[-1] = val_ref
        elif val_ref < vals[-2]:
            simplex[-1] = x_ref
            vals[-1] = val_ref
        else:
            # Contraction
            beta = 0.5
            x_con = centroid + beta * (worst - centroid)
            val_con = phi(x_con)
            if val_con < worst_val:
                simplex[-1] = x_con
                vals[-1] = val_con
            else:
                # Shrink
                for i in range(1, 3):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = phi(simplex[i])
    
    return best_x


def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    x = np.array(x0, dtype=float)
    call_count = [0]  # mutable counter
    
    def F_counted(x):
        call_count[0] += 1
        return F(x)
    
    # Try Newton's method first
    best_x = x.copy()
    best_norm = np.inf
    
    for restart in range(6):
        if time.time() > deadline:
            break
        if call_count[0] >= 480:
            break
        
        # Generate diverse starting points
        if restart == 0:
            x_start = x.copy()
        else:
            angle = restart * np.pi / 3
            radius = 0.5 * (np.linalg.norm(x) + 1.0) * (restart + 1)
            x_start = x + radius * np.array([np.cos(angle), np.sin(angle)])
        
        # Initial F evaluation
        Fx = _safe_F(F, x_start, call_count)
        if Fx is None:
            continue
        
        norm_F = np.linalg.norm(Fx)
        if norm_F < best_norm:
            best_norm = norm_F
            best_x = x_start.copy()
        
        if norm_F <= 1e-8:
            return x_start
        
        # Compute finite-difference Jacobian (4 calls: 2 forward + reuse)
        h = 1e-6
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_p = _safe_F(F, x_start + e, call_count)
            if Fx_p is None:
                break
            J[:, j] = (Fx_p - Fx) / h
        
        if Fx_p is None:
            continue
        
        # Newton iterations
        x_curr = x_start.copy()
        Fx_curr = Fx
        J_curr = J.copy()
        
        for _ in range(30):
            if time.time() > deadline:
                break
            if call_count[0] >= 480:
                break
            
            norm_F = np.linalg.norm(Fx_curr)
            if norm_F < best_norm:
                best_norm = norm_F
                best_x = x_curr.copy()
            
            if norm_F <= 1e-10:
                return x_curr
            
            # Broyden step
            result = _broyden_step(x_curr, Fx_curr, J_curr, F, call_count)
            if result[0] is None:
                break
            
            x_curr, Fx_curr, J_curr = result
            
            norm_F = np.linalg.norm(Fx_curr)
            if norm_F < best_norm:
                best_norm = norm_F
                best_x = x_curr.copy()
        
        # If Newton/Broyden didn't converge, try Nelder-Mead
        if best_norm > 1e-8 and time.time() < deadline and call_count[0] < 450:
            nm_result = _nelder_mead(x_curr, F, call_count, max_iter=100)
            Fx_nm = _safe_F(F, nm_result, call_count)
            if Fx_nm is not None:
                norm_nm = np.linalg.norm(Fx_nm)
                if norm_nm < best_norm:
                    best_norm = norm_nm
                    best_x = nm_result.copy()
    
    return best_x
# EVOLVE-BLOCK-END
