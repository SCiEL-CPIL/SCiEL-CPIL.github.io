"""Root-finder for 3 fixed 2D systems using a multi-phase hybrid approach.
Phase 1: Nelder-Mead simplex on g(x)=0.5*||F(x)||^2 for global robustness.
Phase 2: Broyden's method (good variant) for fast local quadratic convergence.
Phase 3: FD-Newton polish for final precision to 1e-8.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 with ||F(x)|| <= 1e-8 (Euclidean norm).
    
    Multi-phase strategy:
    - Phase 1: Nelder-Mead simplex method on g(x)=0.5*||F(x)||^2
    - Phase 2: Broyden's method with damped steps
    - Phase 3: FD-Newton polish for high accuracy
    
    Budget: <500 F-calls per system. Time-limited to ~130s.
    """
    x = np.array(x0, dtype=float, copy=True)
    t_start = time.time()
    deadline = t_start + 130.0
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _time_ok():
        return time.time() < deadline

    def _call_F(x_val):
        nonlocal call_count
        if not _time_ok() or call_count >= 492:
            return None
        call_count += 1
        try:
            result = F(np.asarray(x_val, dtype=float))
        except Exception:
            return None
        result = np.asarray(result, dtype=float)
        if result.shape != (2,):
            try:
                result = result.reshape(2)
            except Exception:
                return None
        if not np.all(np.isfinite(result)):
            return None
        return result

    def _g(x_val):
        """Objective: 0.5 * ||F(x)||^2"""
        Fx = _call_F(x_val)
        if Fx is None:
            return None
        return 0.5 * np.dot(Fx, Fx)

    def _fd_jacobian(x_val, Fx_val, h=1e-5):
        """Compute Jacobian via forward differences. Uses 2 calls (Fx already known)."""
        nonlocal call_count
        J = np.zeros((2, 2))
        for j in range(2):
            if not _time_ok() or call_count >= 490:
                break
            e = np.zeros(2)
            e[j] = h
            Fx_plus = _call_F(x_val + e)
            if Fx_plus is not None:
                J[:, j] = (Fx_plus - Fx_val) / h
            else:
                J[:, j] = 1.0
        return J

    def _nelder_mead(x0_nm, max_iter=200):
        """
        Nelder-Mead simplex method on g(x) = 0.5*||F(x)||^2.
        Returns (x_best, g_best, F_best).
        """
        nonlocal call_count
        # Initialize simplex
        x0_nm = np.array(x0_nm, dtype=float)
        simplex = [x0_nm.copy()]
        g_vals = [_g(x0_nm)]
        if g_vals[0] is None:
            return x0_nm, np.inf, None
        simplex_g = [g_vals[0]]
        
        # Expand simplex
        step = 0.1 * max(1.0, np.linalg.norm(x0_nm))
        for j in range(2):
            x_new = x0_nm.copy()
            x_new[j] += step
            g_new = _g(x_new)
            if g_new is None:
                g_new = 1e10
            simplex.append(x_new)
            simplex_g.append(g_new)
        
        best_x_nm = simplex[0].copy()
        best_g_nm = simplex_g[0]
        best_F_nm = _call_F(best_x_nm)
        
        for it in range(max_iter):
            if not _time_ok() or call_count >= 480:
                break
            if best_g_nm <= 1e-16:
                break
            
            # Sort simplex by g value
            order = np.argsort(simplex_g)
            simplex = [simplex[i] for i in order]
            simplex_g = [simplex_g[i] for i in order]
            
            # Check convergence
            if simplex_g[-1] - simplex_g[0] < 1e-14:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            x_reflect = centroid + alpha * (centroid - simplex[-1])
            g_reflect = _g(x_reflect)
            if g_reflect is None:
                g_reflect = 1e10
            
            if g_reflect < simplex_g[0]:
                # Expansion
                gamma = 2.0
                x_expand = centroid + gamma * (centroid - simplex[-1])
                g_expand = _g(x_expand)
                if g_expand is None:
                    g_expand = g_reflect
                if g_expand < g_reflect:
                    simplex[-1] = x_expand
                    simplex_g[-1] = g_expand
                else:
                    simplex[-1] = x_reflect
                    simplex_g[-1] = g_reflect
            elif g_reflect < simplex_g[-2]:
                simplex[-1] = x_reflect
                simplex_g[-1] = g_reflect
            else:
                # Contraction
                if g_reflect < simplex_g[-1]:
                    # Outside contraction
                    rho = 0.5
                    x_contract = centroid + rho * (x_reflect - centroid)
                else:
                    # Inside contraction
                    rho = 0.5
                    x_contract = centroid - rho * (centroid - simplex[-1])
                g_contract = _g(x_contract)
                if g_contract is None:
                    g_contract = 1e10
                if g_contract < simplex_g[-1]:
                    simplex[-1] = x_contract
                    simplex_g[-1] = g_contract
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        g_shrink = _g(simplex[i])
                        if g_shrink is not None:
                            simplex_g[i] = g_shrink
            
            # Update best
            for i in range(len(simplex)):
                if simplex_g[i] < best_g_nm:
                    best_g_nm = simplex_g[i]
                    best_x_nm = simplex[i].copy()
                    best_F_nm = _call_F(best_x_nm)
        
        return best_x_nm, best_g_nm, best_F_nm

    def _broyden(x0_br, Fx0, max_iter=150):
        """
        Broyden's method (good variant) with damped steps.
        Returns (x_best, F_best, norm_best).
        """
        nonlocal call_count
        x = np.array(x0_br, dtype=float)
        Fx = np.array(Fx0, dtype=float) if Fx0 is not None else _call_F(x)
        if Fx is None:
            return x, None, np.inf
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x, Fx, norm_Fx
        
        best_x_br = x.copy()
        best_F_br = Fx.copy()
        best_norm_br = norm_Fx
        
        # Compute initial Jacobian
        J = _fd_jacobian(x, Fx)
        try:
            det_J = np.linalg.det(J)
            if abs(det_J) < 1e-10:
                J += np.eye(2) * 1e-6
        except np.linalg.LinAlgError:
            J = np.eye(2)
        
        stalled_count = 0
        prev_norm = norm_Fx
        
        for it in range(max_iter):
            if not _time_ok() or call_count >= 485:
                break
            
            # Solve J @ dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.pinv(J) @ (-Fx)
                except Exception:
                    break
            
            norm_dx = np.linalg.norm(dx)
            if norm_dx == 0:
                break
            
            # Damped step
            alpha = 1.0
            success = False
            for _ in range(10):
                if not _time_ok() or call_count >= 488:
                    break
                x_new = x + alpha * dx
                if not np.all(np.isfinite(x_new)):
                    alpha *= 0.5
                    continue
                Fx_new = _call_F(x_new)
                if Fx_new is None:
                    break
                norm_new = np.linalg.norm(Fx_new)
                if norm_new < norm_Fx * (1 + 1e-4):
                    success = True
                    break
                alpha *= 0.5
                if alpha < 1e-12:
                    break
            
            if not success:
                break
            
            # Update Broyden Jacobian (good variant)
            s = x_new - x
            y = Fx_new - Fx
            s_norm_sq = np.dot(s, s)
            if s_norm_sq > 1e-18:
                J = J + np.outer(y - J @ s, s) / s_norm_sq
            
            x = x_new
            Fx = Fx_new
            norm_Fx = norm_new
            
            if norm_Fx <= 1e-8:
                return x, Fx, norm_Fx
            
            if norm_Fx < best_norm_br:
                best_x_br = x.copy()
                best_F_br = Fx.copy()
                best_norm_br = norm_Fx
            
            # Detect stalling
            if norm_Fx > prev_norm * 0.999:
                stalled_count += 1
            else:
                stalled_count = 0
            prev_norm = norm_Fx
            
            if stalled_count >= 10:
                # Restart with fresh Jacobian
                if _time_ok() and call_count < 480:
                    J = _fd_jacobian(x, Fx)
                    try:
                        if abs(np.linalg.det(J)) < 1e-10:
                            J += np.eye(2) * 1e-6
                    except np.linalg.LinAlgError:
                        J = np.eye(2)
                    stalled_count = 0
                    prev_norm = np.inf
        
        return best_x_br, best_F_br, best_norm_br

    def _polish(x0_pol, Fx0, max_iter=15):
        """
        FD-Newton polish for high precision.
        Returns (x_best, F_best, norm_best).
        """
        nonlocal call_count
        x = np.array(x0_pol, dtype=float)
        Fx = np.array(Fx0, dtype=float) if Fx0 is not None else _call_F(x)
        if Fx is None:
            return x, None, np.inf
        norm_Fx = np.linalg.norm(Fx)
        
        best_x_pol = x.copy()
        best_F_pol = Fx.copy()
        best_norm_pol = norm_Fx
        
        for it in range(max_iter):
            if not _time_ok() or call_count >= 490:
                break
            if norm_Fx <= 1e-8:
                break
            
            # Small FD Jacobian
            h = max(1e-7, 1e-5 * np.sqrt(norm_Fx))
            J = _fd_jacobian(x, Fx, h)
            try:
                if abs(np.linalg.det(J)) < 1e-12:
                    J += np.eye(2) * 1e-8
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            norm_dx = np.linalg.norm(dx)
            if norm_dx < 1e-14:
                break
            
            # Damped step
            alpha = 1.0
            for _ in range(6):
                if not _time_ok() or call_count >= 490:
                    break
                x_new = x + alpha * dx
                if not np.all(np.isfinite(x_new)):
                    alpha *= 0.5
                    continue
                Fx_new = _call_F(x_new)
                if Fx_new is None:
                    break
                norm_new = np.linalg.norm(Fx_new)
                if norm_new < norm_Fx:
                    x = x_new
                    Fx = Fx_new
                    norm_Fx = norm_new
                    if norm_Fx < best_norm_pol:
                        best_x_pol = x.copy()
                        best_F_pol = Fx.copy()
                        best_norm_pol = norm_Fx
                    break
                alpha *= 0.5
        
        return best_x_pol, best_F_pol, best_norm_pol

    # Initial evaluation
    Fx = _call_F(x)
    if Fx is None:
        return best_x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # Phase 1: Nelder-Mead for global search
    if norm_Fx > 1e-3 and _time_ok() and call_count < 400:
        x_nm, g_nm, Fx_nm = _nelder_mead(x, max_iter=150)
        if Fx_nm is not None:
            norm_nm = np.linalg.norm(Fx_nm)
            if norm_nm < best_norm:
                best_x = x_nm.copy()
                best_norm = norm_nm
                x = x_nm
                Fx = Fx_nm
                norm_Fx = norm_nm
                if norm_Fx <= 1e-8:
                    return x

    # Phase 2: Broyden's method for local convergence
    if norm_Fx > 1e-8 and _time_ok() and call_count < 450:
        x_br, Fx_br, norm_br = _broyden(x, Fx, max_iter=100)
        if Fx_br is not None and norm_br < best_norm:
            best_x = x_br.copy()
            best_norm = norm_br
            x = x_br
            Fx = Fx_br
            norm_Fx = norm_br
            if norm_Fx <= 1e-8:
                return x

    # Phase 3: FD-Newton polish for final precision
    if best_norm < 1e-2 and _time_ok() and call_count < 485:
        x_pol, Fx_pol, norm_pol = _polish(best_x, Fx, max_iter=10)
        if Fx_pol is not None and norm_pol < best_norm:
            best_x = x_pol.copy()
            best_norm = norm_pol
            if norm_pol <= 1e-8:
                return best_x

    # Final verification
    if call_count < 495 and _time_ok():
        Fx_final = _call_F(best_x)
        if Fx_final is not None:
            nf = np.linalg.norm(Fx_final)
            if nf <= 1e-8:
                return best_x
            if nf < best_norm:
                best_x = best_x.copy()
                best_norm = nf

    return best_x
# EVOLVE-BLOCK-END
