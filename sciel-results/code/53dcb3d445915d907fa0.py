import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 135

    def softplus(x, scale=1.0):
        x_scaled = x / scale
        return scale * np.log1p(np.exp(-np.abs(x_scaled))) + np.maximum(0, x)

    def _elapsed():
        return time.time() - start_time

    def _is_valid(result, tol=1e-10):
        n = N
        xy = result[:2*n].reshape(n, 2)
        r = result[2*n:]
        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        return True

    def _sum_r(result):
        return np.sum(result[2*N:])

    def _make_valid(params):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        for _ in range(200):
            if _is_valid(np.concatenate([xy.flatten(), r])):
                break
            r *= 0.97
            xy[:, 0] = np.clip(xy[:, 0], r, 1 - r)
            xy[:, 1] = np.clip(xy[:, 1], r, 1 - r)
        return np.concatenate([xy.flatten(), r])

    def _hex_init(rng):
        n = N
        side = 6
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        for i in range(n):
            row = i // side
            col = i % side
            x = (col + 0.5) / side + (0.5 / side if row % 2 == 1 else 0)
            y = (row + 0.5) / side
            xy[i] = [x, y]
            r[i] = 0.5 / side - 0.002
        xy += rng.uniform(-0.015, 0.015, size=(n, 2))
        r += rng.uniform(-0.005, 0.005, size=n)
        r = np.clip(r, 0.02, 0.2)
        return np.concatenate([xy.flatten(), r])

    def _rect_init(rng):
        n = N
        side = int(np.ceil(np.sqrt(N)))
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        for i in range(n):
            row = i // side
            col = i % side
            xy[i] = [(col + 0.5) / side, (row + 0.5) / side]
            r[i] = 0.5 / side - 0.002
        xy += rng.uniform(-0.02, 0.02, size=(n, 2))
        r += rng.uniform(-0.005, 0.005, size=n)
        r = np.clip(r, 0.02, 0.2)
        return np.concatenate([xy.flatten(), r])

    def _random_init(rng):
        n = N
        xy = rng.uniform(0.08, 0.92, size=(n, 2))
        r = rng.uniform(0.05, 0.1, size=n)
        return np.concatenate([xy.flatten(), r])

    def _push_apart(params, n_iters=100):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        for _ in range(n_iters):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[i] + r[j]
                    if dist < min_dist and dist > 1e-12:
                        overlap = min_dist - dist
                        wi = r[i] / (r[i] + r[j]) if r[i] + r[j] > 1e-15 else 0.5
                        wj = 1 - wi
                        push_x = (dx / dist) * overlap
                        push_y = (dy / dist) * overlap
                        xy[i, 0] -= push_x * wi
                        xy[i, 1] -= push_y * wi
                        xy[j, 0] += push_x * wj
                        xy[j, 1] += push_y * wj
                        moved = True
                    elif dist <= 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        xy[i, 0] -= 0.01 * np.cos(angle)
                        xy[i, 1] -= 0.01 * np.sin(angle)
                        xy[j, 0] += 0.01 * np.cos(angle)
                        xy[j, 1] += 0.01 * np.sin(angle)
                        moved = True
            for i in range(n):
                if xy[i, 0] < r[i]:
                    xy[i, 0] = r[i]
                    moved = True
                if xy[i, 0] > 1 - r[i]:
                    xy[i, 0] = 1 - r[i]
                    moved = True
                if xy[i, 1] < r[i]:
                    xy[i, 1] = r[i]
                    moved = True
                if xy[i, 1] > 1 - r[i]:
                    xy[i, 1] = 1 - r[i]
                    moved = True
            if not moved:
                break
        return np.concatenate([xy.flatten(), r])

    def _objective(params, weight_overlap=100.0, weight_boundary=100.0):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        boundary_viol = np.sum(softplus(-d_left, scale=0.004)**2) + \
                        np.sum(softplus(-d_right, scale=0.004)**2) + \
                        np.sum(softplus(-d_bottom, scale=0.004)**2) + \
                        np.sum(softplus(-d_top, scale=0.004)**2)
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        contact_margin = min_dist[iu] - dist[iu]
        overlap_viol = np.sum(softplus(contact_margin, scale=0.003)**2)
        return -np.sum(r) + weight_boundary * boundary_viol + weight_overlap * overlap_viol

    def _objective_tight(params):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        boundary_viol = np.sum(softplus(-d_left, scale=0.002)**2) + \
                        np.sum(softplus(-d_right, scale=0.002)**2) + \
                        np.sum(softplus(-d_bottom, scale=0.002)**2) + \
                        np.sum(softplus(-d_top, scale=0.002)**2)
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        contact_margin = min_dist[iu] - dist[iu]
        overlap_viol = np.sum(softplus(contact_margin, scale=0.0015)**2)
        # Log-sum-exp for smooth max
        temp = 0.003
        top_k = 12
        sorted_margins = np.sort(contact_margin)[::-1]
        top_margins = sorted_margins[:top_k]
        lse = temp * np.log(np.sum(np.exp(top_margins / temp)))
        lse_penalty = softplus(lse, scale=0.002)**2
        return -np.sum(r) + 150 * boundary_viol + 150 * overlap_viol + 20 * lse_penalty

    def _optimize(params, max_iter=600, tol=1e-14, use_tight=False):
        try:
            obj = _objective_tight if use_tight else _objective
            result = minimize(obj, params, method='L-BFGS-B',
                              options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-13})
            return result.x
        except Exception:
            return params

    def _tight_polish(params, max_iter=1000):
        """High-precision polish using tight objective and epigraph."""
        n = N
        # First pass with tight objective
        polished = _optimize(params, max_iter=max_iter, tol=1e-15, use_tight=True)
        if not _is_valid(polished):
            polished = _make_valid(polished)
        
        # Epigraph formulation for better conditioning
        xy = polished[:2*n].reshape(n, 2)
        r = polished[2*n:]
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        t_init = np.max(min_dist[iu] - dist[iu]) + 0.005
        t_init = max(t_init, 0.0)
        
        def epigraph_obj(full_params):
            xy_e = full_params[:2*n].reshape(n, 2)
            r_e = full_params[2*n:3*n]
            t_e = full_params[3*n]
            d_left = xy_e[:, 0] - r_e
            d_right = 1 - xy_e[:, 0] - r_e
            d_bottom = xy_e[:, 1] - r_e
            d_top = 1 - xy_e[:, 1] - r_e
            boundary_viol = np.sum(softplus(-d_left, scale=0.002)**2) + \
                            np.sum(softplus(-d_right, scale=0.002)**2) + \
                            np.sum(softplus(-d_bottom, scale=0.002)**2) + \
                            np.sum(softplus(-d_top, scale=0.002)**2)
            dx_e = xy_e[:, 0][:, None] - xy_e[:, 0][None, :]
            dy_e = xy_e[:, 1][:, None] - xy_e[:, 1][None, :]
            dist_e = np.sqrt(dx_e**2 + dy_e**2 + 1e-12)
            min_dist_e = r_e[:, None] + r_e[None, :]
            iu_e = np.triu_indices(n, k=1)
            contact_margin_e = min_dist_e[iu_e] - dist_e[iu_e]
            temp_e = 0.002
            lse_e = temp_e * np.log(np.sum(np.exp(contact_margin_e / temp_e)))
            epigraph_viol = softplus(lse_e - t_e, scale=0.001)**2
            t_penalty = softplus(t_e, scale=0.005)**2 * 0.1
            active_contact = softplus(contact_margin_e, scale=0.001)
            overlap_viol = np.sum(active_contact**2)
            return -np.sum(r_e) + 150 * boundary_viol + 150 * epigraph_viol + \
                   10 * t_penalty + 80 * overlap_viol
        
        full_params = np.concatenate([polished, [t_init]])
        try:
            result = minimize(epigraph_obj, full_params, method='L-BFGS-B',
                              options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-14})
            epigraph_result = result.x[:2*n+n]
            if _is_valid(epigraph_result) and _sum_r(epigraph_result) > _sum_r(polished):
                polished = epigraph_result
        except Exception:
            pass
        
        # Final high-precision polish
        final = _optimize(polished, max_iter=500, tol=1e-16, use_tight=True)
        if _is_valid(final) and _sum_r(final) > _sum_r(polished):
            polished = final
        
        return polished

    def _grow_radii(params, max_attempts=50):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        for attempt in range(max_attempts):
            dr = 0.0008 / (1 + attempt * 0.4)
            r_new = r + dr
            feasible = True
            for i in range(n):
                if r_new[i] < 1e-6 or r_new[i] > 0.5:
                    feasible = False
                    break
                if xy[i, 0] - r_new[i] < -1e-10 or 1 - xy[i, 0] - r_new[i] < -1e-10:
                    feasible = False
                    break
                if xy[i, 1] - r_new[i] < -1e-10 or 1 - xy[i, 1] - r_new[i] < -1e-10:
                    feasible = False
                    break
            if not feasible:
                break
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < r_new[i] + r_new[j] - 1e-10:
                        feasible = False
                        break
                if not feasible:
                    break
            if not feasible:
                break
            r = r_new
        return np.concatenate([xy.flatten(), r])

    def _delaunay_contact_graph(params):
        """Extract contact graph from Delaunay triangulation."""
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        try:
            tri = Delaunay(xy)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        edges.add((simplex[i], simplex[j]))
            return list(edges)
        except Exception:
            # Fallback: all pairs within some distance
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j] + 0.05:
                        edges.append((i, j))
            return edges

    def _newton_tangency_solve(params, max_iter=100):
        """
        Newton-Raphson solver on contact graph equations.
        For each edge (i,j) in the contact graph, enforce:
            dist(i,j) = r[i] + r[j]
        For each circle touching a boundary, enforce boundary contact.
        This is a system of nonlinear equations in (x_i, y_i, r_i).
        """
        n = N
        edges = _delaunay_contact_graph(params)
        if len(edges) == 0:
            return params
        
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        
        # Identify boundary contacts
        boundary_contacts = []  # (i, side) where side: 0=left,1=right,2=bottom,3=top
        for i in range(n):
            if xy[i, 0] - r[i] < 0.01:
                boundary_contacts.append((i, 0))
            if 1 - xy[i, 0] - r[i] < 0.01:
                boundary_contacts.append((i, 1))
            if xy[i, 1] - r[i] < 0.01:
                boundary_contacts.append((i, 2))
            if 1 - xy[i, 1] - r[i] < 0.01:
                boundary_contacts.append((i, 3))
        
        # Build residual function: for each edge, dist - (r_i + r_j) = 0
        # For each boundary contact, appropriate equation = 0
        # We use a damped Newton approach
        
        for iteration in range(max_iter):
            if _elapsed() > deadline - 5:
                break
            
            # Compute residuals
            residuals = []
            jac_rows = []  # List of (variable_index, derivative) for each residual
            
            for (i, j) in edges:
                dx = xy[j, 0] - xy[i, 0]
                dy = xy[j, 1] - xy[i, 1]
                dist = np.sqrt(dx*dx + dy*dy) + 1e-15
                res = dist - (r[i] + r[j])
                residuals.append(res)
                # d(res)/d(xy[i,0]) = -dx/dist
                jac_rows.append((2*i, -dx/dist))
                jac_rows.append((2*i+1, -dy/dist))
                jac_rows.append((2*j, dx/dist))
                jac_rows.append((2*j+1, dy/dist))
                jac_rows.append((2*n+i, -1.0))
                jac_rows.append((2*n+j, -1.0))
            
            for (i, side) in boundary_contacts:
                if side == 0:  # left
                    res = xy[i, 0] - r[i]
                    jac_rows.append((2*i, 1.0))
                    jac_rows.append((2*n+i, -1.0))
                elif side == 1:  # right
                    res = 1 - xy[i, 0] - r[i]
                    jac_rows.append((2*i, -1.0))
                    jac_rows.append((2*n+i, -1.0))
                elif side == 2:  # bottom
                    res = xy[i, 1] - r[i]
                    jac_rows.append((2*i+1, 1.0))
                    jac_rows.append((2*n+i, -1.0))
                else:  # top
                    res = 1 - xy[i, 1] - r[i]
                    jac_rows.append((2*i+1, -1.0))
                    jac_rows.append((2*n+i, -1.0))
                residuals.append(res)
            
            if len(residuals) == 0:
                break
            
            # Build Jacobian matrix (sparse, but we'll use dense for simplicity)
            n_vars = 3 * n
            J = np.zeros((len(residuals), n_vars))
            for row_idx, row_data in enumerate(jac_rows):
                for (var_idx, deriv) in row_data:
                    if var_idx < n_vars:
                        J[row_idx, var_idx] += deriv
            
            # Solve J * delta = -residuals
            try:
                delta = np.linalg.lstsq(J, -np.array(residuals), rcond=None)[0]
            except Exception:
                break
            
            # Damped update
            step_size = 1.0
            for _ in range(10):
                xy_new = xy.copy()
                r_new = r.copy()
                xy_new[:, 0] += step_size * delta[:n]
                xy_new[:, 1] += step_size * delta[n:2*n]
                r_new += step_size * delta[2*n:]
                
                # Check feasibility
                if np.all(r_new > 1e-6) and np.all(r_new < 0.5) and \
                   np.all(xy_new[:, 0] >= r_new) and np.all(xy_new[:, 0] <= 1 - r_new) and \
                   np.all(xy_new[:, 1] >= r_new) and np.all(xy_new[:, 1] <= 1 - r_new):
                    # Check no overlaps
                    overlap = False
                    for (i, j) in edges:
                        dx = xy_new[j, 0] - xy_new[i, 0]
                        dy = xy_new[j, 1] - xy_new[i, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r_new[i] + r_new[j] - 1e-8:
                            overlap = True
                            break
                    if not overlap:
                        xy = xy_new
                        r = r_new
                        break
                step_size *= 0.5
            
            # Check convergence
            if np.max(np.abs(delta)) < 1e-14:
                break
        
        return np.concatenate([xy.flatten(), r])

    rng = np.random.RandomState(12345)
    
    best_result = None
    best_score = -np.inf
    
    def _update_best(params):
        nonlocal best_result, best_score
        if _is_valid(params):
            score = _sum_r(params)
            if score > best_score:
                best_score = score
                best_result = params.copy()
    
    # Phase 1: Diverse initializations and optimization
    n_restarts = 80
    init_funcs = [_hex_init, _rect_init, _random_init]
    
    for restart in range(n_restarts):
        if _elapsed() > deadline - 20:
            break
        
        init_func = init_funcs[restart % 3]
        params = init_func(rng)
        params = _push_apart(params, n_iters=60)
        _update_best(params)
        
        # Standard optimization
        refined = _optimize(params, max_iter=500, tol=1e-13, use_tight=False)
        if not _is_valid(refined):
            refined = _make_valid(refined)
        _update_best(refined)
        
        # Tight polish
        if _elapsed() < deadline - 15:
            polished = _tight_polish(refined, max_iter=600)
            _update_best(polished)
        
        # Try Newton tangency solve on the best so far
        if _elapsed() < deadline - 10 and best_result is not None:
            newton_result = _newton_tangency_solve(best_result, max_iter=50)
            if _is_valid(newton_result):
                _update_best(newton_result)
                # Polish after Newton
                if _elapsed() < deadline - 5:
                    polished_newton = _tight_polish(newton_result, max_iter=300)
                    _update_best(polished_newton)
    
    # Phase 2: High-precision final polish on the best result
    if best_result is not None and _elapsed() < deadline - 10:
        # Multi-stage polish
        stage1 = _tight_polish(best_result, max_iter=1000)
        _update_best(stage1)
        
        if _elapsed() < deadline - 8:
            # Newton solve
            newton_final = _newton_tangency_solve(best_result, max_iter=80)
            if _is_valid(newton_final):
                _update_best(newton_final)
                if _elapsed() < deadline - 5:
                    polished_final = _tight_polish(newton_final, max_iter=500)
                    _update_best(polished_final)
        
        # Grow radii
        if _elapsed() < deadline - 5:
            grown = _grow_radii(best_result, max_attempts=60)
            _update_best(grown)
        
        # Final tight polish
        if _elapsed() < deadline - 3:
            final_polish = _tight_polish(best_result, max_iter=800)
            _update_best(final_polish)
    
    # Local perturbation search
    if best_result is not None and _elapsed() < deadline - 10:
        for polish_iter in range(15):
            if _elapsed() > deadline - 3:
                break
            perturbed = best_result.copy()
            xy_p = perturbed[:2*N].reshape(N, 2).copy()
            r_p = perturbed[2*N:].copy()
            pert_scale = 0.001 * (1.0 - polish_iter / 20.0)
            pert_scale = max(pert_scale, 0.0002)
            xy_p += rng.uniform(-pert_scale, pert_scale, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            perturbed = _push_apart(perturbed, n_iters=30)
            
            if polish_iter % 2 == 0:
                polished2 = _optimize(perturbed, max_iter=400, tol=1e-15, use_tight=True)
            else:
                polished2 = _tight_polish(perturbed, max_iter=400)
            
            if _is_valid(polished2):
                grown2 = _grow_radii(polished2, max_attempts=20)
                _update_best(grown2)
                _update_best(polished2)
    
    if best_result is not None:
        n = N
        xy = best_result[:2*n].reshape(n, 2).copy()
        r = best_result[2*n:].copy()
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1 - r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1 - r)
        final_params = np.concatenate([xy.flatten(), r])
        if _is_valid(final_params):
            grown_final = _grow_radii(final_params, max_attempts=30)
            if _is_valid(grown_final) and _sum_r(grown_final) > _sum_r(final_params):
                xy_f = grown_final[:2*n].reshape(n, 2).copy()
                r_f = grown_final[2*n:].copy()
                return np.column_stack([xy_f, r_f])
            return np.column_stack([xy, r])
        for scale in [0.95, 0.9, 0.85, 0.8]:
            r2 = r * scale
            xy2 = xy.copy()
            xy2[:, 0] = np.clip(xy2[:, 0], r2, 1 - r2)
            xy2[:, 1] = np.clip(xy2[:, 1], r2, 1 - r2)
            fp2 = np.concatenate([xy2.flatten(), r2])
            if _is_valid(fp2):
                return np.column_stack([xy2, r2])
    
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
