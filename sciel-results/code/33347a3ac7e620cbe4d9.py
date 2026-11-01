import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 135

    def softplus(x, scale=1.0):
        """Numerically stable softplus for smooth max(0, x)."""
        x_scaled = x / scale
        return scale * np.log1p(np.exp(-np.abs(x_scaled))) + np.maximum(0, x)

    def physics_optimize(params, max_iter=200, alpha=100.0, beta=50.0):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        pair_i, pair_j = np.triu_indices(n, k=1)
        
        for it in range(max_iter):
            fx = np.zeros(n)
            fy = np.zeros(n)
            
            # Vectorized pair forces with smooth contact detection
            dx = xy[pair_j, 0] - xy[pair_i, 0]
            dy = xy[pair_j, 1] - xy[pair_i, 1]
            d2 = dx*dx + dy*dy
            d = np.sqrt(np.maximum(d2, 1e-12))
            min_d = r[pair_i] + r[pair_j]
            
            # Soft contact: smooth transition around contact distance
            contact_margin = min_d - d  # positive when overlapping
            # Use softplus for smooth force activation
            soft_contact = softplus(contact_margin, scale=0.01)
            
            # Force magnitude: strong repulsion when overlapping, smooth decay
            force_mag = alpha * soft_contact / (d + 0.01)
            # Add weak attraction for near-contact pairs to encourage tight packing
            near_contact = softplus(-contact_margin, scale=0.02) * 0.1
            force_mag -= near_contact * alpha * 0.05
            force_mag = np.clip(force_mag, -alpha*10, alpha*10)
            
            fx[pair_i] -= force_mag * dx / d
            fy[pair_i] -= force_mag * dy / d
            fx[pair_j] += force_mag * dx / d
            fy[pair_j] += force_mag * dy / d
            
            # Vectorized boundary forces with smooth margins
            dist_left = xy[:, 0] - r
            dist_right = 1 - xy[:, 0] - r
            dist_bottom = xy[:, 1] - r
            dist_top = 1 - xy[:, 1] - r
            
            # Smooth boundary repulsion
            margin = 0.05
            fx += beta * softplus(margin - dist_left, scale=0.01)
            fx -= beta * softplus(margin - dist_right, scale=0.01)
            fy += beta * softplus(margin - dist_bottom, scale=0.01)
            fy -= beta * softplus(margin - dist_top, scale=0.01)
            
            # Centering force
            fx += 0.1 * (0.5 - xy[:, 0])
            fy += 0.1 * (0.5 - xy[:, 1])
            
            lr = 0.01 / (1 + it * 0.01)
            xy[:, 0] += lr * fx
            xy[:, 1] += lr * fy
            
            xy[:, 0] = np.clip(xy[:, 0], r, 1 - r)
            xy[:, 1] = np.clip(xy[:, 1], r, 1 - r)
            
            if np.max(np.abs(fx)) < 1e-8 and np.max(np.abs(fy)) < 1e-8:
                break
        
        return np.concatenate([xy.flatten(), r])

    best_result = None
    best_score = -np.inf

    def penalty_function(params):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]

        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        boundary_viol = np.sum(np.maximum(0, -d_left)**2) + np.sum(np.maximum(0, -d_right)**2) + \
                        np.sum(np.maximum(0, -d_bottom)**2) + np.sum(np.maximum(0, -d_top)**2)

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        overlap = np.maximum(0, min_dist[iu] - dist[iu])
        overlap_viol = np.sum(overlap**2)

        return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol

    def smooth_penalty_function(params):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]

        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r

        # Softplus-based smooth max for better gradient flow
        def smooth_max_square(val, scale=0.005):
            # val > 0 means violation
            sm = softplus(val, scale=scale)
            return np.sum(sm**2)

        boundary_viol = smooth_max_square(r - xy[:, 0]) + \
                        smooth_max_square(r - (1 - xy[:, 0])) + \
                        smooth_max_square(r - xy[:, 1]) + \
                        smooth_max_square(r - (1 - xy[:, 1]))

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        
        diff = min_dist[iu] - dist[iu]
        overlap_viol = smooth_max_square(diff, scale=0.01)

        return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol

    def soft_contact_objective(params):
        """
        Objective using soft contact graph: only penalize pairs that are
        close to being in contact, reducing constraint dimensionality.
        Uses epigraph formulation with auxiliary smoothness.
        """
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]

        # Boundary constraints with softplus
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        boundary_viol = np.sum(softplus(-d_left, scale=0.01)**2) + \
                        np.sum(softplus(-d_right, scale=0.01)**2) + \
                        np.sum(softplus(-d_bottom, scale=0.01)**2) + \
                        np.sum(softplus(-d_top, scale=0.01)**2)

        # Soft contact graph: only activate constraints for near-contact pairs
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        
        # Contact margin: positive when overlapping
        contact_margin = min_dist[iu] - dist[iu]
        
        # Soft activation: only strongly penalize when actually overlapping,
        # with smooth transition
        active_contact = softplus(contact_margin, scale=0.005)
        overlap_viol = np.sum(active_contact**2)
        
        # Encourage tight packing: small penalty for pairs that are close but not overlapping
        # This creates a "desired contact" effect
        near_but_not_contact = softplus(-contact_margin, scale=0.02) * 0.1
        tightness_bonus = np.sum(near_but_not_contact) * 0.01

        return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol - tightness_bonus

    def is_valid(result, tol=1e-9):
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

    def get_score(result):
        n = N
        r = result[2*n:]
        return np.sum(r)

    def push_apart(params, n_iters=100):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()

        for _ in range(n_iters):
            moved = False

            for i in range(n):
                for j in range(i+1, n):
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
                        angle = np.random.uniform(0, 2*np.pi)
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

    def refine_solution(params, max_iter=500, tol=1e-14, use_smooth=False):
        try:
            if use_smooth:
                obj = smooth_penalty_function
            else:
                # Use soft contact objective for better gradient flow
                obj = soft_contact_objective
            result = minimize(obj, params, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12})
            return result.x
        except Exception:
            return params

    def physics_refine(params, max_iter=200):
        phys_result = physics_optimize(params, max_iter=max_iter, alpha=100.0, beta=50.0)
        
        if is_valid(phys_result):
            # Try soft contact objective first for better local optimization
            polished_soft = refine_solution(phys_result, max_iter=300, tol=1e-13, use_smooth=False)
            if is_valid(polished_soft):
                # Then polish with smooth penalty
                polished = refine_solution(polished_soft, max_iter=200, tol=1e-14, use_smooth=True)
                if is_valid(polished):
                    return polished
                return polished_soft
            else:
                polished = refine_solution(phys_result, max_iter=300, tol=1e-13, use_smooth=True)
                if is_valid(polished):
                    return polished
                return phys_result
        else:
            repaired = push_apart(phys_result, n_iters=80)
            if is_valid(repaired):
                polished = refine_solution(repaired, max_iter=200, tol=1e-13, use_smooth=True)
                if is_valid(polished):
                    return polished
            return params

    def get_initial_config(restart, rng):
        n = N
        side = int(np.ceil(np.sqrt(N)))

        if restart % 10 == 0:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
        elif restart % 10 == 1:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
            xy += rng.uniform(-0.02, 0.02, size=(n, 2))
        elif restart % 10 == 2:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.25 if row % 2 == 1 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.08
        elif restart % 10 == 3:
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.05, 0.1, size=n)
        elif restart % 10 == 4:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 8
                col = i % 8
                xy[i] = [(col + 0.5)/8, (row + 0.5)/4]
                r[i] = 0.5/8 - 0.001
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        elif restart % 10 == 5:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 4
                col = i % 4
                xy[i] = [(col + 0.5)/4, (row + 0.5)/8]
                r[i] = 0.5/4 - 0.001
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        elif restart % 10 == 6:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.2 if row % 2 == 0 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.075
        elif restart % 10 == 7:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            for i in range(n):
                group = i // 8
                cx, cy = centers[group]
                sub_i = i % 8
                sx = (sub_i % 3) / 3.0 + 0.5/3
                sy = (sub_i // 3) / 3.0 + 0.5/3
                xy[i] = [cx + (sx - 0.5)*0.4, cy + (sy - 0.5)*0.4]
                r[i] = 0.06
        elif restart % 10 == 8:
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.06, 0.09, size=n)
        else:
            xy = rng.uniform(0.05, 0.95, size=(n, 2))
            r = rng.uniform(0.03, 0.06, size=n)

        params = np.concatenate([xy.flatten(), r])
        return params

    def voronoi_based_packing(rng):
        n = N
        side = 6
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        
        for i in range(n):
            row = i // side
            col = i % side
            xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
            r[i] = 0.08
        
        xy += rng.uniform(-0.03, 0.03, size=(n, 2))
        r += rng.uniform(-0.01, 0.01, size=n)
        r = np.clip(r, 0.03, 0.15)
        
        params = np.concatenate([xy.flatten(), r])
        params = push_apart(params, n_iters=80)
        
        refined = refine_solution(params, max_iter=400, tol=1e-14, use_smooth=True)
        
        if is_valid(refined):
            return refined
        else:
            repaired = push_apart(refined, n_iters=100)
            if is_valid(repaired):
                return repaired
            else:
                return params if is_valid(params) else None

    rng = np.random.RandomState(12345)

    n_restarts = 80
    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        if restart % 5 == 0:
            params = voronoi_based_packing(rng)
            if params is None:
                continue
        else:
            params = get_initial_config(restart, rng)
            params = push_apart(params, n_iters=50)

        if is_valid(params):
            score = get_score(params)
            if score > best_score:
                best_score = score
                best_result = params.copy()

        use_smooth = (restart % 2 == 0)
        refined = refine_solution(params, max_iter=300, tol=1e-13, use_smooth=use_smooth)
        
        if restart % 3 == 0 and time.time() < deadline:
            phys_refined = physics_refine(params, max_iter=150)
            if is_valid(phys_refined):
                phys_score = get_score(phys_refined)
                if phys_score > best_score:
                    best_score = phys_score
                    best_result = phys_refined.copy()

        if is_valid(refined):
            score = get_score(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        else:
            repaired = push_apart(refined, n_iters=80)
            if is_valid(repaired):
                score = get_score(repaired)
                if score > best_score:
                    best_score = score
                    best_result = repaired.copy()
            else:
                xy = refined[:2*N].reshape(N, 2).copy()
                r = refined[2*N:].copy()
                for _ in range(200):
                    if is_valid(np.concatenate([xy.flatten(), r])):
                        break
                    r *= 0.95
                final_params = np.concatenate([xy.flatten(), r])
                if is_valid(final_params):
                    score = get_score(final_params)
                    if score > best_score:
                        best_score = score
                        best_result = final_params.copy()

    if best_result is not None and time.time() < deadline - 5:
        # Multi-stage high-precision polish with soft contact graph
        # Stage 1: Soft contact objective for better gradient flow
        polished = refine_solution(best_result, max_iter=600, tol=1e-15, use_smooth=False)
        if is_valid(polished):
            score = get_score(polished)
            if score > best_score:
                best_score = score
                best_result = polished.copy()

        # Stage 2: Smooth penalty for final tightening
        polished_smooth = refine_solution(best_result, max_iter=600, tol=1e-15, use_smooth=True)
        if is_valid(polished_smooth):
            score = get_score(polished_smooth)
            if score > best_score:
                best_score = score
                best_result = polished_smooth.copy()

        if time.time() < deadline - 10:
            # Physics-based refinement with soft contacts
            phys_polished = physics_refine(best_result, max_iter=250)
            if is_valid(phys_polished):
                phys_score = get_score(phys_polished)
                if phys_score > best_score:
                    best_score = phys_score
                    best_result = phys_polished.copy()
                # Follow up with L-BFGS-B on the physics result
                final_polish = refine_solution(phys_polished, max_iter=400, tol=1e-16, use_smooth=False)
                if is_valid(final_polish):
                    fp_score = get_score(final_polish)
                    if fp_score > best_score:
                        best_score = fp_score
                        best_result = final_polish.copy()
                # Final smooth polish
                final_smooth = refine_solution(best_result, max_iter=300, tol=1e-16, use_smooth=True)
                if is_valid(final_smooth):
                    fs_score = get_score(final_smooth)
                    if fs_score > best_score:
                        best_score = fs_score
                        best_result = final_smooth.copy()

        # Local perturbation search with tighter perturbations
        for polish_iter in range(12):
            if time.time() > deadline - 3:
                break
            perturbed = best_result.copy()
            xy_p = perturbed[:2*N].reshape(N, 2).copy()
            r_p = perturbed[2*N:].copy()
            # Smaller perturbations for later iterations
            pert_scale = 0.002 * (1.0 - polish_iter / 15.0)
            pert_scale = max(pert_scale, 0.0003)
            xy_p += rng.uniform(-pert_scale, pert_scale, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            perturbed = push_apart(perturbed, n_iters=40)
            
            # Alternate between soft contact and smooth objectives
            if polish_iter % 2 == 0:
                polished2 = refine_solution(perturbed, max_iter=400, tol=1e-16, use_smooth=False)
            else:
                polished2 = physics_refine(perturbed, max_iter=150)
            
            if is_valid(polished2):
                score = get_score(polished2)
                if score > best_score:
                    best_score = score
                    best_result = polished2.copy()

    if best_result is not None:
        n = N
        xy = best_result[:2*n].reshape(n, 2).copy()
        r = best_result[2*n:].copy()
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            return np.column_stack([xy, r])
        # Fallback: try to repair with smaller radii
        for scale in [0.95, 0.9, 0.85, 0.8]:
            r2 = r * scale
            xy2 = xy.copy()
            xy2[:, 0] = np.clip(xy2[:, 0], r2, 1-r2)
            xy2[:, 1] = np.clip(xy2[:, 1], r2, 1-r2)
            fp2 = np.concatenate([xy2.flatten(), r2])
            if is_valid(fp2):
                return np.column_stack([xy2, r2])

    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
