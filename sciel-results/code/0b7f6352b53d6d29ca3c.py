import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    best_result = None
    best_score = -np.inf

    def is_valid(params, tol=1e-9):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        
        # Check radii are positive
        if np.any(r < -tol):
            return False
        
        # Check boundaries
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False

        # Check overlaps
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist_sq = dx**2 + dy**2
        min_dist = r[:, None] + r[None, :]
        
        # Only check upper triangle to avoid double counting and self-overlap
        iu = np.triu_indices(n, k=1)
        # dist < min_dist - tol  =>  dist_sq < (min_dist - tol)^2
        # Using squared distance for speed and precision
        if np.any(dist_sq[iu] < (min_dist[iu] - tol)**2):
            return False

        return True

    def get_score(params):
        return np.sum(params[2*N:])

    def penalty_function(params, penalty_weight=10000):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]

        # Boundary penalty
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        boundary_viol = np.sum(np.maximum(0, -d_left)**2) + np.sum(np.maximum(0, -d_right)**2) + \
                        np.sum(np.maximum(0, -d_bottom)**2) + np.sum(np.maximum(0, -d_top)**2)

        # Overlap penalty
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        overlap = np.maximum(0, min_dist[iu] - dist[iu])
        overlap_viol = np.sum(overlap**2)

        # We want to maximize sum(r), so minimize -sum(r) + penalties
        return -np.sum(r) + penalty_weight * (boundary_viol + overlap_viol)

    def physics_optimize(params, n_steps=200, dt=0.005, repulsion_strength=10.0):
        """
        Physics-based simulation to resolve overlaps and push circles apart.
        Circles repel each other. Boundary constraints are handled by clamping.
        """
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        
        # Initialize velocities
        v = np.zeros((n, 2))
        
        for step in range(n_steps):
            if time.time() > deadline - 10:
                break
                
            forces = np.zeros((n, 2))
            
            # Pairwise repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq)
                    min_dist = r[i] + r[j]
                    
                    if dist < 1e-6:
                        # Randomize direction if overlapping heavily
                        angle = np.random.uniform(0, 2*np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1e-6
                        dist_sq = 1e-12
                    
                    if dist < min_dist:
                        # Force proportional to overlap, stronger if closer
                        # Using a stiff spring force
                        overlap = min_dist - dist
                        force_mag = repulsion_strength * overlap
                        # Normalize direction
                        nx = dx / dist
                        ny = dy / dist
                        
                        forces[i, 0] -= force_mag * nx
                        forces[i, 1] -= force_mag * ny
                        forces[j, 0] += force_mag * nx
                        forces[j, 1] += force_mag * ny
            
            # Boundary forces (push away from walls)
            # We will just clamp positions after integration, which is more robust for hard constraints
            # But adding a soft force helps stability
            for i in range(n):
                if xy[i, 0] < r[i]:
                    forces[i, 0] += repulsion_strength * (r[i] - xy[i, 0])
                if xy[i, 0] > 1 - r[i]:
                    forces[i, 0] -= repulsion_strength * (xy[i, 0] - (1 - r[i]))
                if xy[i, 1] < r[i]:
                    forces[i, 1] += repulsion_strength * (r[i] - xy[i, 1])
                if xy[i, 1] > 1 - r[i]:
                    forces[i, 1] -= repulsion_strength * (xy[i, 1] - (1 - r[i]))
            
            # Integrate
            v += forces * dt
            v *= 0.9 # Damping
            xy += v * dt
            
            # Clamp to boundaries
            for i in range(n):
                xy[i, 0] = np.clip(xy[i, 0], r[i], 1 - r[i])
                xy[i, 1] = np.clip(xy[i, 1], r[i], 1 - r[i])
                # Zero out velocity component if hitting wall
                if xy[i, 0] == r[i] or xy[i, 0] == 1 - r[i]:
                    v[i, 0] = 0
                if xy[i, 1] == r[i] or xy[i, 1] == 1 - r[i]:
                    v[i, 1] = 0

        return np.concatenate([xy.flatten(), r])

    def refine_solution(params, max_iter=300, penalty_weight=10000):
        try:
            # L-BFGS-B handles bounds well, but our penalty function handles them too.
            # We use penalty function to ensure we don't break validity too much during optimization
            # However, for final polish, we can try to optimize sum(r) directly if valid.
            # Here we stick to the penalty method for robustness.
            result = minimize(lambda p: penalty_function(p, penalty_weight), params, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
            return result.x
        except Exception:
            return params

    def grow_radii(params, n_grow=20, growth_rate=0.005):
        """
        Iteratively grow radii and re-optimize.
        """
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        
        for _ in range(n_grow):
            if time.time() > deadline - 15:
                break
                
            r_new = r * (1 + growth_rate)
            # Check if max radius exceeds 0.5 (impossible in unit square for multiple circles usually, but check bounds)
            if np.max(r_new) > 0.5:
                break
                
            params_trial = np.concatenate([xy.flatten(), r_new])
            
            # Optimize positions only
            def pos_penalty(p):
                xy_p = p.reshape(n, 2)
                d_left = xy_p[:, 0] - r_new
                d_right = 1 - xy_p[:, 0] - r_new
                d_bottom = xy_p[:, 1] - r_new
                d_top = 1 - xy_p[:, 1] - r_new
                boundary_viol = np.sum(np.maximum(0, -d_left)**2) + np.sum(np.maximum(0, -d_right)**2) + \
                                np.sum(np.maximum(0, -d_bottom)**2) + np.sum(np.maximum(0, -d_top)**2)
                
                dx = xy_p[:, 0][:, None] - xy_p[:, 0][None, :]
                dy = xy_p[:, 1][:, None] - xy_p[:, 1][None, :]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = r_new[:, None] + r_new[None, :]
                iu = np.triu_indices(n, k=1)
                overlap = np.maximum(0, min_dist[iu] - dist[iu])
                overlap_viol = np.sum(overlap**2)
                
                return boundary_viol + overlap_viol

            # Use L-BFGS-B to find a valid configuration for the new radii
            result = minimize(pos_penalty, xy.flatten(), method='L-BFGS-B',
                            options={'maxiter': 50, 'ftol': 1e-12})
            
            if pos_penalty(result.x) < 1e-8:
                xy = result.x.reshape(n, 2)
                r = r_new
            else:
                # If we can't fit them, try a smaller growth or break
                break

        return np.concatenate([xy.flatten(), r])

    def get_initial_config(restart, rng):
        n = N
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        
        if restart % 5 == 0:
            # Hexagonal packing
            # 6x6 grid approx
            rows = 6
            cols = 6
            dx = 1.0 / cols
            dy = 1.0 / rows
            r_val = 0.4 * dx
            count = 0
            for i in range(rows):
                for j in range(cols):
                    if count >= n: break
                    x = (j + 0.5) * dx
                    y = (i + 0.5) * dy
                    if i % 2 == 1:
                        x += dx / 2
                    # Keep inside
                    x = np.clip(x, r_val, 1 - r_val)
                    y = np.clip(y, r_val, 1 - r_val)
                    xy[count] = [x, y]
                    r[count] = r_val
                    count += 1
        elif restart % 5 == 1:
            # Grid 6x6
            rows = 6
            cols = 6
            dx = 1.0 / cols
            dy = 1.0 / rows
            r_val = 0.45 * dx
            count = 0
            for i in range(rows):
                for j in range(cols):
                    if count >= n: break
                    x = (j + 0.5) * dx
                    y = (i + 0.5) * dy
                    x = np.clip(x, r_val, 1 - r_val)
                    y = np.clip(y, r_val, 1 - r_val)
                    xy[count] = [x, y]
                    r[count] = r_val
                    count += 1
        elif restart % 5 == 2:
            # Random
            xy = rng.uniform(0.1, 0.9, size=(n, 2))
            r = rng.uniform(0.05, 0.08, size=n)
        elif restart % 5 == 3:
            # Clustered random
            centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            for i in range(n):
                c = centers[i % 4]
                xy[i] = [c[0] + rng.uniform(-0.1, 0.1), c[1] + rng.uniform(-0.1, 0.1)]
                r[i] = rng.uniform(0.06, 0.08)
            # Clip
            for i in range(n):
                xy[i, 0] = np.clip(xy[i, 0], r[i], 1-r[i])
                xy[i, 1] = np.clip(xy[i, 1], r[i], 1-r[i])
        else:
            # Mixed
            xy = rng.uniform(0.05, 0.95, size=(n, 2))
            r = rng.uniform(0.04, 0.09, size=n)
            for i in range(n):
                xy[i, 0] = np.clip(xy[i, 0], r[i], 1-r[i])
                xy[i, 1] = np.clip(xy[i, 1], r[i], 1-r[i])
                
        return np.concatenate([xy.flatten(), r])

    rng = np.random.RandomState(42)
    
    # Phase 1: Physics + Refine loop
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        params = get_initial_config(restart, rng)
        
        # 1. Physics to resolve overlaps
        params_phys = physics_optimize(params, n_steps=100)
        
        # 2. Refine with gradient descent
        params_ref = refine_solution(params_phys, max_iter=200)
        
        # 3. Check validity and score
        if is_valid(params_ref):
            score = get_score(params_ref)
            if score > best_score:
                best_score = score
                best_result = params_ref.copy()
        else:
            # Try to repair with stronger physics
            params_repair = physics_optimize(params_ref, n_steps=200)
            if is_valid(params_repair):
                score = get_score(params_repair)
                if score > best_score:
                    best_score = score
                    best_result = params_repair.copy()
            else:
                # Shrink radii to make valid
                xy = params_ref[:2*N].reshape(N, 2).copy()
                r = params_ref[2*N:].copy()
                for _ in range(50):
                    if is_valid(np.concatenate([xy.flatten(), r])):
                        break
                    r *= 0.95
                final_params = np.concatenate([xy.flatten(), r])
                if is_valid(final_params):
                    score = get_score(final_params)
                    if score > best_score:
                        best_score = score
                        best_result = final_params.copy()

        # 4. Grow radii from best result
        if best_result is not None and time.time() < deadline - 10:
            grown = grow_radii(best_result, n_grow=10, growth_rate=0.002)
            if is_valid(grown):
                score = get_score(grown)
                if score > best_score:
                    best_score = score
                    best_result = grown.copy()

    # Phase 2: Final Polish
    if best_result is not None and time.time() < deadline - 5:
        # High precision polish
        polished = refine_solution(best_result, max_iter=500, penalty_weight=50000)
        if is_valid(polished):
            score = get_score(polished)
            if score > best_score:
                best_score = score
                best_result = polished.copy()
        
        # Try local perturbations
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            perturbed = best_result.copy()
            xy_p = perturbed[:2*N].reshape(N, 2).copy()
            r_p = perturbed[2*N:].copy()
            xy_p += rng.uniform(-0.005, 0.005, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            
            # Quick physics to fix any overlaps introduced by perturbation
            perturbed = physics_optimize(perturbed, n_steps=50)
            
            polished2 = refine_solution(perturbed, max_iter=200, penalty_weight=50000)
            if is_valid(polished2):
                score = get_score(polished2)
                if score > best_score:
                    best_score = score
                    best_result = polished2.copy()

    if best_result is not None:
        n = N
        xy = best_result[:2*n].reshape(n, 2).copy()
        r = best_result[2*n:].copy()
        # Final safety clip
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            return np.column_stack([xy, r])

    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
