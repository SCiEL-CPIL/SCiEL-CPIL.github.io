import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay, Voronoi
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s out of 150s, safety margin

    def physics_optimize(params, max_iter=300, alpha=100.0, beta=50.0):
        """
        Vectorized physics-based optimization using soft repulsive potential.
        """
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        
        # Precompute pair indices
        pair_i, pair_j = np.triu_indices(n, k=1)
        
        for it in range(max_iter):
            # Compute forces (vectorized)
            dx = xy[pair_j, 0] - xy[pair_i, 0]
            dy = xy[pair_j, 1] - xy[pair_i, 1]
            d2 = dx*dx + dy*dy
            d = np.sqrt(np.maximum(d2, 1e-12))
            min_d = r[pair_i] + r[pair_j]
            
            # Force magnitudes
            overlap = min_d - d
            # Strong repulsion when overlapping, mild when close
            force_mag = np.where(overlap > 0,
                                 alpha * overlap / (d + 1e-8),
                                 alpha * 0.1 / (np.maximum(overlap, 0) + 0.01)**2)
            
            # Only apply for nearby pairs
            mask = d < min_d * 1.5
            force_mag = np.where(mask, force_mag, 0.0)
            
            fx = np.zeros(n)
            fy = np.zeros(n)
            fx[pair_i] -= force_mag * dx / d
            fy[pair_i] -= force_mag * dy / d
            fx[pair_j] += force_mag * dx / d
            fy[pair_j] += force_mag * dy / d
            
            # Boundary forces (vectorized)
            dist_left = xy[:, 0] - r
            dist_right = 1 - xy[:, 0] - r
            dist_bottom = xy[:, 1] - r
            dist_top = 1 - xy[:, 1] - r
            
            fx += beta * np.maximum(0, 0.05 - dist_left)
            fx -= beta * np.maximum(0, 0.05 - dist_right)
            fy += beta * np.maximum(0, 0.05 - dist_bottom)
            fy -= beta * np.maximum(0, 0.05 - dist_top)
            
            # Mild attraction toward center
            fx += 0.1 * (0.5 - xy[:, 0])
            fy += 0.1 * (0.5 - xy[:, 1])
            
            # Damped update
            lr = 0.01 / (1 + it * 0.01)
            xy[:, 0] += lr * fx
            xy[:, 1] += lr * fy
            
            # Clamp to valid range
            xy[:, 0] = np.clip(xy[:, 0], r, 1 - r)
            xy[:, 1] = np.clip(xy[:, 1], r, 1 - r)
            
            # Check convergence
            if np.max(np.abs(fx)) < 1e-8 and np.max(np.abs(fy)) < 1e-8:
                break
        
        return np.concatenate([xy.flatten(), r])

    best_result = None
    best_score = -np.inf

    def penalty_function(params):
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

        return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol

    def smooth_penalty_function(params):
        """
        Smooth surrogate using log-sum-exp for the penalty terms.
        This avoids the kink at zero violation and may help the optimizer
        find better local minima.
        """
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]

        # Boundary violations (smooth)
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r

        # Use soft penalty: -log(1 + exp(-k * d)) approximates max(0, -d) smoothly
        # But for minimization, we want to penalize violations.
        # Let's use a smooth approximation of max(0, -d)^2
        # Actually, let's stick with the quadratic but use a smooth version of the max
        # For boundary: penalty = sum(softplus(-d)^2) where softplus(x) = log(1+exp(x))
        # But that's expensive. Let's try a different approach.
        
        # Simple smooth quadratic: for each violation v = max(0, -d), use v^2
        # The kink is in the max. We can smooth it with:
        # smooth_max(0, x) = 0.5 * (x + sqrt(x^2 + eps))
        # Then square it.
        
        eps = 1e-8
        
        def smooth_max_square(a, b):
            """Smooth approximation of max(0, a-b)^2"""
            diff = a - b
            sm = 0.5 * (diff + np.sqrt(diff**2 + eps))
            return np.sum(sm**2)

        boundary_viol = smooth_max_square(r, xy[:, 0]) + \
                        smooth_max_square(r, 1 - xy[:, 0]) + \
                        smooth_max_square(r, xy[:, 1]) + \
                        smooth_max_square(r, 1 - xy[:, 1])

        # Overlap penalty (smooth)
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        
        # Smooth max(0, min_dist - dist)^2
        diff = min_dist[iu] - dist[iu]
        sm = 0.5 * (diff + np.sqrt(diff**2 + eps))
        overlap_viol = np.sum(sm**2)

        return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol

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
            obj = smooth_penalty_function if use_smooth else penalty_function
            result = minimize(obj, params, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12})
            return result.x
        except Exception:
            return params

    def physics_refine(params, max_iter=200):
        """
        Combine physics-based optimization with L-BFGS-B refinement.
        """
        # First, use physics to find a good configuration
        phys_result = physics_optimize(params, max_iter=max_iter, alpha=100.0, beta=50.0)
        
        if is_valid(phys_result):
            # Then polish with L-BFGS-B
            polished = refine_solution(phys_result, max_iter=300, tol=1e-13, use_smooth=True)
            if is_valid(polished):
                return polished
            else:
                return phys_result
        else:
            # Try to repair
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
            # Grid packing
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
        elif restart % 10 == 1:
            # Perturbed grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
            xy += rng.uniform(-0.02, 0.02, size=(n, 2))
        elif restart % 10 == 2:
            # Hexagonal packing
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
            # Random uniform
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.05, 0.1, size=n)
        elif restart % 10 == 4:
            # 8x4 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 8
                col = i % 8
                xy[i] = [(col + 0.5)/8, (row + 0.5)/4]
                r[i] = 0.5/8 - 0.001
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        elif restart % 10 == 5:
            # 4x8 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 4
                col = i % 4
                xy[i] = [(col + 0.5)/4, (row + 0.5)/8]
                r[i] = 0.5/4 - 0.001
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        elif restart % 10 == 6:
            # Hexagonal variant
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
            # Clustered: 4 groups of 8
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
            # Random medium
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.06, 0.09, size=n)
        else:
            # Random small
            xy = rng.uniform(0.05, 0.95, size=(n, 2))
            r = rng.uniform(0.03, 0.06, size=n)

        params = np.concatenate([xy.flatten(), r])
        return params

    def voronoi_based_packing(rng):
        """
        Use Delaunay/Voronoi approach: compute inradii from Delaunay triangulation,
        then optimize centers to maximize sum of inradii.
        """
        n = N
        # Start with a perturbed grid
        side = 6
        xy = np.zeros((n, 2))
        for i in range(n):
            row = i // side
            col = i % side
            xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
        
        # Add randomness
        xy += rng.uniform(-0.02, 0.02, size=(n, 2))
        xy = np.clip(xy, 0.05, 0.95)
        
        # Compute inradii from Delaunay triangulation
        def compute_inradii(centers):
            """Compute inradius for each site from Delaunay triangulation."""
            try:
                tri = Delaunay(centers)
                inradii = np.zeros(len(centers))
                for simplex in tri.simplices:
                    # Compute inradius of triangle
                    pts = centers[simplex]
                    a = np.linalg.norm(pts[1] - pts[2])
                    b = np.linalg.norm(pts[0] - pts[2])
                    c = np.linalg.norm(pts[0] - pts[1])
                    s = (a + b + c) / 2
                    if s > 1e-12:
                        area = np.sqrt(s * (s-a) * (s-b) * (s-c))
                        if area > 1e-12:
                            r_in = area / s
                            for idx in simplex:
                                inradii[idx] = min(inradii[idx], r_in) if inradii[idx] > 0 else r_in
                # For sites not in any triangle (boundary), use distance to nearest
                mask = inradii == 0
                if np.any(mask):
                    for i in np.where(mask)[0]:
                        dists = np.sqrt(np.sum((centers - centers[i])**2, axis=1))
                        dists[i] = np.inf
                        inradii[i] = np.min(dists) / 2
                return inradii
            except Exception:
                # Fallback: distance to nearest neighbor / 2
                dists = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            dists[i, j] = np.linalg.norm(centers[i] - centers[j])
                dists[np.arange(n), np.arange(n)] = np.inf
                return np.min(dists, axis=1) / 2
        
        r = compute_inradii(xy) * 0.95  # Slightly smaller to ensure separation
        r = np.clip(r, 0.02, 0.15)
        
        params = np.concatenate([xy.flatten(), r])
        params = push_apart(params, n_iters=60)
        
        # Refine with smooth penalty
        refined = refine_solution(params, max_iter=500, tol=1e-14, use_smooth=True)
        
        if is_valid(refined):
            return refined
        else:
            repaired = push_apart(refined, n_iters=100)
            if is_valid(repaired):
                return repaired
            else:
                return params if is_valid(params) else None

    rng = np.random.RandomState(12345)

    # Phase 1: Many restarts with diverse initializations
    n_restarts = 100
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break

        # Use Voronoi-based approach for more restarts
        if restart % 3 == 0:
            params = voronoi_based_packing(rng)
            if params is None:
                continue
        else:
            params = get_initial_config(restart, rng)
            params = push_apart(params, n_iters=40)

        if is_valid(params):
            score = get_score(params)
            if score > best_score:
                best_score = score
                best_result = params.copy()

        # Refine with L-BFGS-B, alternating between sharp and smooth penalty
        use_smooth = (restart % 2 == 0)
        refined = refine_solution(params, max_iter=400, tol=1e-14, use_smooth=use_smooth)
        
        # Also try physics-based refinement for some restarts
        if restart % 2 == 0 and time.time() < deadline - 20:
            phys_refined = physics_refine(params, max_iter=200)
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
            # Try to repair
            repaired = push_apart(refined, n_iters=80)
            if is_valid(repaired):
                score = get_score(repaired)
                if score > best_score:
                    best_score = score
                    best_result = repaired.copy()
            else:
                # Shrink radii until valid
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

    # Phase 2: High-precision polish on best solution
    if best_result is not None and time.time() < deadline - 5:
        # Polish with sharp penalty
        polished = refine_solution(best_result, max_iter=1000, tol=1e-15, use_smooth=False)
        if is_valid(polished):
            score = get_score(polished)
            if score > best_score:
                best_score = score
                best_result = polished.copy()

        # Polish with smooth penalty
        polished_smooth = refine_solution(best_result, max_iter=1000, tol=1e-15, use_smooth=True)
        if is_valid(polished_smooth):
            score = get_score(polished_smooth)
            if score > best_score:
                best_score = score
                best_result = polished_smooth.copy()

        # Try physics-based polish
        if time.time() < deadline - 10:
            phys_polished = physics_refine(best_result, max_iter=300)
            if is_valid(phys_polished):
                phys_score = get_score(phys_polished)
                if phys_score > best_score:
                    best_score = phys_score
                    best_result = phys_polished.copy()

        # Try perturbations and re-refine
        for polish_iter in range(15):
            if time.time() > deadline - 3:
                break
            perturbed = best_result.copy()
            xy_p = perturbed[:2*N].reshape(N, 2).copy()
            r_p = perturbed[2*N:].copy()
            xy_p += rng.uniform(-0.003, 0.003, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            perturbed = push_apart(perturbed, n_iters=40)
            
            # Alternate between L-BFGS-B and physics-based refinement
            if polish_iter % 2 == 0:
                polished2 = refine_solution(perturbed, max_iter=500, tol=1e-15, use_smooth=False)
            else:
                polished2 = physics_refine(perturbed, max_iter=200)
            
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

    # Fallback: grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
