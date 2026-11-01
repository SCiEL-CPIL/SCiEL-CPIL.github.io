import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    def voronoi_radii(xy):
        """Given centers xy (N,2), compute max radii via Voronoi cells."""
        n = N
        r = np.zeros(n)
        
        # Add boundary points far away to clip Voronoi to [0,1]^2
        # We'll use a different approach: for each circle, r_i = min(dist to all other circles - r_j, dist to boundary)
        # But that's circular. Instead, use Voronoi.
        
        # Create Voronoi with extra points to handle boundaries
        # Add points just outside the square to force clipping
        extra_pts = np.array([
            [-0.1, -0.1], [1.1, -0.1], [-0.1, 1.1], [1.1, 1.1],
            [0.5, -0.1], [0.5, 1.1], [-0.1, 0.5], [1.1, 0.5],
            [0, 0], [1, 0], [0, 1], [1, 1],
            [0.5, 0], [0.5, 1], [0, 0.5], [1, 0.5]
        ])
        all_pts = np.vstack([xy, extra_pts])
        
        try:
            vor = Voronoi(all_pts)
        except Exception:
            # Fallback: use simple min distance
            for i in range(n):
                min_dist = 1.0
                for j in range(n):
                    if i != j:
                        d = np.sqrt((xy[i,0]-xy[j,0])**2 + (xy[i,1]-xy[j,1])**2)
                        min_dist = min(min_dist, d/2.0)
                min_dist = min(min_dist, xy[i,0], 1-xy[i,0], xy[i,1], 1-xy[i,1])
                r[i] = max(0, min_dist - 1e-6)
            return r
        
        for i in range(n):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region or len(region) == 0:
                # Unbounded region, shouldn't happen with our extra points
                r[i] = 0.01
                continue
            
            vertices = vor.vertices[region]
            
            # Radius is min distance from center to any edge of the Voronoi cell
            # Edges are between consecutive vertices
            min_r = np.inf
            for k in range(len(vertices)):
                v1 = vertices[k]
                v2 = vertices[(k+1) % len(vertices)]
                # Distance from point to line segment
                dx = v2[0] - v1[0]
                dy = v2[1] - v1[1]
                if dx == 0 and dy == 0:
                    continue
                t = ((xy[i,0]-v1[0])*dx + (xy[i,1]-v1[1])*dy) / (dx*dx + dy*dy)
                t = np.clip(t, 0, 1)
                px = v1[0] + t*dx
                py = v1[1] + t*dy
                dist = np.sqrt((xy[i,0]-px)**2 + (xy[i,1]-py)**2)
                min_r = min(min_r, dist)
            
            # Also clamp to boundary
            min_r = min(min_r, xy[i,0], 1-xy[i,0], xy[i,1], 1-xy[i,1])
            r[i] = max(0, min_r - 1e-7)
        
        return r

    def objective(params):
        """Negative sum of radii (we maximize sum, so minimize negative)."""
        xy = params.reshape(N, 2)
        r = voronoi_radii(xy)
        return -np.sum(r)

    def objective_with_grad(params):
        """Objective with finite difference gradient."""
        f0 = objective(params)
        return f0

    def is_valid(xy, r, tol=1e-8):
        n = N
        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False
        # Check overlaps
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        return True

    def get_score(xy, r):
        return np.sum(r)

    def refine_voronoi(xy, max_iter=100):
        """Refine centers using L-BFGS-B on Voronoi-derived radii."""
        try:
            result = minimize(objective, xy.flatten(), method='L-BFGS-B',
                            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-7})
            return result.x.reshape(N, 2)
        except Exception:
            return xy

    def get_initial_configs(rng):
        """Generate diverse initial configurations."""
        configs = []
        
        # 1. Regular grid
        side = int(np.ceil(np.sqrt(N)))
        xy = np.zeros((N, 2))
        for i in range(N):
            row = i // side
            col = i % side
            xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
        configs.append(xy.copy())
        
        # 2. Perturbed grid
        xy2 = xy.copy()
        xy2 += rng.uniform(-0.03, 0.03, size=(N, 2))
        configs.append(xy2)
        
        # 3. Hexagonal packing (approximate)
        xy3 = np.zeros((N, 2))
        rows = 6
        cols = 6
        for i in range(N):
            row = i // cols
            col = i % cols
            x = (col + 0.5)/cols + (0.5/cols if row % 2 == 1 else 0)
            y = (row + 0.5)/rows
            xy3[i] = [min(0.95, max(0.05, x)), min(0.95, max(0.05, y))]
        configs.append(xy3)
        
        # 4. Random
        xy4 = rng.uniform(0.1, 0.9, size=(N, 2))
        configs.append(xy4)
        
        # 5. Random with clustering
        centers = rng.uniform(0.2, 0.8, size=(4, 2))
        xy5 = np.zeros((N, 2))
        for i in range(N):
            c = centers[i % 4]
            xy5[i] = c + rng.normal(0, 0.08, size=2)
            xy5[i] = np.clip(xy5[i], 0.05, 0.95)
        configs.append(xy5)
        
        # 6. Another random
        xy6 = rng.uniform(0.08, 0.92, size=(N, 2))
        configs.append(xy6)
        
        # 7. Grid with different aspect
        xy7 = np.zeros((N, 2))
        for i in range(N):
            row = i // 8
            col = i % 8
            xy7[i] = [(col + 0.5)/8, (row + 0.5)/4.0]
        configs.append(xy7)
        
        # 8. Concentric rings
        xy8 = np.zeros((N, 2))
        ring1 = 4
        ring2 = 8
        ring3 = 12
        ring4 = 8
        idx = 0
        for angle in np.linspace(0, 2*np.pi, ring1, endpoint=False):
            xy8[idx] = [0.5 + 0.15*np.cos(angle), 0.5 + 0.15*np.sin(angle)]
            idx += 1
        for angle in np.linspace(0, 2*np.pi, ring2, endpoint=False):
            xy8[idx] = [0.5 + 0.3*np.cos(angle), 0.5 + 0.3*np.sin(angle)]
            idx += 1
        for angle in np.linspace(0, 2*np.pi, ring3, endpoint=False):
            xy8[idx] = [0.5 + 0.42*np.cos(angle), 0.5 + 0.42*np.sin(angle)]
            idx += 1
        for angle in np.linspace(0, 2*np.pi, ring4, endpoint=False):
            xy8[idx] = [0.5 + 0.48*np.cos(angle), 0.5 + 0.48*np.sin(angle)]
            idx += 1
        configs.append(xy8)
        
        return configs

    rng = np.random.RandomState(123)
    best_xy = None
    best_r = None
    best_score = -np.inf
    
    initial_configs = get_initial_configs(rng)
    
    for restart, xy_init in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
        
        # First refinement
        xy_refined = refine_voronoi(xy_init, max_iter=80)
        r = voronoi_radii(xy_refined)
        
        if is_valid(xy_refined, r):
            score = get_score(xy_refined, r)
            if score > best_score:
                best_score = score
                best_xy = xy_refined.copy()
                best_r = r.copy()
        
        # Second refinement with more iterations
        if time.time() < deadline - 15:
            xy_refined2 = refine_voronoi(xy_refined, max_iter=120)
            r2 = voronoi_radii(xy_refined2)
            if is_valid(xy_refined2, r2):
                score2 = get_score(xy_refined2, r2)
                if score2 > best_score:
                    best_score = score2
                    best_xy = xy_refined2.copy()
                    best_r = r2.copy()
        
        # Third refinement
        if time.time() < deadline - 20:
            xy_refined3 = refine_voronoi(xy_refined2, max_iter=100)
            r3 = voronoi_radii(xy_refined3)
            if is_valid(xy_refined3, r3):
                score3 = get_score(xy_refined3, r3)
                if score3 > best_score:
                    best_score = score3
                    best_xy = xy_refined3.copy()
                    best_r = r3.copy()
    
    # Final polish on best solution
    if best_xy is not None and time.time() < deadline - 5:
        xy_final = refine_voronoi(best_xy, max_iter=150)
        r_final = voronoi_radii(xy_final)
        if is_valid(xy_final, r_final):
            score_final = get_score(xy_final, r_final)
            if score_final > best_score:
                best_score = score_final
                best_xy = xy_final.copy()
                best_r = r_final.copy()
    
    if best_xy is not None:
        # Final safety clamp
        r_safe = np.clip(best_r, 1e-6, 0.49)
        xy_safe = best_xy.copy()
        for i in range(N):
            xy_safe[i, 0] = np.clip(xy_safe[i, 0], r_safe[i], 1 - r_safe[i])
            xy_safe[i, 1] = np.clip(xy_safe[i, 1], r_safe[i], 1 - r_safe[i])
        
        if is_valid(xy_safe, r_safe, tol=1e-7):
            return np.column_stack([xy_safe, r_safe])
        else:
            # Return unclamped if clamping broke validity
            return np.column_stack([best_xy, best_r])
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    result = np.zeros((N, 3))
    for i in range(N):
        row = i // side
        col = i % side
        result[i] = [(col + 0.5)/side, (row + 0.5)/side, 0.5/side - 0.002]
    return result

