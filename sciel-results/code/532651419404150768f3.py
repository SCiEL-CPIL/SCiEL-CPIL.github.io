import math
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def compute_min_enclosing_hex_side(centers, angles, otheta=0.0, ocenter=(0.0, 0.0)):
        """
        Compute the minimum side length of a regular hexagon (centered at ocenter, rotated by otheta)
        containing all inner hexagons defined by centers and angles.
        """
        otheta_rad = math.radians(otheta)
        max_abs_proj = 0.0
        # Precompute vertices for all inner hexagons relative to ocenter
        all_vertices = []
        for i in range(len(centers)):
            cx, cy = centers[i]
            angle_rad = math.radians(angles[i])
            for vi in range(6):
                vert_angle = angle_rad + vi * math.pi / 3
                vx = cx + math.cos(vert_angle) - ocenter[0]
                vy = cy + math.sin(vert_angle) - ocenter[1]
                all_vertices.append((vx, vy))
        
        all_vertices = np.array(all_vertices)
        
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            projections = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
            # The hexagon is defined by |proj| <= s * sqrt(3) / 2
            # So we need max |proj|
            max_proj = np.max(np.abs(projections))
            if max_proj > max_abs_proj:
                max_abs_proj = max_proj
        
        return 2.0 * max_abs_proj / math.sqrt(3)

    def generate_lattice_clusters():
        """
        Generate unique 12-hexagon clusters from the hexagonal lattice.
        Returns a list of lists of (x, y) coordinates.
        """
        clusters = []
        # Basis vectors for hexagonal lattice of side 1 hexagons
        # Distance between centers of touching hexagons is sqrt(3)
        d = math.sqrt(3)
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        
        # Core: 7 hexagons (0,0) + 6 neighbors
        core = [np.array([0.0, 0.0])]
        ring1 = [u, v, v - u, -u, -v, u - v]
        for p in ring1:
            core.append(p.copy())
            
        # Ring 2 candidates: 12 hexagons
        # These are at distance 2*sqrt(3) from center in 6 directions, and intermediate positions
        # Actually, the 12 neighbors of the ring 1 hexagons that are not in core or ring 1.
        ring2_candidates = []
        basis = [u, v, v - u, -u, -v, u - v]
        seen = set()
        for i in range(6):
            p1 = ring1[i]
            for b in basis:
                p2 = p1 + b
                # Check if p2 is in core or ring1
                is_core = np.allclose(p2, np.array([0.0, 0.0]))
                is_ring1 = False
                for p1j in ring1:
                    if np.allclose(p2, p1j):
                        is_ring1 = True
                        break
                if not is_core and not is_ring1:
                    # Normalize for set check
                    key = (round(p2[0], 5), round(p2[1], 5))
                    if key not in seen:
                        seen.add(key)
                        ring2_candidates.append(p2)
                        
        # We have 12 candidates in ring2_candidates.
        # We need to choose 5 of them to add to the core (7) to make 12.
        # Total combinations: C(12, 5) = 792.
        # We will evaluate all of them to find the one with the smallest enclosing hexagon.
        
        for i in range(12):
            for j in range(i + 1, 12):
                for k in range(j + 1, 12):
                    for l in range(k + 1, 12):
                        for m in range(l + 1, 12):
                            cluster = core.copy()
                            cluster.append(ring2_candidates[i])
                            cluster.append(ring2_candidates[j])
                            cluster.append(ring2_candidates[k])
                            cluster.append(ring2_candidates[l])
                            cluster.append(ring2_candidates[m])
                            clusters.append(cluster)
                            
        return clusters

    def objective(params, cluster):
        """
        Objective function for a specific cluster topology.
        params: [ox, oy, otheta]
        cluster: list of (x, y) relative to cluster center (0,0).
        Inner hexagons are assumed to have angle 0 (or 30, doesn't matter for symmetry if outer is optimized).
        """
        ox, oy, otheta = params
        # Centers are fixed relative to each other.
        # We can translate the whole cluster.
        # But the objective function computes enclosing hexagon centered at (ox, oy).
        # So we can just set inner centers to cluster points, and optimize outer center/angle.
        # Wait, if we fix inner centers at cluster points, we can't move the cluster relative to the outer hexagon center independently?
        # Yes, we can. The outer hexagon center (ox, oy) is a variable.
        # The inner hexagons are at fixed positions (cluster points).
        # This effectively allows the cluster to "move" relative to the outer hexagon by changing (ox, oy).
        
        centers = [(p[0], p[1]) for p in cluster]
        angles = [0.0] * 12
        
        return compute_min_enclosing_hex_side(centers, angles, otheta, (ox, oy))

    best_side = float('inf')
    best_params = None
    best_cluster = None

    clusters = generate_lattice_clusters()
    
    # To save time, we can prune clusters that are "long".
    # Calculate bounding box size for each cluster quickly.
    pruned_clusters = []
    for c in clusters:
        xs = [p[0] for p in c]
        ys = [p[1] for p in c]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        # Approximate side length
        approx_side = max(width, height) / math.sqrt(3) * 1.1 # heuristic
        if approx_side < 5.5: # Threshold
            pruned_clusters.append(c)
            
    # If pruning removes too many, use all
    if len(pruned_clusters) < 10:
        pruned_clusters = clusters

    for i, cluster in enumerate(pruned_clusters):
        if time.time() > deadline - 10:
            break
            
        # Initial guess for outer hexagon
        # Center of mass of the cluster
        cx = np.mean([p[0] for p in cluster])
        cy = np.mean([p[1] for p in cluster])
        init_params = [cx, cy, 0.0]
        
        try:
            res = minimize(objective, init_params, args=(cluster,), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
                best_cluster = cluster
        except Exception:
            pass

    # If we found a good cluster, we can try to improve it by allowing small deviations from the lattice
    # This is the "continuous refinement" part.
    if best_cluster is not None and time.time() < deadline - 20:
        # Create a variable set for the 12 centers
        # Initial values from the lattice cluster
        init_centers = []
        for p in best_cluster:
            init_centers.extend([p[0], p[1]])
            
        # Add outer hexagon params
        init_full = init_centers + [best_params[0], best_params[1], best_params[2]]
        
        def full_objective(params):
            centers = [(params[i*2], params[i*2+1]) for i in range(12)]
            angles = [0.0] * 12
            ox, oy, otheta = params[24], params[25], params[26]
            
            # Penalty for overlapping
            # Check distance between centers
            # For aligned hexagons, min distance is sqrt(3)
            pen = 0.0
            for i in range(12):
                for j in range(i+1, 12):
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < math.sqrt(3) - 1e-6:
                        pen += (math.sqrt(3) - dist) ** 2
            
            side = compute_min_enclosing_hex_side(centers, angles, otheta, (ox, oy))
            return side + 100.0 * pen

        try:
            res2 = minimize(full_objective, init_full, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            # Check validity
            centers = [(res2.x[i*2], res2.x[i*2+1]) for i in range(12)]
            valid = True
            for i in range(12):
                for j in range(i+1, 12):
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < math.sqrt(3) - 1e-5:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                side = compute_min_enclosing_hex_side(centers, [0.0]*12, res2.x[26], (res2.x[24], res2.x[25]))
                if side < best_side:
                    best_side = side
                    # Update best params to reflect the refined centers
                    best_params = res2.x
                    best_cluster = centers
        except Exception:
            pass

    # Finalize result
    if best_cluster is None:
        # Fallback to a simple grid
        d = math.sqrt(3)
        centers = []
        for i in range(3):
            for j in range(4):
                centers.append([(j-1.5)*d, (i-1)*3.0])
        best_cluster = centers
        best_params = [0,0,0]
        best_side = compute_min_enclosing_hex_side([(p[0],p[1]) for p in best_cluster], [0.0]*12, 0, (0,0))

    # Extract inner hexagons
    if len(best_params) == 3:
        # Rigid cluster case
        inner = [[p[0], p[1], 0.0] for p in best_cluster]
        outer_center = (best_params[0], best_params[1])
        outer_angle = best_params[2]
    else:
        # Refined case
        inner = [[best_params[i*2], best_params[i*2+1], 0.0] for i in range(12)]
        outer_center = (best_params[24], best_params[25])
        outer_angle = best_params[26]

    return inner, outer_center, best_side, outer_angle

