import math
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    # Precompute constants
    SQRT3 = math.sqrt(3.0)
    APOTHEM = SQRT3 / 2.0  # For unit hexagon
    
    # Helper: Get vertices of a regular hexagon centered at (cx, cy) with side 1 and angle ang
    # Vertices are at angles ang + i*60
    def get_hex_verts(cx, cy, ang):
        # Optimization: Precompute cos/sin for the 6 offsets relative to ang
        # But ang changes, so we compute per call. 
        # For speed, we can vectorize later if needed, but N is small.
        verts = []
        for i in range(6):
            theta = ang + i * math.pi / 3.0
            verts.append((cx + math.cos(theta), cy + math.sin(theta)))
        return verts

    # Helper: Check if two convex polygons (hexagons) are disjoint
    # Returns True if disjoint, False if overlapping
    # Uses Separating Axis Theorem (SAT) which is robust for convex polygons
    # Since they are regular hexagons, we can also use distance between centers,
    # but SAT is more general for rotations.
    # However, for regular hexagons, the distance between centers must be at least 
    # the sum of apothems if aligned, but rotation complicates it.
    # SAT is O(12) checks.
    
    def project_polygon(poly, axis):
        min_val = float('inf')
        max_val = float('-inf')
        for p in poly:
            proj = p[0] * axis[0] + p[1] * axis[1]
            if proj < min_val: min_val = proj
            if proj > max_val: max_val = proj
        return min_val, max_val

    def polygons_disjoint(poly1, poly2):
        # Check all edges of poly1 and poly2
        for i in range(6):
            # Edge vector of poly1
            p1 = poly1[i]
            p2 = poly1[(i+1)%6]
            edge = (p2[0]-p1[0], p2[1]-p1[1])
            # Normal to edge
            ax = -edge[1]
            ay = edge[0]
            # Normalize? Not strictly necessary for overlap check, but helps for stability
            # For regular hexagons, edge lengths are constant, so normals have same magnitude
            min1, max1 = project_polygon(poly1, (ax, ay))
            min2, max2 = project_polygon(poly2, (ax, ay))
            if max1 < min2 or max2 < min1:
                return True # Disjoint
        for i in range(6):
            p1 = poly2[i]
            p2 = poly2[(i+1)%6]
            edge = (p2[0]-p1[0], p2[1]-p1[1])
            ax = -edge[1]
            ay = edge[0]
            min1, max1 = project_polygon(poly1, (ax, ay))
            min2, max2 = project_polygon(poly2, (ax, ay))
            if max1 < min2 or max2 < min1:
                return True # Disjoint
        return False # Overlapping

    # Helper: Compute the required outer side length S for a given configuration
    # and outer hexagon parameters (center, angle)
    # The outer hexagon is centered at (0,0) for simplicity in optimization, 
    # we can fix outer center at (0,0) and outer angle at 0 due to symmetry,
    # or allow them to vary. Fixing them reduces dimensionality.
    # Let's fix outer center at (0,0) and outer angle at 0.
    # The constraint is: For all 6 normals of the outer hex, the max projection
    # of all inner vertices onto that normal must be <= S * apothem_outer.
    # apothem_outer = S * sqrt(3)/2.
    # So S >= max_proj / (sqrt(3)/2).
    
    def compute_required_S(inner_verts_list):
        max_S = 0.0
        # Outer hexagon normals for angle 0 (pointy top? or flat top?)
        # Let's define outer hexagon with vertices at 0, 60, ...
        # Normals are at 30, 90, ...
        for i in range(6):
            theta = math.pi/6 + i * math.pi/3.0
            nx = math.cos(theta)
            ny = math.sin(theta)
            max_proj = 0.0
            for verts in inner_verts_list:
                for v in verts:
                    proj = v[0]*nx + v[1]*ny
                    if proj > max_proj:
                        max_proj = proj
            # S * (sqrt(3)/2) >= max_proj
            S_needed = max_proj / APOTHEM
            if S_needed > max_S:
                max_S = S_needed
        return max_S

    def objective_and_grads(state):
        """
        state: list of [x, y, angle] for N hexagons.
        Returns: (score, grads)
        score is a penalty function to minimize.
        """
        # Compute vertices for all hexagons
        all_verts = []
        for i in range(N):
            x, y, a = state[i]
            verts = get_hex_verts(x, y, a)
            all_verts.append(verts)
        
        # 1. Outer boundary constraint
        # We want to minimize S, but S is determined by the configuration.
        # Instead, we can fix S to a target value and penalize violations,
        # or minimize S directly.
        # Let's minimize S directly, but S is a max of linear functions, which is non-smooth.
        # Alternative: Fix S to a small value and penalize going out?
        # Better: Minimize S. The gradient of S with respect to positions is
        # the normal of the "active" constraint (the vertex/edge determining the max projection).
        
        # Let's find the active constraint for S
        S_val = 0.0
        active_normal = (0,0)
        active_vertex_idx = 0
        active_hex_idx = 0
        active_vert_idx = 0
        
        for i in range(6):
            theta = math.pi/6 + i * math.pi/3.0
            nx = math.cos(theta)
            ny = math.sin(theta)
            max_proj = 0.0
            for h_idx in range(N):
                for v_idx in range(6):
                    vx, vy = all_verts[h_idx][v_idx]
                    proj = vx*nx + vy*ny
                    if proj > max_proj:
                        max_proj = proj
                        S_candidate = max_proj / APOTHEM
                        if S_candidate > S_val:
                            S_val = S_candidate
                            active_normal = (nx, ny)
                            active_hex_idx = h_idx
                            active_vert_idx = v_idx
        
        # Gradient of S_val w.r.t position of active_hex_idx
        # S_val = (vx*nx + vy*ny) / APOTHEM
        # dS/dvx = nx / APOTHEM
        # dS/dvy = ny / APOTHEM
        # dS/dangle: vx = x + cos(a + k*pi/3), vy = y + sin(a + k*pi/3)
        # dS/da = (dx/dx_a * nx + dy/dy_a * ny) / APOTHEM
        # dx/da = -sin(a + k*pi/3), dy/da = cos(a + k*pi/3)
        
        grads = [[0.0, 0.0, 0.0] for _ in range(N)]
        
        # Gradient contribution from outer boundary
        an_x, an_y = active_normal
        # The vertex is at angle a + v_idx * pi/3
        a_active = state[active_hex_idx][2]
        theta_v = a_active + active_vert_idx * math.pi/3.0
        # d(vx)/da = -sin(theta_v), d(vy)/da = cos(theta_v)
        dS_dx = an_x / APOTHEM
        dS_dy = an_y / APOTHEM
        dS_da = (-math.sin(theta_v) * an_x + math.cos(theta_v) * an_y) / APOTHEM
        
        grads[active_hex_idx][0] += dS_dx
        grads[active_hex_idx][1] += dS_dy
        grads[active_hex_idx][2] += dS_da

        # 2. Overlap constraints
        # Penalty: sum of overlaps.
        # For each pair, if overlapping, add a penalty.
        # Gradient: push them apart.
        # This is non-smooth. We use a soft penalty based on distance.
        # Approximate overlap penalty: if distance < min_dist, push.
        # For hexagons, min distance is roughly 2*apothem = sqrt(3) if aligned.
        # With rotation, it's complex.
        # Let's use a simple repulsive force: F = 1/r^2 * (direction) if r < r0.
        # This is not exact for hexagons but drives them apart.
        # To be more accurate, we can use the SAT gap.
        # But SAT gap gradient is hard.
        # Let's stick to center-based repulsion with a strong constant,
        # and rely on the outer boundary to compress them.
        # If they overlap, the repulsion pushes them out.
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, a1 = state[i]
                x2, y2, a2 = state[j]
                dx = x2 - x1
                dy = y2 - y1
                dist_sq = dx*dx + dy*dy
                dist = math.sqrt(dist_sq)
                # Minimum center distance for disjoint hexagons (approx)
                # If aligned, it's sqrt(3). If rotated 30 deg, it's 2.
                # Let's use 1.7 as a soft threshold.
                r0 = 1.7
                if dist < r0:
                    # Penalty: (r0 - dist)^2
                    # Grad: 2*(r0-dist) * (d(dist)/dx)
                    # d(dist)/dx1 = -dx/dist, d(dist)/dx2 = dx/dist
                    # Force on i: -grad_i, Force on j: -grad_j
                    # We want to minimize penalty, so move away.
                    # grad_penalty/dx1 = 2*(r0-dist)*(-dx/dist)
                    # grad_penalty/dx2 = 2*(r0-dist)*(dx/dist)
                    if dist > 1e-6:
                        factor = 2.0 * (r0 - dist) / dist
                        gx1 = -dx * factor
                        gy1 = -dy * factor
                        gx2 = dx * factor
                        gy2 = dy * factor
                        
                        # Weight the overlap penalty heavily
                        w = 10.0
                        grads[i][0] += w * gx1
                        grads[i][1] += w * gy1
                        grads[j][0] += w * gx2
                        grads[j][1] += w * gy2
                        
                        # Angle gradients for overlap are ignored in this simple model
                        # This might lead to suboptimal rotations, but positions will be valid.
                        # We can add a small random angle jitter to escape local minima.
                        
                        overlap_penalty += w * (r0 - dist)**2
                    else:
                        # If very close, push randomly
                        grads[i][0] -= 1.0
                        grads[j][0] += 1.0
                        overlap_penalty += 100.0
        
        total_score = S_val + overlap_penalty
        return total_score, grads, S_val

    def optimize(initial_state, max_iter=2000, lr=0.05):
        state = [list(x) for x in initial_state]
        best_S = float('inf')
        best_state = [list(x) for x in state]
        
        # Compute initial S
        _, _, S_init = objective_and_grads(state)
        if S_init < best_S:
            best_S = S_init
            best_state = [list(x) for x in state]
            
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            score, grads, S_val = objective_and_grads(state)
            
            if S_val < best_S:
                # Check validity: are they disjoint?
                # We check disjointness using SAT
                valid = True
                all_verts = [get_hex_verts(state[i][0], state[i][1], state[i][2]) for i in range(N)]
                for i in range(N):
                    for j in range(i+1, N):
                        if not polygons_disjoint(all_verts[i], all_verts[j]):
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    best_S = S_val
                    best_state = [list(x) for x in state]
            
            # Gradient step
            for i in range(N):
                state[i][0] -= lr * grads[i][0]
                state[i][1] -= lr * grads[i][1]
                state[i][2] -= lr * 0.1 * grads[i][2] # Damp angle gradient
                # Keep angles in [0, 2pi)
                state[i][2] = state[i][2] % (2 * math.pi)
                
            # Add small noise to escape local minima
            if it % 100 == 0:
                for i in range(N):
                    state[i][0] += np.random.normal(0, 0.01)
                    state[i][1] += np.random.normal(0, 0.01)
                    state[i][2] += np.random.normal(0, 0.01)
        
        return best_S, best_state

    # Initial configurations
    configs = []
    
    # 1. Hexagonal lattice
    sp = math.sqrt(3.0)
    hex_cfg = [[0.0, 0.0, 0.0]]
    for i in range(6):
        theta = i * math.pi / 3.0
        hex_cfg.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
    # Add 4 more
    for i in [0, 1, 4, 5]:
        theta = i * math.pi / 3.0 + math.pi/6
        hex_cfg.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
    while len(hex_cfg) < N:
        hex_cfg.append([0.0, 0.0, 0.0])
    configs.append(hex_cfg[:N])
    
    # 2. Linear
    lin_cfg = []
    for k in range(N):
        lin_cfg.append([(k - N//2) * sp, 0.0, 0.0])
    configs.append(lin_cfg)
    
    # 3. Random
    np.random.seed(42)
    for _ in range(5):
        rand_cfg = []
        for _ in range(N):
            rand_cfg.append([np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(0, 2*math.pi)])
        configs.append(rand_cfg)
    
    best_overall_S = float('inf')
    best_overall_state = None
    
    for cfg in configs:
        S, state = optimize(cfg)
        if S < best_overall_S:
            best_overall_S = S
            best_overall_state = state
            
        if time.time() > deadline - 10:
            break
            
    if best_overall_state is None:
        # Fallback
        best_overall_S = 40.0
        best_overall_state = [[4.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        
    # Convert angles to degrees for output
    inner = []
    for x, y, a in best_overall_state:
        ang_deg = math.degrees(a) % 360.0
        inner.append([x, y, ang_deg])
        
    return inner, (0.0, 0.0), best_overall_S, 0.0

# EVOLVE-BLOCK-END
