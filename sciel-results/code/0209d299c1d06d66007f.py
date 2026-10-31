import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def hex_vertices(cx, cy, S, angle_deg):
        """Return 6 vertices of regular hexagon (pointy-top by default)."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + math.radians(60 * i)
            verts.append((cx + S * math.cos(theta), cy + S * math.sin(theta)))
        return verts

    def min_distance_point_to_segment(p, a, b):
        """Calculate minimum distance from point p to segment ab."""
        # Vector AB
        abx = b[0] - a[0]
        aby = b[1] - a[1]
        # Vector AP
        apx = p[0] - a[0]
        apy = p[1] - a[1]
        
        # Length squared of AB
        ab_len_sq = abx*abx + aby*aby
        if ab_len_sq == 0:
            return math.hypot(apx, apy)
        
        # Projection of AP onto AB
        t = (apx * abx + apy * aby) / ab_len_sq
        
        # Clamp t to [0, 1]
        if t < 0:
            t = 0
        elif t > 1:
            t = 1
            
        # Closest point
        px = a[0] + t * abx
        py = a[1] + t * aby
        
        return math.hypot(p[0] - px, p[1] - py)

    def polygon_collision(verts_a, verts_b):
        """
        Check if two convex polygons (hexagons) overlap.
        Returns True if they overlap (invalid), False if disjoint.
        Uses Separating Axis Theorem (SAT) which is efficient for convex polygons.
        """
        axes = []
        for i in range(6):
            # Edge vectors
            p1 = verts_a[i]
            p2 = verts_a[(i + 1) % 6]
            # Normal vector to edge
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            # Perpendicular
            nx = -ey
            ny = ex
            # Normalize (not strictly necessary for SAT if we just compare ranges, but good for consistency)
            length = math.hypot(nx, ny)
            if length > 0:
                axes.append((nx/length, ny/length))
                
        # We only need to check axes from one polygon if we project both onto them?
        # Actually, for convex polygons, checking edges of both is sufficient.
        # But since they are regular hexagons, the set of normals is the same (just rotated).
        # However, their orientations might differ.
        # Let's just check all 6 axes from A and all 6 from B.
        for i in range(6):
            p1 = verts_b[i]
            p2 = verts_b[(i + 1) % 6]
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            nx = -ey
            ny = ex
            length = math.hypot(nx, ny)
            if length > 0:
                axes.append((nx/length, ny/length))

        for nx, ny in axes:
            # Project A
            min_a = float('inf')
            max_a = float('-inf')
            for v in verts_a:
                proj = v[0]*nx + v[1]*ny
                if proj < min_a: min_a = proj
                if proj > max_a: max_a = proj
                
            # Project B
            min_b = float('inf')
            max_b = float('-inf')
            for v in verts_b:
                proj = v[0]*nx + v[1]*ny
                if proj < min_b: min_b = proj
                if proj > max_b: max_b = proj
            
            # Check for gap
            if max_a < min_b or max_b < min_a:
                return False # Disjoint
        
        return True # Overlap

    def compute_required_side(inner, ocx, ocy, oangle):
        """
        Compute the minimum side length S of the outer hexagon centered at (ocx, ocy)
        with angle oangle to contain all inner hexagons.
        """
        # Precompute inner vertices
        inner_verts_list = []
        for (ix, iy, iangle) in inner:
            inner_verts_list.append(hex_vertices(ix, iy, 1.0, iangle))
        
        # Outer hexagon normals
        # The apothem of a hexagon with side S is S * sqrt(3) / 2
        # Distance from center to edge is S * sqrt(3) / 2
        # So for a projection max_proj, we need S * sqrt(3)/2 >= max_proj
        # S >= max_proj * 2 / sqrt(3)
        
        max_S = 0.0
        for i in range(6):
            theta = math.radians(oangle) + math.radians(30 + 60 * i)
            nx = math.cos(theta)
            ny = math.sin(theta)
            
            max_proj = 0.0
            for iv in inner_verts_list:
                for v in iv:
                    proj = (v[0] - ocx) * nx + (v[1] - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
            
            # If all points are on the negative side, max_proj might be negative?
            # No, we take max. If all are negative, the "extent" in that direction is 0?
            # Wait, the hexagon is centered. The constraint is |proj| <= apothem?
            # No, the vertices must be inside.
            # The distance from center to the edge in direction normal n is S*sqrt(3)/2.
            # So for any vertex v, (v - c) . n <= S*sqrt(3)/2 is NOT sufficient if v is on the other side?
            # Actually, a regular hexagon is the intersection of 6 half-planes defined by its edges.
            # The half-plane for edge i is: (x - c) . n_i <= S*sqrt(3)/2.
            # Since it's a convex polygon containing the center, if it satisfies all 6 inequalities, it's inside.
            # So we just need max_proj <= S*sqrt(3)/2 for all 6 directions.
            
            if max_proj > 0:
                S_needed = max_proj * 2.0 / math.sqrt(3)
                if S_needed > max_S:
                    max_S = S_needed
        
        return max_S

    def check_validity(inner, ocx, ocy, S, oangle):
        """Check if layout is valid (disjoint and inside)."""
        # 1. Check disjointness
        verts_list = []
        for (ix, iy, iangle) in inner:
            verts_list.append(hex_vertices(ix, iy, 1.0, iangle))
            
        for i in range(N):
            for j in range(i+1, N):
                if polygon_collision(verts_list[i], verts_list[j]):
                    return False, 0.0

        # 2. Check containment
        # We can just check if the computed S is <= the provided S?
        # Or better, just return the required S. If required S <= provided S, it's valid.
        # But for optimization, we want to minimize S.
        # So we just need to ensure they are disjoint. The "inside" constraint is handled by the objective function (minimizing S).
        # If they are disjoint, we can always fit them in *some* hexagon.
        # The objective is the size of that hexagon.
        # So validity for the search is just "are they disjoint?".
        # The "inside" part is implicit in the S calculation.
        
        return True, 0.0

    def objective_function(params):
        """
        Objective: Minimize the side length S of the outer hexagon.
        Params: [x1, y1, a1, ..., xN, yN, aN, ocx, ocy, oangle]
        Angles are in degrees.
        """
        inner = []
        for i in range(N):
            x = params[2*i]
            y = params[2*i+1]
            a = params[3*N + i]
            inner.append((x, y, a))
            
        ocx = params[3*N + N]
        ocy = params[3*N + N + 1]
        oangle = params[3*N + N + 2]
        
        # Check disjointness
        verts_list = []
        for (ix, iy, iangle) in inner:
            verts_list.append(hex_vertices(ix, iy, 1.0, iangle))
            
        for i in range(N):
            for j in range(i+1, N):
                if polygon_collision(verts_list[i], verts_list[j]):
                    # Penalty for overlap
                    # We can estimate overlap penalty by how much they violate separation
                    # Simple penalty: large number
                    return 1000.0
        
        # Calculate required S
        S = compute_required_side(inner, ocx, ocy, oangle)
        
        # Small penalty if S is very large (shouldn't happen if we have bounds)
        if S > 100:
            return 1000.0
            
        return S

    def local_search(inner, ocx, ocy, oangle, seed):
        """
        Simulated Annealing local search.
        """
        rng = random.Random(seed)
        
        # Initialize state
        cur_inner = [list(x) for x in inner]
        cur_ocx = ocx
        cur_ocy = ocy
        cur_oangle = oangle
        
        cur_params = []
        for x, y, a in cur_inner:
            cur_params.extend([x, y, a])
        cur_params.extend([cur_ocx, cur_ocy, cur_oangle])
        
        cur_cost = objective_function(cur_params)
        best_cost = cur_cost
        best_params = list(cur_params)
        
        T = 1.0
        T_min = 1e-6
        cooling = 0.995
        
        steps = 0
        while time.time() < deadline - 2.0:
            if T < T_min:
                break
                
            # Perturb a random parameter
            idx = rng.randrange(len(cur_params))
            old_val = cur_params[idx]
            
            # Step size
            if idx < 3*N: # Inner hex positions/angles
                if idx % 3 == 2: # Angle
                    step = rng.uniform(-15, 15)
                else: # Position
                    step = rng.gauss(0, 0.2)
            else: # Outer params
                step = rng.gauss(0, 0.1)
                
            cur_params[idx] = old_val + step
            
            new_cost = objective_function(cur_params)
            
            # Acceptance
            if new_cost < cur_cost:
                cur_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_params = list(cur_params)
            else:
                # Metropolis criterion
                delta = new_cost - cur_cost
                if delta < 0:
                    # Should not happen if new_cost < cur_cost handled above
                    pass
                else:
                    # P(accept) = exp(-delta / T)
                    if T > 1e-9:
                        prob = math.exp(-delta / T)
                        if rng.random() < prob:
                            cur_cost = new_cost
                    else:
                        # Revert
                        cur_params[idx] = old_val
            
            T *= cooling
            steps += 1
            
            if steps % 1000 == 0:
                # Check time
                if time.time() > deadline - 5.0:
                    break

        return best_cost, best_params

    # --- Main Construction Logic ---
    
    best_cost = float('inf')
    best_params = None
    
    # 1. Generate diverse initial configurations
    
    # Config 1: Hexagonal Lattice (Dense)
    # Center hex at (0,0)
    # 6 neighbors at distance sqrt(3)
    # 6 next neighbors?
    # 11 hexes: 1 center, 6 ring 1, 4 ring 2?
    # Or just a straight line?
    # Let's try a "flower" shape: 1 center, 6 around, 4 more around the gaps?
    
    sp = math.sqrt(3)
    
    # Lattice 1: Hexagonal packing
    # (0,0)
    # (sp, 0), (sp/2, 1.5), (-sp/2, 1.5), (-sp, 0), (-sp/2, -1.5), (sp/2, -1.5) -> 7 hexes
    # Add 4 more:
    # (2*sp, 0), (sp, 3), (-sp, 3), (0, 3*sqrt(3)/2 * 2?) 
    # Actually, let's just use a linear array for one, and a cluster for another.
    
    configs = []
    
    # A: Linear
    lin = []
    for i in range(N):
        lin.append([(i - N//2) * sp, 0.0, 0.0])
    configs.append((lin, 0.0, 0.0, 0.0))
    
    # B: Hexagonal Cluster (1 center, 6 ring, 4 outer)
    hex_cl = [[0.0, 0.0, 0.0]]
    for i in range(6):
        theta = math.radians(60 * i)
        hex_cl.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
    # Add 4 more in the "corners" of the hexagon shape
    # Corners are at distance 2*sp from center, angles 0, 60, 120...
    # Let's place them at angles 0, 60, 120, 180? No, that overlaps with ring 1 if not careful.
    # Ring 2 positions in hex lattice:
    # (2*sp, 0), (sp, 1.5*sqrt(3)), (-sp, 1.5*sqrt(3)), (-2*sp, 0), ...
    # Let's just add 4 random ones near the perimeter to break symmetry
    for i in range(4):
        theta = math.radians(45 + 90*i)
        r = 2.0 * sp
        hex_cl.append([r * math.cos(theta), r * math.sin(theta), 0.0])
    configs.append((hex_cl, 0.0, 0.0, 0.0))
    
    # C: Triangular Lattice (Rows)
    # Row 1: 4 hexes
    # Row 2: 4 hexes
    # Row 3: 3 hexes
    tri = []
    for i in range(4):
        tri.append([(i - 1.5) * sp, 0.0, 0.0])
    for i in range(4):
        tri.append([(i - 1.5) * sp, 1.5, 0.0])
    for i in range(3):
        tri.append([(i - 1.0) * sp, 3.0, 0.0])
    configs.append((tri, 0.0, 1.5, 0.0))
    
    # D: Random
    rng_init = random.Random(42)
    rand = []
    for i in range(N):
        rand.append([rng_init.uniform(-3, 3), rng_init.uniform(-3, 3), 0.0])
    configs.append((rand, 0.0, 0.0, 0.0))

    # 2. Run local search on each config
    for i, (inner, ocx, ocy, oangle) in enumerate(configs):
        if time.time() > deadline - 10.0:
            break
            
        # Convert to params
        params = []
        for x, y, a in inner:
            params.extend([x, y, a])
        params.extend([ocx, ocy, oangle])
        
        # Run local search
        cost, best_p = local_search(inner, ocx, ocy, oangle, seed=1000 + i)
        
        if cost < best_cost:
            best_cost = cost
            best_params = best_p
    
    # 3. Final refinement if time permits
    if best_params is not None and time.time() < deadline - 5.0:
        # Run one more long SA on the best found
        inner_best = []
        for i in range(N):
            inner_best.append([best_params[2*i], best_params[2*i+1], best_params[3*N+i]])
        ocx_best = best_params[3*N+N]
        ocy_best = best_params[3*N+N+1]
        oangle_best = best_params[3*N+N+2]
        
        # Re-run local search with more iterations / higher T start
        # We can just call local_search again with a different seed
        cost, best_p = local_search(inner_best, ocx_best, ocy_best, oangle_best, seed=9999)
        if cost < best_cost:
            best_cost = cost
            best_params = best_p

    # 4. Construct output
    if best_params is None:
        # Fallback
        inner = [[(i - N//2) * sp, 0.0, 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 10.0, 0.0

    inner_out = []
    for i in range(N):
        x = best_params[2*i]
        y = best_params[2*i+1]
        a = best_params[3*N+i]
        inner_out.append([x, y, a])
        
    ocx = best_params[3*N+N]
    ocy = best_params[3*N+N+1]
    oangle = best_params[3*N+N+2]
    
    # Final verification
    # Calculate S
    S = compute_required_side(inner_out, ocx, ocy, oangle)
    
    # Ensure validity
    # If invalid (overlap), we should probably expand S or move things?
    # The objective function penalizes overlap heavily, so it should be disjoint.
    # But if it's not, we return a safe large hexagon?
    # Let's check disjointness explicitly.
    verts_list = []
    for (ix, iy, iangle) in inner_out:
        verts_list.append(hex_vertices(ix, iy, 1.0, iangle))
        
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if polygon_collision(verts_list[i], verts_list[j]):
                valid = False
                break
        if not valid: break
        
    if not valid:
        # Fallback to linear
        inner_out = [[(i - N//2) * sp, 0.0, 0.0] for i in range(N)]
        S = compute_required_side(inner_out, 0.0, 0.0, 0.0)
        return inner_out, (0.0, 0.0), S, 0.0
        
    return inner_out, (ocx, ocy), S, oangle

# EVOLVE-BLOCK-END
