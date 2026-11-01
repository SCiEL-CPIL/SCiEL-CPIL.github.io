import numpy as np
import time
from scipy.optimize import linprog

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions, angles_deg):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions and angles.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    def check_validity(positions, angles_deg):
        """
        Check if all hexagons are pairwise disjoint using SAT.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            verts = hexagon_vertices(x, y, 1.0, angles_deg[i])
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                p1 = vertices_list[i]
                p2 = vertices_list[j]
                
                # SAT check
                disjoint = False
                for i_sat in range(6):
                    j_sat = (i_sat + 1) % 6
                    edge = p1[j_sat] - p1[i_sat]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - 1e-5 or np.max(proj2) < np.min(proj1) - 1e-5:
                        disjoint = True
                        break
                
                if not disjoint:
                    for i_sat in range(6):
                        j_sat = (i_sat + 1) % 6
                        edge = p2[j_sat] - p2[i_sat]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10:
                            continue
                        normal /= norm
                        proj1 = p1.dot(normal)
                        proj2 = p2.dot(normal)
                        if np.max(proj1) < np.min(proj2) - 1e-5 or np.max(proj2) < np.min(proj1) - 1e-5:
                            disjoint = True
                            break
                    if not disjoint:
                        return False
        return True

    def solve_lp_for_positions(angles_deg, max_side):
        """
        Solve LP to find positions for given angles that fit in hexagon of side max_side.
        Variables: x1, y1, ..., xN, yN (2N variables)
        Constraints:
        1. Containment: For each hexagon i and each of 6 directions, the projection must be within [-max_side, max_side]
        2. Separation: For each pair (i,j), for each separating direction, the gap must be >= 0
        """
        n_vars = 2 * N
        c = np.zeros(n_vars)  # Minimize 0 (feasibility problem)
        
        A_ub = []
        b_ub = []
        
        sqrt3 = np.sqrt(3.0)
        
        # Containment constraints
        # For a regular hexagon of side S centered at origin, the 6 half-spaces are:
        # x + y/sqrt3 <= S, y*2/sqrt3 <= S, -x + y/sqrt3 <= S, -x - y/sqrt3 <= S, -y*2/sqrt3 <= S, x - y/sqrt3 <= S
        # And their negatives for the lower bounds.
        
        # For each hexagon i, we need to ensure all its vertices are inside the outer hexagon.
        # A vertex of hexagon i at position (xi, yi) with angle ai is at:
        # xi + cos(ai + k*60), yi + sin(ai + k*60) for k=0..5
        # This must satisfy the 6 inequalities.
        
        # Instead of vertex-based, use the fact that a convex polygon is inside another convex polygon
        # if and only if all its vertices are inside. But that's 36 constraints per hexagon.
        # Better: use support functions. The support function of a regular hexagon of side 1 in direction u
        # is h(u). The condition that hexagon i is inside outer hexagon of side S centered at origin is:
        # h_outer(u) >= |center_i|_u + h_inner(u) for all u.
        # For regular hexagons, this reduces to checking 6 directions.
        
        # Let's use the 6 normal directions of the outer hexagon.
        # Outer hexagon normals (for angle 0):
        # n1 = (1, 1/sqrt3)/norm, n2 = (0, 1), n3 = (-1, 1/sqrt3)/norm, etc.
        # Actually, the 6 support directions for a regular hexagon of side S are:
        # The maximum of x + y/sqrt3, y*2/sqrt3, -x + y/sqrt3, -x - y/sqrt3, -y*2/sqrt3, x - y/sqrt3
        # must be <= S.
        
        # For inner hexagon i with center (xi, yi) and angle ai, the support in direction d is:
        # d·(xi, yi) + h_inner(d)
        # where h_inner(d) is the support function of a unit regular hexagon rotated by ai.
        
        # For a unit regular hexagon (side 1), the support function in direction (cos θ, sin θ) is:
        # h(θ) = max over vertices of (cos θ * x_v + sin θ * y_v)
        # For a regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300 (angle 0):
        # vertices: (1,0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1,0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
        # h(θ) = max(cos θ, 0.5 cos θ + sqrt3/2 sin θ, -0.5 cos θ + sqrt3/2 sin θ, -cos θ, -0.5 cos θ - sqrt3/2 sin θ, 0.5 cos θ - sqrt3/2 sin θ)
        # = max(|cos θ|, |0.5 cos θ + sqrt3/2 sin θ|, |-0.5 cos θ + sqrt3/2 sin θ|)
        # This is a piecewise linear function.
        
        # For simplicity, let's just use the vertex containment approach with 6 outer constraints.
        # For each inner hexagon i, each of its 6 vertices must satisfy the 6 outer half-space inequalities.
        # That's 36 constraints per inner hexagon, 432 total. Plus separation constraints.
        
        # Actually, let's use a simpler approach: just check the 6 support directions of the outer hexagon.
        # The outer hexagon (side S, angle 0) has 6 outward normals.
        # n_k = (cos(60k), sin(60k)) for k=0..5? No.
        # The 6 directions where the support function is maximized are the vertex normals.
        # For a regular hexagon with flat top (angle 0), the normals are at angles 0, 60, 120, 180, 240, 300.
        # Wait, for a regular hexagon with vertices at 0, 60, ..., the outward normals to the edges are at 30, 90, 150, 210, 270, 330.
        
        # Let's just use the 6 linear forms directly.
        # Outer hexagon constraint: max(L1..L6) <= S
        # This is equivalent to 6 inequalities: L1 <= S, L2 <= S, ..., L6 <= S
        # And -L1 <= S, -L2 <= S, ..., -L6 <= S (for the other side)
        # So 12 inequalities per point.
        
        # For inner hexagon i, we need all its vertices to satisfy these 12 inequalities.
        # Vertex k of hexagon i: (xi + cos(ai + 60k), yi + sin(ai + 60k))
        # L1(vertex) = xi + cos(ai + 60k) + (yi + sin(ai + 60k))/sqrt3 <= S
        # L1(vertex) = (xi + yi/sqrt3) + cos(ai + 60k) + sin(ai + 60k)/sqrt3 <= S
        # So the constraint is: xi + yi/sqrt3 <= S - max_k [cos(ai + 60k) + sin(ai + 60k)/sqrt3]
        
        # Let's compute the max offset for each of the 6 directions for a given angle ai.
        
        for i in range(N):
            ai = np.radians(angles_deg[i])
            # 6 vertex offsets
            offsets = []
            for k in range(6):
                vx = np.cos(ai + np.radians(60 * k))
                vy = np.sin(ai + np.radians(60 * k))
                offsets.append((vx, vy))
            
            # For each of the 6 outer half-space directions, compute the max projection of the inner hexagon
            # Direction 1: (1, 1/sqrt3)
            # Direction 2: (0, 2/sqrt3)
            # Direction 3: (-1, 1/sqrt3)
            # Direction 4: (-1, -1/sqrt3)
            # Direction 5: (0, -2/sqrt3)
            # Direction 6: (1, -1/sqrt3)
            
            dirs = [
                (1, 1/sqrt3),
                (0, 2/sqrt3),
                (-1, 1/sqrt3),
                (-1, -1/sqrt3),
                (0, -2/sqrt3),
                (1, -1/sqrt3)
            ]
            
            for d in dirs:
                # Max projection of inner hexagon vertices in direction d
                max_proj = max(d[0] * vx + d[1] * vy for vx, vy in offsets)
                # Constraint: d[0]*xi + d[1]*yi + max_proj <= max_side
                row = np.zeros(n_vars)
                row[2*i] = d[0]
                row[2*i+1] = d[1]
                A_ub.append(row)
                b_ub.append(max_side - max_proj)
            
            # Negative directions
            for d in dirs:
                min_proj = min(d[0] * vx + d[1] * vy for vx, vy in offsets)
                # Constraint: -d[0]*xi - d[1]*yi - min_proj <= max_side
                # i.e., d[0]*xi + d[1]*yi + min_proj >= -max_side
                # -d[0]*xi - d[1]*yi >= -max_side - min_proj
                row = np.zeros(n_vars)
                row[2*i] = -d[0]
                row[2*i+1] = -d[1]
                A_ub.append(row)
                b_ub.append(-max_side - min_proj)
        
        # Separation constraints: For each pair (i, j), they must be disjoint.
        # This is tricky with LP. Let's use a simpler approach: ensure center distance >= 2 (approximation)
        # Or use linearized separation constraints.
        
        # For now, let's just do containment and then check validity separately.
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds: positions should be within [-max_side, max_side]
        bounds = [(-max_side, max_side)] * n_vars
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                positions = res.x.reshape(N, 2)
                return positions
            else:
                return None
        except:
            return None

    def discrete_search():
        """
        Search over discrete angle combinations (multiples of 30°) and use LP to find positions.
        """
        best_side = float('inf')
        best_positions = None
        best_angles = None
        
        # Canonical angles: multiples of 30°
        canonical_angles = list(range(0, 360, 30))
        
        # For efficiency, try a subset of angle combinations
        # Start with all hexagons at same angle, then vary
        
        # Strategy: Try all combinations where most hexagons are at 0°, and vary a few
        # Or use random sampling of angle combinations
        
        np.random.seed(42)
        
        # First, try all hexagons at angle 0
        angles0 = [0] * N
        positions0 = solve_lp_for_positions(angles0, 5.0)
        if positions0 is not None and check_validity(positions0, angles0):
            side0 = required_outer_side(positions0, angles0)
            if side0 < best_side:
                best_side = side0
                best_positions = positions0.copy()
                best_angles = angles0.copy()
        
        # Try random angle combinations
        for trial in range(200):
            if time.time() > deadline:
                break
            
            # Random angles from canonical set
            angles = [np.random.choice(canonical_angles) for _ in range(N)]
            
            # Try to find positions with LP
            # Start with a generous bound and binary search for minimum
            low = 3.0
            high = 10.0
            
            # First check if feasible at high
            positions = solve_lp_for_positions(angles, high)
            if positions is None:
                continue
            
            if not check_validity(positions, angles):
                continue
            
            # Binary search for minimum side
            for _ in range(10):
                if time.time() > deadline:
                    break
                mid = (low + high) / 2
                positions = solve_lp_for_positions(angles, mid)
                if positions is not None and check_validity(positions, angles):
                    high = mid
                else:
                    low = mid
            
            if check_validity(positions, angles):
                side = required_outer_side(positions, angles)
                if side < best_side:
                    best_side = side
                    best_positions = positions.copy()
                    best_angles = angles.copy()
        
        return best_positions, best_angles, best_side

    # Run discrete search
    best_positions, best_angles, best_side = discrete_search()
    
    # Fallback if no valid solution found
    if best_positions is None:
        # Use a simple valid configuration
        angles = [0] * N
        positions = np.array([[i * 2.5, 0.0] for i in range(N)])
        best_side = required_outer_side(positions, angles)
    
    # Final validation
    if not check_validity(best_positions, best_angles):
        # Spread out more
        positions = best_positions * 1.5
        if not check_validity(positions, best_angles):
            positions = np.array([[i * 3.0, 0.0] for i in range(N)])
            best_angles = [0] * N
        best_positions = positions
        best_side = required_outer_side(best_positions, best_angles)
    
    final_side = required_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
