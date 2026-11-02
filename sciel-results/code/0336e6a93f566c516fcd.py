import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # 1. Generate hexagonal lattice positions in axial coordinates
    # A hexagon of side length K (in terms of unit hexagons) contains 3K^2 - 3K + 1 hexes.
    # For N=11, K=2 gives 3(4)-6+1 = 7. K=3 gives 3(9)-9+1 = 19.
    # So we need a shape that fits 11. A "hexagon of side 2" is too small (7).
    # A "hexagon of side 3" is 19, which is too big if we want minimal side, but we can take a subset.
    # However, the most efficient packing of 11 hexagons is likely a central hex plus a ring.
    # Let's generate coordinates for a larger hexagon (side 3) and pick the 11 closest to center.
    
    K = 3
    axial_coords = []
    for q in range(-K+1, K):
        for r in range(max(-K+1, -q-K+1), min(K, -q+K)):
            axial_coords.append((q, r))
            
    # Sort by distance from center (0,0)
    def axial_dist_sq(q, r):
        return q*q + q*r + r*r
        
    axial_coords.sort(key=lambda x: axial_dist_sq(x[0], x[1]))
    
    # Take the first N coordinates
    selected_axial = axial_coords[:N]
    
    # 2. Convert axial to Cartesian for flat-top hexagons
    # Standard flat-top unit hexagon (side 1):
    # Center to corner distance = 1
    # x = 3/2 * q
    # y = sqrt(3) * (r + q/2)
    
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        return np.array([x, y])
        
    centers = np.array([axial_to_cartesian(q, r) for q, r in selected_axial])
    angles = np.zeros(N)
    
    # 3. Determine the minimum outer hexagon side length
    # The outer hexagon is also flat-top, centered at (0,0), side length S.
    # A point (x,y) is inside a flat-top hexagon of side S centered at origin if:
    # |x| <= S * 3/2
    # |y| <= S * sqrt(3)
    # |x|/S + |y|/(S*sqrt(3)) <= 1  (Wait, this is not quite right for hexagons)
    
    # Let's use the distance function to the boundary.
    # For a flat-top regular hexagon with side S, the distance from center to a side is S * sqrt(3)/2.
    # The distance from center to a vertex is S.
    # The hexagon is the intersection of 6 strips.
    # Let d = sqrt(3)/2 * S.
    # The inequalities for a point (x,y) to be inside are:
    # 1. |x| <= 1.5 * S
    # 2. |y| <= sqrt(3) * S
    # 3. |x|/1.5 + |y|/sqrt(3) <= S  (This is for the corners? No.)
    
    # Let's derive the support function or just use the distance to the 6 sides.
    # The 6 normal vectors for a flat-top hexagon (pointing outwards) are:
    # 0 deg: (1, 0) -> Side is vertical? No, flat top means flat sides at top/bottom.
    # Flat-top hexagon vertices:
    # (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
    
    # The sides are:
    # Right: x = S
    # Top-Right: x/2 - y*sqrt(3)/2 = -S/2 ? No.
    # Let's check the line through (S, 0) and (S/2, S*sqrt(3)/2).
    # Slope = (S*sqrt(3)/2 - 0) / (S/2 - S) = (S*sqrt(3)/2) / (-S/2) = -sqrt(3).
    # Equation: y - 0 = -sqrt(3) (x - S) => y = -sqrt(3)x + S*sqrt(3) => sqrt(3)x + y = S*sqrt(3).
    # Distance from origin to this line: |0 - S*sqrt(3)| / sqrt(3+1) = S*sqrt(3)/2. Correct.
    
    # So the 6 inequalities for (x,y) to be inside hexagon of side S are:
    # 1. x <= S
    # 2. -x <= S
    # 3. sqrt(3)x + y <= S*sqrt(3)
    # 4. -sqrt(3)x + y <= S*sqrt(3)
    # 5. -sqrt(3)x - y <= S*sqrt(3)  (Wait, symmetry)
    # 6. sqrt(3)x - y <= S*sqrt(3)
    
    # Actually, simpler:
    # d = S * sqrt(3) / 2 is the apothem.
    # The normal vectors are at angles 0, 60, 120, 180, 240, 300.
    # n0 = (1, 0) -> x <= S
    # n1 = (1/2, sqrt(3)/2) -> x/2 + y*sqrt(3)/2 <= S*sqrt(3)/2 => x + y*sqrt(3) <= S*sqrt(3)
    # n2 = (-1/2, sqrt(3)/2) -> -x/2 + y*sqrt(3)/2 <= S*sqrt(3)/2 => -x + y*sqrt(3) <= S*sqrt(3)
    # n3 = (-1, 0) -> -x <= S
    # n4 = (-1/2, -sqrt(3)/2) -> -x/2 - y*sqrt(3)/2 <= S*sqrt(3)/2 => -x - y*sqrt(3) <= S*sqrt(3)
    # n5 = (1/2, -sqrt(3)/2) -> x/2 - y*sqrt(3)/2 <= S*sqrt(3)/2 => x - y*sqrt(3) <= S*sqrt(3)
    
    # We want to find the minimum S such that all inner hexagons are inside.
    # An inner hexagon at center (cx, cy) with side 1 is inside the outer hexagon (side S)
    # if and only if the maximum distance from (cx, cy) to the boundary in the direction of the outer hexagon's normals is <= S - 1?
    # No, it's simpler. The inner hexagon is a set of points. The furthest point of the inner hexagon in direction n must be <= S.
    # The furthest point of a unit hexagon centered at (cx,cy) in direction n is cx*nx + cy*ny + 1 (since the support function of a unit hexagon is 1 in all directions? No, the support function of a unit hexagon depends on direction.
    # For a flat-top unit hexagon, the support function h(n) is:
    # If n is aligned with a normal (0, 60, etc), h(n) = 1 * cos(0) + ... wait.
    # The distance from center to the side perpendicular to n is the apothem a = sqrt(3)/2.
    # The distance to the vertex is 1.
    # So if n is a normal direction, the max projection is a = sqrt(3)/2.
    # If n is a vertex direction, the max projection is 1.
    # The support function is continuous.
    
    # However, since the inner hexagons are axis-aligned (angle 0), and the outer is axis-aligned, we can just check the 6 vertices of each inner hexagon against the outer hexagon inequalities.
    # Or, we can just optimize S by checking the constraint:
    # For each inner hexagon i, and for each of the 6 outer normals n_j:
    #   cx_i * n_j_x + cy_i * n_j_y + h_i(n_j) <= S
    # where h_i(n_j) is the support function of the unit hexagon in direction n_j.
    # Since the inner hexagons are flat-top and axis-aligned, their normals are also at 0, 60, etc.
    # So for n_j at 0 deg (1,0), h_i(n_j) = 1 (vertex is at x=1). Wait, for flat-top, the vertices are at (1,0), (1/2, sqrt(3)/2)...
    # So in direction (1,0), the max x is 1.
    # In direction (1/2, sqrt(3)/2), which is a normal direction for the inner hexagon?
    # Inner hexagon normals are (1,0), (1/2, sqrt(3)/2), etc.
    # So h_i(1/2, sqrt(3)/2) is the apothem = sqrt(3)/2.
    
    # Let's compute h(n) for the 6 directions.
    # n0 = (1, 0). Inner hex vertex at (1,0). h = 1.
    # n1 = (1/2, sqrt(3)/2). Inner hex side perpendicular to this. Apothem = sqrt(3)/2. h = sqrt(3)/2.
    # n2 = (-1/2, sqrt(3)/2). Apothem = sqrt(3)/2. h = sqrt(3)/2.
    # n3 = (-1, 0). Vertex at (-1,0). h = 1.
    # n4 = (-1/2, -sqrt(3)/2). Apothem = sqrt(3)/2. h = sqrt(3)/2.
    # n5 = (1/2, -sqrt(3)/2). Apothem = sqrt(3)/2. h = sqrt(3)/2.
    
    # So the constraints are:
    # S >= cx_i * 1 + cy_i * 0 + 1
    # S >= cx_i * 0.5 + cy_i * sqrt(3)/2 + sqrt(3)/2
    # S >= cx_i * -0.5 + cy_i * sqrt(3)/2 + sqrt(3)/2
    # S >= cx_i * -1 + cy_i * 0 + 1
    # S >= cx_i * -0.5 + cy_i * -sqrt(3)/2 + sqrt(3)/2
    # S >= cx_i * 0.5 + cy_i * -sqrt(3)/2 + sqrt(3)/2
    
    # We want to minimize S. This is a linear program in S, but we can just calculate the required S for the current centers and then maybe tweak centers?
    # Actually, the centers are fixed by the lattice. We just need to calculate the minimum S that satisfies all constraints.
    # S_min = max over all i, j of (cx_i * n_jx + cy_i * n_jy + h(n_j))
    
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    h_values = np.array([1.0, np.sqrt(3)/2, np.sqrt(3)/2, 1.0, np.sqrt(3)/2, np.sqrt(3)/2])
    
    # Calculate required S
    # For each center, project onto normals
    # centers: (N, 2)
    # normals: (6, 2)
    # projections = centers @ normals.T -> (N, 6)
    projections = centers @ normals.T
    # Add h_values
    # h_values is (6,), so add to each column
    required_S = np.max(projections + h_values)
    
    current_S = required_S
    
    # Now, can we improve by moving the centers?
    # The lattice is rigid, but we can scale it? No, inner hexagons are unit.
    # We can try to slightly perturb the centers to reduce the max projection.
    # This is a non-convex problem if we allow arbitrary movement, but for small perturbations, it's local.
    # Let's define an objective function that minimizes S given the centers.
    # S(centers) = max_{i,j} (cx_i n_jx + cy_i n_jy + h_j)
    # This function is convex (max of linear functions).
    # So we can use gradient descent or a solver.
    # But we also have the constraint that inner hexagons must not overlap.
    # In the lattice, they touch. If we move them, they might overlap.
    # Overlap constraint: distance between centers >= 2 * apothem = sqrt(3) (for flat-top, touching sides).
    # Actually, for two unit hexagons to not overlap, the distance between centers must be at least sqrt(3) if they are aligned.
    # If we rotate them, it's different, but let's keep them aligned.
    
    # Let's use scipy.optimize.minimize to minimize S(centers) subject to non-overlap.
    # Variables: 2*N coordinates.
    # Objective: S(x) = max( ... )
    # Constraints: d_ij >= sqrt(3) for all i != j.
    
    def objective(params):
        c = params.reshape(-1, 2)
        proj = c @ normals.T
        return np.max(proj + h_values)
        
    def constraint_overlap(params):
        c = params.reshape(-1, 2)
        # Check all pairs
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(c[i] - c[j])
                if d < min_dist:
                    min_dist = d
        # Constraint: min_dist >= sqrt(3)
        # Return min_dist - sqrt(3)
        return min_dist - np.sqrt(3)
        
    # Initial guess
    x0 = centers.flatten()
    
    # We might not have time for a full optimization, but let's try a quick one.
    # Since the lattice is optimal for infinite packing, the boundary is the only issue.
    # The "bulging" of the lattice might be reducible.
    
    # Let's just return the lattice result if time is short.
    if time.time() > deadline - 10:
         return [[c[0], c[1], 0.0] for c in centers], (0.0, 0.0), current_S, 0.0
    
    # Try to optimize
    # Note: The constraint is non-smooth (max). Nelder-Mead might work.
    # But we must ensure we don't overlap.
    # The lattice has distance sqrt(3) between neighbors.
    # If we move them, we might violate this.
    
    # Let's try a simple local search:
    # Try to move the outermost hexagons slightly inwards.
    
    # Identify the hexagons that are at the boundary (maximizing S).
    # This is complex. Let's just use the lattice result. It's a very strong baseline.
    # The theoretical minimum for 11 hexagons is likely close to the lattice bound.
    
    # Let's try one step of gradient descent on the objective, ignoring overlap for a moment to see the gradient direction, then project back?
    # No, let's just return the lattice. It's safe and fast.
    
    # Wait, the prompt says "minimizing the outer side length".
    # The lattice side length for K=3 is 3?
    # Let's check the side length of the bounding hexagon of the lattice.
    # The lattice points are in a hexagon of side 3 (axial).
    # The Cartesian extent is roughly 3 units.
    # The calculated S should be around 3 or 4.
    
    # Let's verify S for the single hexagon at (0,0).
    # proj = 0. S = 1.
    # For a hexagon at (1.5, 0) (q=1, r=0).
    # proj onto (1,0) is 1.5. S = 1.5 + 1 = 2.5.
    # For a hexagon at (1.5, sqrt(3)/2) (q=1, r=-1? No, q=1, r=0 is (1.5, 0). q=1, r=-1 is (1.5, sqrt(3)/2 * (-1 + 0.5) = -sqrt(3)/4).
    # Let's check the farthest point in the lattice.
    # The lattice is symmetric. The farthest points are the vertices of the large hexagon.
    # For K=3, the vertices are at axial (2, -1), (3, -3), etc?
    # No, the set is q,r in [-2, 2] with |q|+|r|+|q+r| <= 4? No, K=3 means max distance 2?
    # My loop: q in -2..2, r in ...
    # For K=3, range is -2 to 2.
    # Max q is 2.
    # Point (2, 0) -> x = 3, y = 0.
    # proj onto (1,0) is 3. S = 3 + 1 = 4.
    # So S=4.
    
    # Can we do better than 4?
    # If we pack 11 hexagons, maybe we can squeeze them into S < 4?
    # The area of 11 unit hexagons is 11 * 3*sqrt(3)/2 ≈ 28.5.
    # Area of outer hexagon side S is 3*sqrt(3)/2 * S^2.
    # 28.5 / (3*sqrt(3)/2) ≈ 11.
    # So S^2 >= 11 => S >= sqrt(11) ≈ 3.31.
    # So 4 is not optimal. 3.31 is the lower bound.
    # The lattice gives 4.
    # We can likely improve by moving the boundary hexagons inwards.
    
    # Let's try to optimize.
    # We will use a simple local search:
    # 1. Start with lattice centers.
    # 2. Calculate S.
    # 3. Identify the "active" constraints (hexagons/directions achieving max).
    # 4. Move those hexagons slightly towards the center.
    # 5. Check for overlaps. If overlap, revert or adjust.
    # 6. Repeat.
    
    best_S = current_S
    best_centers = centers.copy()
    
    # Simple iterative refinement
    # Move boundary hexagons towards center by a small epsilon.
    # Which ones are boundary? Those with proj + h > best_S - delta.
    
    # Let's just try a random perturbation and keep if S decreases and no overlap.
    # This is a stochastic local search.
    
    for _ in range(100):
        if time.time() > deadline - 5:
            break
            
        # Pick a random hexagon
        i = np.random.randint(N)
        # Move it slightly towards center
        direction = -best_centers[i]
        norm = np.linalg.norm(direction)
        if norm == 0: continue
        direction /= norm
        
        # Try a small step
        step = 0.01
        new_center = best_centers[i] + direction * step
        
        # Check overlap with others
        valid = True
        for j in range(N):
            if i == j: continue
            d = np.linalg.norm(new_center - best_centers[j])
            if d < np.sqrt(3) - 1e-6:
                valid = False
                break
                
        if not valid:
            continue
            
        # Calculate new S
        c = best_centers.copy()
        c[i] = new_center
        proj = c @ normals.T
        new_S = np.max(proj + h_values)
        
        if new_S < best_S:
            best_S = new_S
            best_centers = c
        else:
            # If it doesn't improve, maybe try moving it out? No, we want to minimize S.
            # Maybe try moving a different hexagon.
            pass
            
    # Return best found
    return [[c[0], c[1], 0.0] for c in best_centers], (0.0, 0.0), best_S, 0.0
