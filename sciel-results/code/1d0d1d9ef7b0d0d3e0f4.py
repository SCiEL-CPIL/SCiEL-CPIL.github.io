"""Packing 12 unit hexagons in minimal regular hexagon using LP feasibility."""
import numpy as np
from scipy.optimize import linprog
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Hexagon geometry constants
    # Unit regular hexagon side length s=1
    # Circumradius R = 1
    # Inradius r = sqrt(3)/2
    # Distance between centers of two non-overlapping hexagons (same orientation) is at least 2*r = sqrt(3)
    # However, if we allow rotation, the minimum distance can be smaller.
    # For simplicity and robustness in LP, we assume fixed orientation (0 deg) for inner hexagons.
    # If orientation is fixed, the "exclusion zone" is a hexagon of side length 1 centered at the hexagon center,
    # scaled such that the distance between centers must be >= sqrt(3) along the 3 main axes?
    # Actually, for two identical regular hexagons of side 1 with same orientation,
    # they do not overlap iff the distance between centers is >= sqrt(3) in the "flat-to-flat" direction?
    # No, the minimum distance between centers for non-overlap of two regular hexagons of side 1 
    # (same orientation) is sqrt(3).
    # The constraint is: |dx|/1.5 + |dy|/sqrt(3) ... no, that's for axial coords.
    
    # Let's use Cartesian coordinates.
    # A regular hexagon of side 1 centered at (0,0) with flat top (vertices at 0, 60, 120, 180, 240, 300 degrees)
    # has vertices: (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2).
    # The region is defined by 3 pairs of parallel lines:
    # 1. x <= 1, x >= -1  (Vertical sides? No, vertices at (1,0) means side is vertical? No.)
    # Let's check vertices for "pointy top" vs "flat top".
    # Standard math orientation: vertices at angles 0, 60, ...
    # V0 = (1, 0)
    # V1 = (0.5, sqrt(3)/2)
    # Side V0-V1: x from 1 to 0.5, y from 0 to sqrt(3)/2.
    # The supporting lines are:
    # L1: x <= 1 (Rightmost point) -- Wait, the max x is 1.
    # L2: -x + sqrt(3)y <= sqrt(3) (Top right side)
    # L3: -sqrt(3)x - y <= sqrt(3) (Top left side)
    # L4: -x >= -1 => x >= -1 (Leftmost point)
    # L5: x - sqrt(3)y <= sqrt(3) (Bottom right side)
    # L6: sqrt(3)x + y <= sqrt(3) (Bottom left side)
    
    # So the hexagon H_c centered at (cx, cy) is the intersection of:
    # 1. x - cx <= 1
    # 2. -x + sqrt(3)y - (sqrt(3)cy - cx) <= sqrt(3)  => -x + sqrt(3)y <= sqrt(3)(cy) - cx + sqrt(3)
    #    Let's rewrite in standard form: A x + B y <= C
    #    For H_c:
    #    x <= cx + 1
    #    -x + sqrt(3)y <= -cx + sqrt(3)cy + sqrt(3)
    #    -sqrt(3)x - y <= sqrt(3)cx - cy + sqrt(3)  (Check: at (-1, 0) rel to center, -sqrt(3)(-1) - 0 = sqrt(3). Correct.)
    #    x >= cx - 1  => -x <= -cx + 1
    #    x - sqrt(3)y <= cx - sqrt(3)cy + sqrt(3)
    #    sqrt(3)x + y <= sqrt(3)cx + cy + sqrt(3)
    
    # Outer Hexagon H_o centered at (0,0) with side length L (variable).
    # It is a scaled version.
    # Constraints for a point (x,y) to be inside H_o:
    # 1. x <= L
    # 2. -x + sqrt(3)y <= sqrt(3)L
    # 3. -sqrt(3)x - y <= sqrt(3)L
    # 4. -x <= L  => x >= -L
    # 5. x - sqrt(3)y <= sqrt(3)L
    # 6. sqrt(3)x + y <= sqrt(3)L
    
    # We want to place N hexagons such that:
    # 1. All inner hexagons are inside the outer hexagon.
    # 2. Inner hexagons do not overlap each other.
    # 3. Minimize L.
    
    # Decision variables:
    # x_i, y_i for i=0..N-1 (centers of inner hexagons)
    # L (side length of outer hexagon)
    
    # Total variables: 2*N + 1
    
    # Constraints for containment of inner hexagon i in outer hexagon:
    # For each of the 6 inequalities defining H_i, the corresponding inequality for H_o must hold for the "farthest" point.
    # Actually, H_i is contained in H_o iff all vertices of H_i are in H_o.
    # Since both are convex, it suffices to check vertices.
    # But we can derive linear constraints directly.
    # H_i is defined by 6 half-planes. H_o is defined by 6 half-planes.
    # H_i subset H_o iff for every half-plane defining H_o, the maximum value of the linear function over H_i is <= the bound of H_o.
    
    # Let's define the 6 linear functions f_k(x,y) for k=0..5 such that H_o is { (x,y) | f_k(x,y) <= g_k(L) }.
    # f_0(x,y) = x, g_0(L) = L
    # f_1(x,y) = -x + sqrt(3)y, g_1(L) = sqrt(3)L
    # f_2(x,y) = -sqrt(3)x - y, g_2(L) = sqrt(3)L
    # f_3(x,y) = -x, g_3(L) = L
    # f_4(x,y) = x - sqrt(3)y, g_4(L) = sqrt(3)L
    # f_5(x,y) = sqrt(3)x + y, g_5(L) = sqrt(3)L
    
    # For a specific inner hexagon i centered at (x_i, y_i), the maximum of f_k over H_i is f_k(x_i, y_i) + offset_k.
    # What is the offset?
    # The hexagon H_i is the Minkowski sum of {0} and the hexagon shape.
    # The support function of a regular hexagon of side 1 in direction u is the distance from center to the supporting line perpendicular to u.
    # For the directions defined by f_k:
    # Direction 0: (1, 0). Max x on unit hexagon is 1. Offset = 1.
    # Direction 1: (-1, sqrt(3)). Normalize? The linear form is -x + sqrt(3)y.
    # Max of -x + sqrt(3)y on unit hexagon.
    # Vertices: (1,0)->-1; (0.5, sqrt(3)/2)->-0.5+1.5=1; (-0.5, sqrt(3)/2)->0.5+1.5=2; (-1,0)->1; (-0.5, -sqrt(3)/2)->0.5-1.5=-1; (0.5, -sqrt(3)/2)->-0.5-1.5=-2.
    # Max is 2.
    # Wait, let's re-evaluate.
    # f_1(x,y) = -x + sqrt(3)y.
    # At vertex (-0.5, sqrt(3)/2): -(-0.5) + sqrt(3)(sqrt(3)/2) = 0.5 + 1.5 = 2.
    # So offset_1 = 2.
    # Similarly, by symmetry, offset_2 = 2, offset_4 = 2, offset_5 = 2.
    # Offset_0 = 1, Offset_3 = 1.
    
    # So the containment constraints for inner hexagon i are:
    # f_k(x_i, y_i) + offset_k <= g_k(L) for k=0..5.
    
    # This gives 6*N linear inequalities.
    
    # Constraints for non-overlap of inner hexagons i and j:
    # H_i and H_j are disjoint.
    # Since they are convex, they are disjoint iff there exists a separating line.
    # For identical regular hexagons with same orientation, the minimum distance between centers is sqrt(3).
    # Is this sufficient?
    # If distance between centers >= sqrt(3), do they always not overlap?
    # The "exclusion zone" around a hexagon for another identical hexagon to be placed without overlap is the Minkowski sum of the hexagon and the negated hexagon.
    # The Minkowski sum of two regular hexagons of side 1 is a regular hexagon of side 2.
    # The inradius of a hexagon of side 2 is 2*sqrt(3)/2 = sqrt(3).
    # So, if the center of hexagon j is at distance >= sqrt(3) from the center of hexagon i, in the "hexagonal metric", they don't overlap?
    # The condition is that the vector (x_j - x_i, y_j - y_i) must be outside the Minkowski sum hexagon.
    # The Minkowski sum hexagon (let's call it H_2) is centered at 0, side length 2.
    # Its boundaries are:
    # |x| <= 2
    # |-x + sqrt(3)y| <= 2*sqrt(3) ? No.
    # Let's derive the inequalities for the exclusion zone H_excl.
    # H_excl is the set of vectors v such that H_0 + v intersects H_0.
    # This is H_0 + (-H_0) = H_0 + H_0 (since H_0 is symmetric).
    # H_0 is defined by f_k(x,y) <= c_k.
    # H_excl is defined by f_k(x,y) <= 2*c_k.
    # c_0 = 1, c_1 = 2, c_2 = 2, c_3 = 1, c_4 = 2, c_5 = 2.
    # So H_excl is defined by:
    # 1. x <= 2
    # 2. -x + sqrt(3)y <= 4
    # 3. -sqrt(3)x - y <= 4
    # 4. -x <= 2  => x >= -2
    # 5. x - sqrt(3)y <= 4
    # 6. sqrt(3)x + y <= 4
    
    # To ensure H_i and H_j are disjoint, the vector (x_j - x_i, y_j - y_i) must NOT be in H_excl.
    # This means at least one of the 6 inequalities must be violated.
    # This is a disjunctive constraint, which is not linear.
    # However, if we assume a specific lattice structure or fixed relative positions, we can linearize.
    # Or, we can use the fact that for small N, we can guess the adjacency graph or use a heuristic.
    # But the prompt asks for an LP feasibility check.
    # LP cannot handle disjunctions directly.
    
    # Alternative:
    # We can fix the relative positions of the inner hexagons on a lattice.
    # A common dense packing is the hexagonal lattice.
    # In a hexagonal lattice, each hexagon touches 6 neighbors.
    # The distance between centers is exactly sqrt(3).
    # If we place them on a hexagonal lattice, the non-overlap constraint is automatically satisfied if the lattice spacing is correct.
    # Let's try to place the 12 hexagons on a hexagonal lattice.
    # The lattice vectors are u = (1.5, sqrt(3)/2) and v = (0, sqrt(3)).
    # Or axial coordinates.
    # Let's define the positions in axial coordinates (q, r).
    # Cartesian: x = 1.5 * q, y = sqrt(3) * (r + q/2).
    # We need to choose 12 positions (q,r) that form a compact cluster.
    # A hexagonal cluster of radius 2 (side length 3) has 1 + 6 + 12 = 19 positions.
    # We only need 12.
    # We can choose the 12 positions that are closest to the center to minimize the bounding box.
    # The 12 positions with smallest max(|q|, |r|, |q+r|) are:
    # Center (0,0) - 1
    # Ring 1 (6 positions): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (12 positions): (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,1), (1,2) -- wait, let's list them properly.
    # Axial coords for ring 2:
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,1), (2,1), (2,2)?? No.
    # Let's just generate all (q,r) with max(|q|,|r|,|q+r|) <= 2 and pick the 12 with smallest distance to origin.
    
    # If we fix the positions, the only variables are the rotation of the lattice and the position of the outer hexagon?
    # Actually, if we fix the lattice orientation, we can just compute the required L.
    # But the prompt suggests L is a variable in LP.
    # If the positions are fixed, L is determined by the max of the support functions of the cluster.
    # L = max over all vertices of all inner hexagons of the support value.
    
    # Let's try a strategy where we optimize the lattice rotation angle theta.
    # For a given theta, we compute the minimal L required.
    # Then we minimize L over theta.
    # This is a 1D optimization, which is easy and fast.
    
    # L(theta) = max_{i, vertex v of H_i} f_k(R_{-theta} v + R_{-theta} c_i) ... no.
    # The outer hexagon is fixed at 0 rotation.
    # The inner hexagons are rotated by theta.
    # The centers are at lattice points rotated by theta.
    # The vertices of inner hexagon i are c_i + R_theta * v_j.
    # We need to find the minimal L such that all these vertices are inside H_o(L).
    # H_o(L) is defined by f_k(x,y) <= g_k(L).
    # So L must be at least f_k(x,y) / (g_k(L)/L) ? No.
    # f_k(x,y) <= g_k(L) = alpha_k * L.
    # So L >= f_k(x,y) / alpha_k.
    # L(theta) = max_{i, j, k} ( f_k(c_i(theta) + R_theta v_j) / alpha_k )
    # where alpha = [1, sqrt(3), sqrt(3), 1, sqrt(3), sqrt(3)].
    
    # We can minimize this function of theta.
    
    # Generate 12 lattice points
    points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) <= 2:
                x = 1.5 * q
                y = math.sqrt(3) * (r + q/2.0)
                dist = math.sqrt(x*x + y*y)
                points.append((dist, q, r, x, y))
                
    # Sort by distance
    points.sort(key=lambda p: p[0])
    
    # Take the first 12
    # Note: The 7th point might be tied in distance?
    # Ring 0: 1 point (dist 0)
    # Ring 1: 6 points (dist sqrt(3))
    # Ring 2: 12 points (dist 2*sqrt(3) or 3?)
    # Distances in axial:
    # (0,0): 0
    # (1,0): 1.5, sqrt(3)/2 -> dist sqrt(2.25 + 0.75) = sqrt(3)
    # (2,0): 3, 0 -> dist 3
    # (1,1): 1.5, sqrt(3)(1.5) -> dist sqrt(2.25 + 6.75) = 3
    # (0,2): 0, 2sqrt(3) -> dist 2sqrt(3) ≈ 3.46
    # So the 12 points are:
    # 1 from ring 0
    # 6 from ring 1
    # 5 from ring 2?
    # Let's check the distances of ring 2 points.
    # (2,0): 3
    # (1,1): 3
    # (0,2): 3.46
    # (-1,2): 3.46
    # (-2,1): 3
    # (-2,0): 3
    # (-1,-1): 3
    # (0,-2): 3.46
    # (1,-2): 3.46
    # (2,-1): 3
    # (2,1): 3.46? No, (2,1) is max(2,1,3)=3. x=3, y=sqrt(3)(1.5)=2.6. dist=sqrt(9+6.75)=3.96
    # Let's just trust the sort.
    
    selected_points = points[:12]
    centers = np.array([(p[3], p[4]) for p in selected_points])
    
    # Vertices of a unit hexagon (centered at 0, rotation 0)
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.deg2rad(np.arange(0, 360, 60))
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Alpha coefficients for outer hexagon constraints
    # f_0: x <= L
    # f_1: -x + sqrt(3)y <= sqrt(3)L
    # f_2: -sqrt(3)x - y <= sqrt(3)L
    # f_3: -x <= L
    # f_4: x - sqrt(3)y <= sqrt(3)L
    # f_5: sqrt(3)x + y <= sqrt(3)L
    sqrt3 = math.sqrt(3)
    alphas = np.array([1, sqrt3, sqrt3, 1, sqrt3, sqrt3])
    
    # Define function to compute L for a given rotation theta
    def calc_L(theta):
        # Rotate centers
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate vertices
        Rv = vertices @ R.T # (6,2)
        
        # New centers
        new_centers = centers @ R.T # (12,2)
        
        # All vertices of all inner hexagons
        # v_total = new_centers[:, np.newaxis, :] + Rv[np.newaxis, :, :]
        # Shape: (12, 6, 2)
        
        # We need to compute f_k for all these points and divide by alpha_k
        # f_0(x,y) = x
        # f_1(x,y) = -x + sqrt(3)y
        # f_2(x,y) = -sqrt(3)x - y
        # f_3(x,y) = -x
        # f_4(x,y) = x - sqrt(3)y
        # f_5(x,y) = sqrt(3)x + y
        
        # Let X = new_centers[:, 0], Y = new_centers[:, 1]
        # Let Vx = Rv[:, 0], Vy = Rv[:, 1]
        
        X = new_centers[:, 0]
        Y = new_centers[:, 1]
        Vx = Rv[:, 0]
        Vy = Rv[:, 1]
        
        # Total X = X[:, None] + Vx[None, :]
        TX = X[:, np.newaxis] + Vx[np.newaxis, :]
        TY = Y[:, np.newaxis] + Vy[np.newaxis, :]
        
        # Compute f_k
        # f0 = TX
        # f1 = -TX + sqrt3*TY
        # f2 = -sqrt3*TX - TY
        # f3 = -TX
        # f4 = TX - sqrt3*TY
        # f5 = sqrt3*TX + TY
        
        F = np.array([
            TX,
            -TX + sqrt3*TY,
            -sqrt3*TX - TY,
            -TX,
            TX - sqrt3*TY,
            sqrt3*TX + TY
        ]) # Shape (6, 12, 6)
        
        # Divide by alphas
        # alphas shape (6,)
        L_vals = F / alphas[:, np.newaxis, np.newaxis]
        
        # Max over all
        return np.max(L_vals)
        
    # Minimize calc_L(theta)
    # Use a simple grid search + local refinement
    best_L = float('inf')
    best_theta = 0
    
    # Grid search
    thetas = np.linspace(0, np.pi/3, 100) # Period is 60 deg
    for theta in thetas:
        if time.time() > deadline:
            break
        L = calc_L(theta)
        if L < best_L:
            best_L = L
            best_theta = theta
            
    # Local refinement
    if time.time() < deadline - 5:
        # Golden section search or similar
        # Just do a finer grid around best_theta
        fine_thetas = np.linspace(best_theta - 0.01, best_theta + 0.01, 100)
        for theta in fine_thetas:
            if time.time() > deadline:
                break
            L = calc_L(theta)
            if L < best_L:
                best_L = L
                best_theta = theta
                
    # Construct result
    # Inner hexagons: (x, y, angle_degrees)
    c = np.cos(best_theta)
    s = np.sin(best_theta)
    R = np.array([[c, -s], [s, c]])
    final_centers = centers @ R.T
    
    inner = []
    for i in range(N):
        x, y = final_centers[i]
        inner.append([float(x), float(y), float(math.degrees(best_theta))])
        
    # Outer hexagon
    center = (0.0, 0.0)
    side = float(best_L)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
