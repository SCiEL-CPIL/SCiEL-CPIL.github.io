import numpy as np
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Set random seed for reproducibility in evaluation if needed, 
    # but usually we want robustness.
    random.seed(42)
    np.random.seed(42)

    # Helper to convert axial coordinates to cartesian (flat-top)
    # Axial (q, r)
    # x = 3/2 * q
    # y = sqrt(3) / 2 * (q + 2*r)  -- This is pointy top?
    # Let's use a consistent system.
    # Flat top hexagon:
    # Centers are at:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # Let's verify distance between (0,0) and (1,0) -> x=1.5, y=sqrt(3)/2. dist = sqrt(2.25+0.75)=sqrt(3). Correct.
    # Distance between (0,0) and (0,1) -> x=0, y=sqrt(3). dist=sqrt(3). Correct.
    
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        return x, y

    # Generate a set of candidate lattice points
    # We want a cluster of 12.
    # Start with a hexagon of radius 2 (19 points) and pick 12.
    # Or start with a specific shape.
    
    # Let's try a local search starting from a compact cluster.
    # Initial positions: 12 points in a 3x4 grid on the lattice?
    # Let's just use the 12 points closest to the origin in the lattice.
    
    candidates = []
    # Generate points in a radius
    for q in range(-3, 4):
        for r in range(-3, 4):
            x, y = axial_to_cartesian(q, r)
            dist = np.sqrt(x**2 + y**2)
            candidates.append((dist, q, r))
    
    candidates.sort(key=lambda x: x[0])
    
    # Take the first 12 unique points (excluding duplicates if any, though axial is unique)
    # Note: (0,0) is included.
    initial_positions = []
    seen = set()
    for dist, q, r in candidates:
        if (q, r) not in seen:
            initial_positions.append((q, r))
            seen.add((q, r))
            if len(initial_positions) == N:
                break

    # Convert to cartesian
    centers = [axial_to_cartesian(q, r) for q, r in initial_positions]
    
    # Function to calculate vertices of a unit flat-top hexagon centered at (cx, cy)
    def get_hex_vertices(cx, cy):
        # Flat top vertices
        # Angles: 0, 60, 120, 180, 240, 300
        # Radius = 1
        vertices = []
        for i in range(6):
            theta = np.deg2rad(60 * i)
            vx = cx + np.cos(theta)
            vy = cy + np.sin(theta)
            vertices.append((vx, vy))
        return vertices

    # Function to calculate minimal enclosing flat-top hexagon side length S
    # and center (X, Y) for a list of points
    def get_min_enclosing_hex(points):
        if not points:
            return 0, 0, 0
        
        # The constraints for a point (px, py) to be inside a flat-top hex of side S centered at (X, Y) are:
        # 1. |px - X| <= S
        # 2. |py - Y| <= S * sqrt(3) / 2
        # 3. |sqrt(3)*(px - X) + (py - Y)| <= S * sqrt(3)
        
        # We want to minimize S.
        # This is equivalent to finding X, Y such that the max of the three terms is minimized.
        # We can approximate the optimal X, Y by the centroid or by iterative refinement.
        # For a small number of points, we can just check the bounding box in the 3 coordinate systems.
        
        # Let's use a simple optimization: 
        # The optimal center is the center of the smallest enclosing circle in each of the 3 metric spaces?
        # Actually, we can just iterate:
        # Start with X, Y as average of points.
        
        X = np.mean([p[0] for p in points])
        Y = np.mean([p[1] for p in points])
        
        for _ in range(50):
            # Calculate S for current X, Y
            S = 0
            for px, py in points:
                c1 = abs(px - X)
                c2 = abs(py - Y) * 2 / np.sqrt(3)
                c3 = abs(np.sqrt(3)*(px - X) + (py - Y)) / np.sqrt(3)
                S = max(S, c1, c2, c3)
            
            # Refine X, Y? 
            # The gradient of S with respect to X, Y points towards the center of the "tightest" constraint.
            # A simpler heuristic: Move X, Y towards the point that violates the most?
            # Or just use the median of the points in each direction?
            
            # Let's try a local search on X, Y
            best_S = S
            best_X, best_Y = X, Y
            
            # Try small shifts
            for dx in [-0.1, 0, 0.1]:
                for dy in [-0.1, 0, 0.1]:
                    X_new = X + dx
                    Y_new = Y + dy
                    S_new = 0
                    for px, py in points:
                        c1 = abs(px - X_new)
                        c2 = abs(py - Y_new) * 2 / np.sqrt(3)
                        c3 = abs(np.sqrt(3)*(px - X_new) + (py - Y_new)) / np.sqrt(3)
                        S_new = max(S_new, c1, c2, c3)
                    if S_new < best_S:
                        best_S = S_new
                        best_X, best_Y = X_new, Y_new
                        
            if best_S == S:
                break
            X, Y = best_X, best_Y

        return best_S, best_X, best_Y

    # Local search on the positions of the 12 hexes
    # We will keep the positions in Cartesian coordinates.
    current_centers = [c[:] for c in centers]
    
    best_S, best_X, best_Y = get_min_enclosing_hex([v for c in current_centers for v in get_hex_vertices(c[0], c[1])])
    
    # Define neighbors in axial coordinates for potential moves
    # We will map current cartesian back to nearest axial? 
    # Easier: Just allow continuous moves?
    # Or restrict to lattice moves.
    # Let's stick to lattice moves for stability.
    # We need to know the axial coords of the current centers.
    # Since we started on the lattice, we can track them.
    
    current_axial = list(initial_positions)
    
    neighbors = [
        (1, 0), (1, -1), (0, -1),
        (-1, 0), (-1, 1), (0, 1)
    ]
    
    improvements = 0
    for _ in range(200):
        improved = False
        for i in range(N):
            q, r = current_axial[i]
            # Try moving to neighbors
            for dq, dr in neighbors:
                nq, nr = q + dq, r + dr
                # Check if the new position is already occupied
                # Simple check: distance to other centers > 1.7 (approx)
                # Or just check if (nq, nr) is in current_axial
                if (nq, nr) in current_axial:
                    continue
                
                # Calculate new centers
                new_centers = [c[:] for c in current_centers]
                new_axial = [a[:] for a in current_axial]
                
                new_x, new_y = axial_to_cartesian(nq, nr)
                new_centers[i] = [new_x, new_y]
                new_axial[i] = [nq, nr]
                
                # Calculate new S
                all_vertices = [v for c in new_centers for v in get_hex_vertices(c[0], c[1])]
                S_new, X_new, Y_new = get_min_enclosing_hex(all_vertices)
                
                if S_new < best_S - 1e-5:
                    best_S = S_new
                    best_X = X_new
                    best_Y = Y_new
                    current_centers = new_centers
                    current_axial = new_axial
                    improved = True
                    break # Move to next hex
        
        if not improved:
            break

    # Final output
    # inner: list of [x, y, angle]
    # We assume angle 0 for all inner hexes (aligned with outer)
    inner = [[c[0], c[1], 0.0] for c in current_centers]
    
    # Outer hexagon center, side, angle
    # Angle 0 for outer hex (flat-top)
    return inner, (best_X, best_Y), best_S, 0.0

# EVOLVE-BLOCK-END
