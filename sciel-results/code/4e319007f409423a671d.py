import math
import time
import itertools
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    
    # 1. Define the lattice and candidate generation
    # Hexagonal lattice vectors for side length 1 hexagons.
    # Distance between centers of touching hexagons is sqrt(3).
    # Let's use a basis:
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 3/2)  -> Note: height is 3/2 * side? No.
    # For flat-top hex (vertex at 0 deg):
    # Width = 2, Height = sqrt(3).
    # Center to center horizontal: sqrt(3).
    # Center to center diagonal: sqrt(3).
    # Let's use the standard axial coordinates (q, r) which map to Cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # This ensures distance between (0,0) and (1,0) is sqrt(3).
    # Distance between (0,0) and (0,1) is sqrt( (sqrt(3)/2)^2 + (3/2)^2 ) = sqrt(3/4 + 9/4) = sqrt(12/4) = sqrt(3). Correct.
    
    SQRT3 = math.sqrt(3)
    
    # Generate a grid of potential center positions in axial coordinates
    # We need to find sets of N=11 points that are "compact".
    # A good heuristic is to generate all subsets of size N from a small grid
    # and filter for those with small bounding box or small moment of inertia.
    
    # Grid range: -3 to 3 should be enough for 11 hexagons.
    # Total points in grid: 7*7 = 49.
    # Subsets of size 11 from 49 is huge (49C11 ~ 10^11). Too many.
    
    # Better strategy: Generate "connected" clusters or simply use a known optimal packing structure.
    # For N=11, the optimal packing is likely a hexagonal shape of radius 1 (7 hexes) plus 4 more.
    # Or a shape closer to a circle.
    # Let's enumerate all subsets of size N from a smaller, denser grid, but prune aggressively.
    
    # Alternative: Use the "blob" property.
    # Generate all combinations of N points from a grid of radius R=2 (axial).
    # Grid points with |q|+|r|+|q+r| <= 2R? No, just square grid in axial.
    # Let's try R=2. Points: q in [-2,2], r in [-2,2]. 5x5=25 points.
    # 25C11 is ~ 4.4 million. This is manageable if the check is fast.
    # But we need to check disjointness (which is guaranteed by lattice if spacing is correct)
    # and compute the enclosing hexagon side length.
    
    # Wait, if we place them on a lattice with spacing sqrt(3), they are touching/disjoint.
    # The only thing to compute is the minimal enclosing hexagon.
    # We can do this vectorized or with a fast loop.
    
    # Let's refine the grid. R=2 might be too small for some shapes, R=3 is 7x7=49 points.
    # 49C11 is too big.
    
    # Strategy: 
    # 1. Identify the set of "core" points (origin and neighbors).
    # 2. Use a beam search or greedy addition? No, we want global optimum.
    # 3. Let's stick to R=2 (25 points) and 25C11. 
    #    We can optimize the evaluation.
    #    For each subset:
    #    - Compute Cartesian coordinates.
    #    - Compute convex hull or just max projection on 6 axes.
    #    - The minimal enclosing hexagon centered at (0,0) with angle 0 has side length S.
    #    - The condition for a point (x,y) to be inside a hexagon of side S centered at origin (flat-top, vertex at 0) is:
    #      |x| <= S
    #      |y| <= S * sqrt(3)
    #      |x/2 + y/sqrt(3)| <= S  (This is for the other axes?)
    #      Actually, the support function of a hexagon with vertices at angles 0, 60, ... is:
    #      h(theta) = S * cos(theta - nearest_vertex_angle) ... no.
    #      For a regular hexagon with circumradius S (side length S), the apothem is S*sqrt(3)/2.
    #      The width in direction theta is 2 * S * cos(theta - 30k) ... 
    #      Let's use the definition: Vertices at (S, 0), (S/2, S*sqrt(3)/2), ...
    #      The inequalities for the interior are:
    #      x <= S
    #      -x <= S
    #      x/2 + y*sqrt(3)/2 <= S
    #      -x/2 - y*sqrt(3)/2 <= S
    #      x/2 - y*sqrt(3)/2 <= S
    #      -x/2 + y*sqrt(3)/2 <= S
    #      y*sqrt(3) <= S*sqrt(3) => y <= S  (Wait, vertex is at 90? No, vertices are at 0, 60, 120. So max y is at 60 deg vertex? No, max y is at 90 deg which is between vertices. 
    #      Let's re-verify.
    #      Vertices: (S,0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S,0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2).
    #      Max y is S*sqrt(3)/2.
      #      So the bounding box is x in [-S, S], y in [-S*sqrt(3)/2, S*sqrt(3)/2].
      #      The constraints are:
      #      x <= S
      #      -x <= S
      #      x/2 + y*sqrt(3)/2 <= S
      #      -x/2 - y*sqrt(3)/2 <= S
      #      x/2 - y*sqrt(3)/2 <= S
      #      -x/2 + y*sqrt(3)/2 <= S
      
      #      To find the minimal S for a set of points (xi, yi) centered at (cx, cy):
      #      We need to find S such that for all i:
      #      xi - cx <= S
      #      cx - xi <= S
      #      (xi - cx)/2 + (yi - cy)*sqrt(3)/2 <= S
      #      -(xi - cx)/2 - (yi - cy)*sqrt(3)/2 <= S
      #      (xi - cx)/2 - (yi - cy)*sqrt(3)/2 <= S
      #      -(xi - cx)/2 + (yi - cy)*sqrt(3)/2 <= S
      
      #      So S >= max( max(xi-cx), max(cx-xi), max((xi-cx)/2 + (yi-cy)*sqrt(3)/2), ... )
      #      We can optimize by centering the cluster at its centroid or bounding box center.
      #      Actually, the optimal center might not be the centroid.
      #      However, for small N, we can just compute the required S for center (0,0) if we center the points first.
      #      Even better: The minimal enclosing hexagon problem can be solved by checking the convex hull.
      #      But since we are enumerating, let's just compute the "radius" of the set of points relative to the origin,
      #      assuming we can translate the whole set.
      #      To minimize S, we should center the set such that the max projection is minimized.
      #      This is the Chebyshev center problem for a hexagon.
      #      For a set of points, the minimal S for a hexagon centered at (cx, cy) is:
      #      S(cx, cy) = max( 
      #          max_i (xi - cx),
      #          max_i (cx - xi),
      #          max_i ( (xi - cx)/2 + (yi - cy)*sqrt(3)/2 ),
      #          max_i ( -(xi - cx)/2 - (yi - cy)*sqrt(3)/2 ),
      #          max_i ( (xi - cx)/2 - (yi - cy)*sqrt(3)/2 ),
      #          max_i ( -(xi - cx)/2 + (yi - cy)*sqrt(3)/2 )
      #      )
      #      This function is convex in (cx, cy). We can find the minimum using ternary search or just check a few candidate centers (e.g., centroid, bounding box center).
      #      Given the small N, checking the centroid is a good approximation, but might not be optimal.
      #      However, for lattice packings, the symmetry often makes the centroid or a lattice point the optimal center.

    # Let's implement the enumeration.
    
    # Generate grid points
    grid_points_axial = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            grid_points_axial.append((q, r))
            
    # Filter to keep only points that could be part of a compact cluster?
    # Actually, let's just take the 25 points from R=2 (q,r in -2..2) to keep 25C11 tractable.
    # 25C11 = 4,457,400.
    # For each, we need to compute S.
    # If we do this in Python, 4.4M iterations might be too slow (120s limit).
    # 4.4M * ~100 ops = 440M ops. Might take ~10-20s in Python if optimized, but risky.
    # Let's reduce the search space.
    
    # Observation: The optimal packing will be connected.
    # We can generate connected subsets of size 11 from the grid.
    # Start with origin. Add neighbors.
    # This is much smaller.
    
    # Let's use a DFS/BFS to generate all connected sets of size 11 from the grid (R=3).
    # This will prune 99% of the subsets.
    
    def generate_connected_sets(grid_pts, target_size):
        # grid_pts is a list of (q, r)
        # Map point to index for fast lookup
        point_to_idx = {p: i for i, p in enumerate(grid_pts)}
        
        # Precompute neighbors
        neighbors = [[] for _ in grid_pts]
        for i, (q, r) in enumerate(grid_pts):
            for dq, dr in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
                nq, nr = q + dq, r + dr
                if (nq, nr) in point_to_idx:
                    neighbors[i].append(point_to_idx[(nq, nr)])
        
        sets_found = []
        
        def dfs(current_set, current_max_idx):
            if len(current_set) == target_size:
                # Canonicalize to avoid duplicates (sort indices)
                sets_found.append(frozenset(current_set))
                return
            if len(current_set) > target_size:
                return
            
            # Find available neighbors
            # To ensure we don't miss any, we can just add any neighbor of any point in the set
            # But to avoid duplicates, we can enforce an ordering?
            # Simpler: Just use a set of visited configurations? No, too big.
            # Standard trick: Only add points with index > min(current_set)? No, that breaks connectivity if we skip.
            # Let's just generate all and use a set of frozensets to deduplicate.
            # Since the number of connected sets is small, this is fine.
            
            candidates = set()
            for idx in current_set:
                for n in neighbors[idx]:
                    candidates.add(n)
            
            for n in candidates:
                if n not in current_set:
                    dfs(current_set | {n}, max(current_set) if current_set else 0)
                    
        # Start with each point? No, just start with (0,0) which is index of (0,0)
        # But the cluster might not contain (0,0).
        # However, due to symmetry, any connected cluster can be translated.
        # So we can just generate all connected clusters of size 11 that contain the origin?
        # No, we need to generate the *shape*.
        # If we generate all connected sets of size 11 from the grid, we will get many translations.
        # We can deduplicate by checking the "shape" (relative positions).
        
        # Better: Generate all connected sets of size 11.
        # There are not that many.
        
        # To be safe and fast, let's limit the grid to R=2.
        # And just run the DFS.
        
        # Initialize
        # We need to start from every point? No, we can just start from (0,0) and assume we can translate.
        # Wait, if we only start from (0,0), we miss shapes that don't include (0,0) in their "canonical" form?
        # No, any shape can be translated so that one of its points is at (0,0).
        # So generating all connected sets containing (0,0) is sufficient to capture all shapes.
        
        idx_origin = point_to_idx.get((0,0))
        if idx_origin is not None:
            dfs({idx_origin}, 0)
            
        # Deduplicate
        unique_sets = list(set(sets_found))
        return unique_sets

    # Run generation
    # Use R=2 grid
    grid_pts_R2 = [(q, r) for q in range(-2, 3) for r in range(-2, 3)]
    connected_sets = generate_connected_sets(grid_pts_R2, N)
    
    best_S = float('inf')
    best_config = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Precompute cartesian coordinates for grid points
    def axial_to_cartesian(q, r):
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    cart_coords = [axial_to_cartesian(q, r) for q, r in grid_pts_R2]
    
    # Evaluate each connected set
    for s in connected_sets:
        if time.time() > deadline:
            break
            
        # Get cartesian coordinates of the points in this set
        pts = [cart_coords[i] for i in s]
        
        # We need to find the minimal enclosing hexagon side length S.
        # We can try centering at the centroid.
        cx = sum(p[0] for p in pts) / N
        cy = sum(p[1] for p in pts) / N
        
        # Translate points to center at (0,0)
        centered_pts = [(p[0] - cx, p[1] - cy) for p in pts]
        
        # Compute required S for center (0,0)
        # S = max of the 6 projections
        max_proj = 0.0
        for x, y in centered_pts:
            p1 = x
            p2 = -x
            p3 = x / 2.0 + y * SQRT3 / 2.0
            p4 = -x / 2.0 - y * SQRT3 / 2.0
            p5 = x / 2.0 - y * SQRT3 / 2.0
            p6 = -x / 2.0 + y * SQRT3 / 2.0
            
            m = max(p1, p2, p3, p4, p5, p6)
            if m > max_proj:
                max_proj = m
                
        # Note: This S corresponds to the hexagon with vertices at (S,0)...
        # The side length of this hexagon is exactly S.
        
        if max_proj < best_S:
            best_S = max_proj
            # Store the configuration
            # We need to return the inner hexagons' positions.
            # The positions are the original pts, but we should center them?
            # The evaluator takes (inner, center, side, angle).
            # inner: list of (x, y, angle)
            # center: (cx, cy)
            # side: S
            # angle: 0
            # If we return the points as they are (relative to global origin), and center as (cx, cy),
            # the evaluator will check if they fit in the hexagon at (cx, cy) with side S.
            # So we should return the absolute coordinates.
            
            # However, we computed S based on centered points.
            # So the hexagon is at (cx, cy) with side S.
            # The points are pts.
            
            best_config = [(p[0], p[1], 0.0) for p in pts]
            best_center = (cx, cy)
            best_angle = 0.0

    # If no connected set found or best_S is inf, fallback to a simple line
    if best_config is None:
        # Fallback: Place in a line
        best_config = []
        for i in range(N):
            x = i * SQRT3
            y = 0.0
            best_config.append((x, y, 0.0))
        cx = sum(p[0] for p in best_config) / N
        cy = 0.0
        # Compute S for this line
        max_proj = 0.0
        for x, y, _ in best_config:
            dx = x - cx
            dy = y - cy
            p1 = dx
            p2 = -dx
            p3 = dx / 2.0 + dy * SQRT3 / 2.0
            p4 = -dx / 2.0 - dy * SQRT3 / 2.0
            p5 = dx / 2.0 - dy * SQRT3 / 2.0
            p6 = -dx / 2.0 + dy * SQRT3 / 2.0
            m = max(p1, p2, p3, p4, p5, p6)
            if m > max_proj:
                max_proj = m
        best_S = max_proj
        best_center = (cx, cy)
        best_angle = 0.0

    # Final check: Ensure validity
    # The lattice spacing is sqrt(3), so hexagons are touching.
    # The evaluator might require strict disjointness (distance > sqrt(3))?
    # "Disjoint" usually allows touching. If not, we can scale up slightly.
    # Let's add a tiny epsilon to the side length to be safe?
    # Or scale the positions?
    # If we scale positions by (1+eps), the enclosing hexagon grows.
    # Let's just return the exact lattice solution.
    
    # One more check: The angle of the inner hexagons.
    # We set them to 0.0. This aligns them with the outer hexagon (if outer is 0).
    # This is optimal for lattice packing.
    
    return best_config, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
