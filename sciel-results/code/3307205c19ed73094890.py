import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initialize positions
    # 1 center, 6 around, 4 more
    # Use a hexagonal grid initialization slightly perturbed
    positions = []
    angles = []
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # Ring 1 (6 hexagons)
    d1 = 2.0 # Touching distance for centers of unit hexagons is 2
    for k in range(6):
        angle_rad = k * math.pi / 3
        x = d1 * math.cos(angle_rad)
        y = d1 * math.sin(angle_rad)
        positions.append([x, y])
        angles.append(0.0)
        
    # Ring 2 partial (4 hexagons to make 11)
    # Place them in the "gaps" of the first ring, pushed out
    d2 = 3.5 # Slightly further out
    gap_angles_deg = [30, 90, 150, 270]
    for deg in gap_angles_deg:
        angle_rad = math.radians(deg)
        x = d2 * math.cos(angle_rad)
        y = d2 * math.sin(angle_rad)
        positions.append([x, y])
        angles.append(0.0)
        
    # Convert to numpy for speed
    pos = np.array(positions)
    ang = np.array(angles)
    
    def hexagon_vertices(cx, cy, angle_deg, scale=1.0):
        """
        Returns vertices of a regular hexagon with side length 1 (radius 1)
        centered at (cx, cy) rotated by angle_deg.
        Note: Side length 1 => Circumradius 1.
        """
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg) + i * math.pi / 3
            x = cx + scale * math.cos(theta)
            y = cy + scale * math.sin(theta)
            verts.append([x, y])
        return np.array(verts)
        
    def check_overlap_and_extent(pos, ang):
        """
        Returns:
        overlap_penalty: float
        max_extent: float (inradius of enclosing hexagon)
        """
        overlap_pen = 0.0
        max_extent = 0.0
        
        n = len(pos)
        
        # Precompute vertices for all hexagons
        verts_list = []
        for i in range(n):
            verts_list.append(hexagon_vertices(pos[i][0], pos[i][1], ang[i]))
            
        # Check overlaps (simplified: check center distance first)
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Minimum distance between centers of non-overlapping unit hexagons is 2.
                # If dist < 2, they might overlap.
                if dist < 2.0:
                    # Strong penalty
                    overlap_pen += (2.0 - dist) ** 2 * 100.0
                    # More accurate check would require polygon intersection,
                    # but for force-directed, this gradient is enough to push apart.
                    
        # Calculate max extent in 6 directions
        # Directions: 0, 60, 120, 180, 240, 300 degrees
        for k in range(6):
            theta = k * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            
            max_proj = -1e9
            for i in range(n):
                # Project all vertices of hexagon i onto normal (nx, ny)
                # Find max projection
                verts = verts_list[i]
                projs = verts[:, 0] * nx + verts[:, 1] * ny
                p_max = np.max(projs)
                if p_max > max_proj:
                    max_proj = p_max
                    
            # Also check the opposite side? 
            # A convex polygon containing origin is contained in hexagon if 
            # max projection on 6 normals <= R_in.
            # But we need the minimal enclosing hexagon centered at origin.
            # The inradius R_in must be >= max_proj for all 6 directions.
            # Wait, the enclosing hexagon is regular. 
            # If we fix the outer hexagon center at (0,0) and rotation 0,
            # the condition is that all inner hexagons are inside.
            # Distance from origin to side k is R_in.
            # Point p is inside if p . n_k <= R_in for all k.
            # So R_in >= max( p . n_k ) for all points p in all inner hexagons.
            
            if max_proj > max_extent:
                max_extent = max_proj
                
        return overlap_pen, max_extent

    # Optimization Loop
    # We want to minimize max_extent, subject to overlap_pen == 0.
    # We can minimize: total_energy = max_extent + alpha * overlap_pen
    # Using a simple gradient-free method or finite differences?
    # Given time limit, let's use a simple iterative "push and compress" scheme.
    
    # 1. Push apart overlaps
    # 2. Compress towards center
    # Iterate
    
    alpha = 100.0
    lr = 0.01
    steps = 500
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Calculate gradients via finite differences? Too slow for 11*3 variables * 500 steps * 6 checks?
        # Variables: 11 positions (22) + 11 angles (11) = 33 vars.
        # 33 * 500 = 16500 evaluations.
        # Each evaluation is O(N^2) for overlap + O(N*6) for extent. N=11.
        # 16500 * 121 ~ 2M ops. Should be fine in Python with NumPy? Maybe tight.
        # Let's optimize the check function.
        
        # Instead of full gradient descent, use a simpler heuristic:
        # 1. Identify overlapping pairs and push them apart.
        # 2. Calculate current max_extent.
        # 3. Move all points slightly towards center to reduce extent?
        #    This might increase overlaps.
        #    Better: Move points that are contributing to max_extent towards center.
        
        # Let's stick to a robust local search: Nelder-Mead on the energy function.
        # Energy = max_extent + alpha * overlap_pen
        # We need to be careful with the non-smooth max.
        
        pass # Placeholder, will implement Nelder-Mead below

    # Implement Nelder-Mead
    from scipy.optimize import minimize
    
    def energy_func(params):
        # params: [x1, y1, a1, x2, y2, a2, ...]
        p = params.reshape(N, 3)
        pos_np = p[:, :2]
        ang_np = p[:, 2]
        
        # Check overlaps
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos_np[i][0] - pos_np[j][0]
                dy = pos_np[i][1] - pos_np[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 2.0:
                    overlap_pen += (2.0 - dist) ** 2 * 500.0
                    
        if overlap_pen > 1.0:
            return overlap_pen # Return large value if heavy overlap
            
        # Check extent
        max_extent = 0.0
        for k in range(6):
            theta = k * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            
            for i in range(N):
                cx = pos_np[i][0]
                cy = pos_np[i][1]
                rot = ang_np[i]
                
                # Support function of hexagon at (cx, cy) with angle rot
                # Max projection of hexagon onto direction (nx, ny)
                # Local angle of normal relative to hexagon
                local_angle = theta - rot
                # Reduce to [0, pi/3]
                local_angle = local_angle % (math.pi / 3)
                # Distance to nearest multiple of pi/3 (vertex direction)
                # Support function h(theta) = cos(delta) where delta is angle to nearest vertex
                # Vertices are at 0, pi/3, ...
                delta = min(local_angle, math.pi / 3 - local_angle)
                extent = math.cos(delta)
                
                proj = cx * nx + cy * ny + extent
                if proj > max_extent:
                    max_extent = proj
                    
        return max_extent + 10.0 * overlap_pen

    # Initial guess
    x0 = []
    for i in range(N):
        x0.extend([pos[i][0], pos[i][1], ang[i]])
        
    # Run Nelder-Mead
    # Max iterations limited by time
    max_iter = 1000
    timeout = deadline - time.time()
    
    try:
        res = minimize(energy_func, x0, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
        
        if res.success or res.fun < 10.0: # If it didn't fail badly
            best_params = res.x
        else:
            best_params = x0
    except Exception:
        best_params = x0

    # Extract final positions
    p_best = best_params.reshape(N, 3)
    inner = []
    for i in range(N):
        inner.append([p_best[i][0], p_best[i][1], p_best[i][2]])
        
    # Calculate final side length
    # Use the energy function's extent calculation logic
    max_extent_final = 0.0
    for k in range(6):
        theta = k * math.pi / 3
        nx = math.cos(theta)
        ny = math.sin(theta)
        for i in range(N):
            cx = inner[i][0]
            cy = inner[i][1]
            rot = inner[i][2]
            
            local_angle = theta - rot
            local_angle = local_angle % (math.pi / 3)
            delta = min(local_angle, math.pi / 3 - local_angle)
            extent = math.cos(delta)
            
            proj = cx * nx + cy * ny + extent
            if proj > max_extent_final:
                max_extent_final = proj
                
    # Side length S = R_in * 2 / sqrt(3)
    side_length = max_extent_final * 2.0 / math.sqrt(3)
    
    # Sanity check: Ensure no overlaps in final result
    # If overlaps exist, the score is 0. 
    # The optimization should have avoided them, but let's verify.
    has_overlap = False
    for i in range(N):
        for j in range(i+1, N):
            dx = inner[i][0] - inner[j][0]
            dy = inner[i][1] - inner[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 1.99: # Tight check
                has_overlap = True
                
    if has_overlap:
        # Fallback to a known valid configuration if optimization failed
        # Use the initial positions but scaled up slightly to avoid overlap?
        # Or just return the initial positions which might have overlaps?
        # The initial positions had overlaps (dist sqrt(3) < 2).
        # We must return a VALID configuration.
        # Let's force a valid configuration by spreading them out if needed.
        # Simple fix: If overlap, increase distances.
        # But the task is to minimize side length.
        # If the optimizer failed, we return a safe, albeit larger, packing.
        
        # Safe packing: Hexagonal lattice
        # 1 center, 6 ring, 4 outer
        # Ensure distances are >= 2.
        inner_safe = []
        inner_safe.append([0.0, 0.0, 0.0])
        for k in range(6):
            angle_rad = k * math.pi / 3
            x = 2.0 * math.cos(angle_rad)
            y = 2.0 * math.sin(angle_rad)
            inner_safe.append([x, y, 0.0])
        d2 = 3.5
        for deg in [30, 90, 150, 270]:
            angle_rad = math.radians(deg)
            x = d2 * math.cos(angle_rad)
            y = d2 * math.sin(angle_rad)
            inner_safe.append([x, y, 0.0])
            
        # Check if this safe packing is valid
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dx = inner_safe[i][0] - inner_safe[j][0]
                dy = inner_safe[i][1] - inner_safe[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 2.0:
                    valid = False
        
        if valid:
            inner = inner_safe
            # Recalculate side length for safe packing
            max_extent_final = 0.0
            for k in range(6):
                theta = k * math.pi / 3
                nx = math.cos(theta)
                ny = math.sin(theta)
                for i in range(N):
                    cx = inner[i][0]
                    cy = inner[i][1]
                    rot = inner[i][2]
                    local_angle = theta - rot
                    local_angle = local_angle % (math.pi / 3)
                    delta = min(local_angle, math.pi / 3 - local_angle)
                    extent = math.cos(delta)
                    proj = cx * nx + cy * ny + extent
                    if proj > max_extent_final:
                        max_extent_final = proj
            side_length = max_extent_final * 2.0 / math.sqrt(3)
            
    return inner, (0.0, 0.0), side_length, 0.0
# EVOLVE-BLOCK-END
