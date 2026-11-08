import numpy as np
import time

N = 11
SQRT3 = np.sqrt(3.0)

def s_req(x, y):
    """Calculate the minimal side length of a flat-top hexagon centered at (0,0) that contains (x,y)."""
    # Inequalities for flat-top hexagon of side s:
    # |x| <= s
    # |y| <= sqrt(3)/2 * s  => s >= 2/sqrt(3) * |y|
    # |0.5*x + sqrt(3)/2*y| <= s
    # |0.5*x - sqrt(3)/2*y| <= s
    
    term1 = abs(x)
    term2 = (2.0 / SQRT3) * abs(y)
    term3 = abs(0.5 * x + (SQRT3 / 2.0) * y)
    term4 = abs(0.5 * x - (SQRT3 / 2.0) * y)
    return max(term1, term2, term3, term4)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_L = float('inf')
    best_positions = None
    
    # Initial configuration: A grid or cluster
    # Let's start with a compact cluster of 11.
    # 1 center, 6 neighbors, 4 more.
    # To ensure validity initially, we can start with a very large L and shrink.
    # Or start with a valid lattice packing.
    
    # A valid packing for 11 hexagons:
    # Row 0: 5 hexagons? No, let's use the lattice coordinates.
    # Hex lattice vectors: v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 1.5)
    # Let's generate 11 points on a hex lattice centered at 0.
    # Indices (i, j) such that distance from center is small.
    # (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1) -> 7 points (flower)
    # Next shell: (2,0), (-2,0), (0,2), (0,-2), (2,-2), (-2,2), (1,1), (-1,-1) ...
    # Let's just pick 11 points from the lattice.
    
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * SQRT3 + j * SQRT3 / 2
            y = j * 1.5
            # Approximate distance from origin
            if i*i + j*j <= 12: # Rough bound
                lattice_points.append((x, y))
    
    # Sort by distance to origin and take the first 11?
    # Or just take a specific set.
    # Let's use the 7 flower and add 4 from the next ring.
    # Flower: (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # Lattice coords (i,j) -> x = (i + j/2)*sqrt(3), y = j*1.5
    # (0,0) -> (0,0)
    # (1,0) -> (sqrt(3), 0)
    # (-1,0) -> (-sqrt(3), 0)
    # (0,1) -> (sqrt(3)/2, 1.5)
    # (0,-1) -> (-sqrt(3)/2, -1.5)
    # (1,-1) -> (sqrt(3)/2, -1.5)
    # (-1,1) -> (-sqrt(3)/2, 1.5)
    
    # Next ring candidates:
    # (2,0) -> (2*sqrt(3), 0)
    # (-2,0) -> (-2*sqrt(3), 0)
    # (0,2) -> (sqrt(3), 3.0)
    # (0,-2) -> (-sqrt(3), -3.0)
    # (2,-2) -> (sqrt(3), -3.0)
    # (-2,2) -> (-sqrt(3), 3.0)
    # (1,1) -> (1.5*sqrt(3), 1.5)
    # (-1,-1) -> (-1.5*sqrt(3), -1.5)
    
    # Let's pick (2,0), (-2,0), (1,1), (-1,-1) as the extra 4.
    # This forms a somewhat compact shape.
    
    init_coords = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1),
        (2, 0),
        (-2, 0),
        (1, 1),
        (-1, -1)
    ]
    
    init_centers = np.array([
        [ (i + j/2.0) * SQRT3, j * 1.5 ] for i, j in init_coords
    ])
    
    # Optimization Loop
    # We will use a simple iterative improvement.
    # Objective: Minimize L = max(s_req(x_i, y_i))
    # Constraint: ||p_i - p_j|| >= sqrt(3) for all i != j
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        if restart == 0:
            centers = init_centers.copy()
        else:
            centers = init_centers + np.random.normal(0, 0.5, init_centers.shape)
        
        step_size = 0.05
        for i_iter in range(500):
            if time.time() > deadline:
                break
                
            # 1. Calculate current L
            s_reqs = np.array([s_req(centers[k, 0], centers[k, 1]) for k in range(N)])
            current_L = np.max(s_reqs)
            
            # 2. Calculate forces
            # Force to reduce L: Move points towards origin, weighted by how much they contribute to L.
            # Force to resolve overlap: Push apart points that are too close.
            
            grad = np.zeros_like(centers)
            
            # Overlap forces
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = SQRT3
                    if dist < min_dist:
                        # Push apart
                        if dist < 1e-6:
                            # Random push
                            angle = np.random.rand() * 2 * np.pi
                            grad[i, 0] += np.cos(angle)
                            grad[i, 1] += np.sin(angle)
                            grad[j, 0] -= np.cos(angle)
                            grad[j, 1] -= np.sin(angle)
                        else:
                            push_strength = (min_dist - dist)
                            grad[i, 0] += (dx / dist) * push_strength
                            grad[i, 1] += (dy / dist) * push_strength
                            grad[j, 0] -= (dx / dist) * push_strength
                            grad[j, 1] -= (dy / dist) * push_strength
            
            # L reduction forces
            # Move points that are on the boundary of the required hexagon towards the center.
            # The "boundary" is defined by the max of the 4 linear terms.
            # We can approximate the gradient of s_req.
            for k in range(N):
                x, y = centers[k]
                # If this point determines the max L, push it in.
                # Simple heuristic: push all points towards center slightly, but more if they are far.
                # Or calculate gradient of s_req.
                t1 = abs(x)
                t2 = (2.0 / SQRT3) * abs(y)
                t3 = abs(0.5 * x + (SQRT3 / 2.0) * y)
                t4 = abs(0.5 * x - (SQRT3 / 2.0) * y)
                max_val = max(t1, t2, t3, t4)
                
                # Gradient of s_req
                gx = 0.0
                gy = 0.0
                if t1 == max_val:
                    gx = np.sign(x)
                    gy = 0.0
                elif t2 == max_val:
                    gx = 0.0
                    gy = (2.0 / SQRT3) * np.sign(y)
                elif t3 == max_val:
                    gx = 0.5 * np.sign(0.5 * x + (SQRT3 / 2.0) * y)
                    gy = (SQRT3 / 2.0) * np.sign(0.5 * x + (SQRT3 / 2.0) * y)
                else:
                    gx = 0.5 * np.sign(0.5 * x - (SQRT3 / 2.0) * y)
                    gy = -(SQRT3 / 2.0) * np.sign(0.5 * x - (SQRT3 / 2.0) * y)
                
                # We want to minimize L, so we move against the gradient of s_req.
                # But we also need to satisfy overlaps.
                # Combine: grad_total = grad_overlap - lambda * grad_L
                # Let's use a fixed lambda that adapts or just balance them.
                # If there is overlap, prioritize overlap resolution.
                # If no overlap, prioritize L reduction.
                
                overlap_mag = np.linalg.norm(grad[k])
                if overlap_mag < 1e-6:
                    # No overlap, reduce L
                    grad[k, 0] -= gx * 0.5
                    grad[k, 1] -= gy * 0.5
                else:
                    # Has overlap, still try to reduce L a bit?
                    # Or just resolve overlap first.
                    # Let's just resolve overlap strongly.
                    pass
            
            # Update positions
            centers += step_size * grad
            
        # End local search
        
        # Evaluate final L and validity
        s_reqs = np.array([s_req(centers[k, 0], centers[k, 1]) for k in range(N)])
        final_L = np.max(s_reqs)
        
        # Check validity
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                if dx*dx + dy*dy < (SQRT3 - 1e-6)**2:
                    valid = False
                    break
            if not valid: break
        
        if valid and final_L < best_L:
            best_L = final_L
            best_positions = centers.copy()
            
    # Fallback if no valid solution found
    if best_positions is None:
        # Use the initial lattice, which is valid but has a large L
        best_positions = init_centers
        s_reqs = np.array([s_req(centers[k, 0], centers[k, 1]) for k in range(N)])
        best_L = np.max(s_reqs)
        
    inner = []
    for i in range(N):
        inner.append([best_positions[i, 0], best_positions[i, 1], 0.0])
        
    return inner, (0.0, 0.0), best_L, 0.0
