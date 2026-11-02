import math
import random
import time

N = 12

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Set random seed for reproducibility
    random.seed(42)
    
    best_s = float('inf')
    best_inner = None
    
    # Function to calculate minimal side length S for a set of centers
    # Centers must be inside hexagon of side L = S-1
    # Condition for (x,y) to be in hexagon of side L centered at (0,0):
    # max(|x|, (2/sqrt(3))*|y| + 0.5*|x|) <= L
    def get_min_side(centers):
        max_l = 0.0
        for x, y in centers:
            l1 = abs(x)
            l2 = (2.0 / math.sqrt(3)) * abs(y) + 0.5 * abs(x)
            max_l = max(max_l, l1, l2)
        return max_l + 1.0

    # Function to check if any pair overlaps
    # Min distance for side 1 hexagons is sqrt(3)
    MIN_DIST = math.sqrt(3)
    def check_overlap(centers):
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                if dx*dx + dy*dy < MIN_DIST*MIN_DIST * 0.999: # Small tolerance for "touching"
                    return True
        return False

    # Generate lattice points with a safety expansion factor
    # Base lattice for side 1 hexagons (touching) has spacing sqrt(3)
    # We expand by 1.02 to ensure strict disjointness
    SCALE = 1.02
    
    # Try different rotations of the lattice
    for angle_deg in [0, 15, 30, 45, 60, 75, 90]:
        if time.time() > deadline:
            break
            
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Generate a grid of lattice points
        # Triangular lattice vectors (scaled)
        u = (math.sqrt(3) * SCALE, 0)
        v = (math.sqrt(3)/2 * SCALE, 1.5 * SCALE)
        
        points = []
        for i in range(-6, 7):
            for j in range(-6, 7):
                # x = i*u_x + j*v_x
                # y = i*u_y + j*v_y
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                
                # Rotate
                xr = x * cos_a - y * sin_a
                yr = x * sin_a + y * cos_a
                
                points.append((xr, yr))
        
        # Sort points by distance from origin
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        # Take the first 12
        centers = [list(p) for p in points[:12]]
        
        # Verify no overlap (should be true with SCALE > 1)
        if check_overlap(centers):
            continue
            
        s = get_min_side(centers)
        
        if s < best_s:
            best_s = s
            best_inner = [[x, y, 0.0] for x, y in centers]
            
    # If best_inner is None (should not happen), fallback to a valid spread
    if best_inner is None:
        # Spread them out on a line, far apart
        # Distance between centers > sqrt(3)
        step = math.sqrt(3) * 1.1
        best_inner = [[k * step, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_s = get_min_side([(p[0], p[1]) for p in best_inner])

    # Simple Local Optimization (Gradient Descent / Nelder-Mead lite)
    # Try to shrink the bounding box by moving points slightly inward
    if best_inner is not None and time.time() < deadline - 5.0:
        current_centers = [[p[0], p[1]] for p in best_inner]
        current_s = best_s
        
        # Simple random perturbation + acceptance if valid and better
        # This is a very basic hill climbing
        for _ in range(1000):
            if time.time() > deadline:
                break
                
            # Pick a random point and perturb it
            idx = random.randint(0, N-1)
            dx = random.uniform(-0.1, 0.1)
            dy = random.uniform(-0.1, 0.1)
            
            new_centers = [list(c) for c in current_centers]
            new_centers[idx][0] += dx
            new_centers[idx][1] += dy
            
            # Check overlap
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d2 = (new_centers[i][0]-new_centers[j][0])**2 + (new_centers[i][1]-new_centers[j][1])**2
                    if d2 < MIN_DIST*MIN_DIST * 0.999:
                        valid = False
                        break
                if not valid: break
            
            if not valid:
                continue
                
            new_s = get_min_side(new_centers)
            
            if new_s < current_s:
                current_s = new_s
                current_centers = new_centers
                if current_s < best_s:
                    best_s = current_s
                    best_inner = [[x, y, 0.0] for x, y in current_centers]
                    break # Restart local search from new best? Or just continue?
                    # Let's just update best and continue loop to find more improvements

    return best_inner, (0.0, 0.0), best_s, 0.0
