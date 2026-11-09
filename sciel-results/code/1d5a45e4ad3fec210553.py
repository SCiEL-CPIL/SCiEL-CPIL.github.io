import math
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    def min_L_for_point(px, py, phi):
        """Minimum L such that point (px,py) is inside hexagon centered at origin with rotation phi."""
        # Rotate point by -phi
        cos_p = math.cos(-phi)
        sin_p = math.sin(-phi)
        rx = px * cos_p - py * sin_p
        ry = px * sin_p + py * cos_p
        
        # For flat-top hexagon (vertices at 0,60,120,180,240,300), the 6 support directions are at 0,60,120,180,240,300
        # A point is inside if |rx*cos(a) + ry*sin(a)| <= L*sqrt(3)/2 for a in {0,60,120,180,240,300}
        # So L >= 2/sqrt(3) * max(|rx*cos(a) + ry*sin(a)|)
        max_val = 0.0
        for k in range(6):
            a = k * math.pi / 3.0
            val = abs(rx * math.cos(a) + ry * math.sin(a))
            max_val = max(max_val, val)
        return (2.0 / sqrt3) * max_val
    
    def min_L_for_arrangement(centers, angles, phi):
        """Compute minimum L such that all inner hexagons fit inside outer hexagon centered at origin."""
        max_L = 0.0
        for (ix, iy), ang in zip(centers, angles):
            # Get vertices of this inner unit hexagon
            for k in range(6):
                theta = ang + k * math.pi / 3.0
                vx = ix + math.cos(theta)
                vy = iy + math.sin(theta)
                L_needed = min_L_for_point(vx, vy, phi)
                max_L = max(max_L, L_needed)
        return max_L
    
    # Generate candidate arrangements
    candidates = []
    
    # Candidate 1: 4 rows of 3 (r from -2 to 1, q from -1 to 1)
    arr1 = [(q, r) for r in range(-2, 2) for q in range(-1, 2)]
    candidates.append(arr1)
    
    # Candidate 2: 3 rows of 4 (r from -1 to 1, q from -2 to 1)
    arr2 = [(q, r) for r in range(-1, 2) for q in range(-2, 2)]
    candidates.append(arr2)
    
    # Candidate 3: hexagonal cluster subset
    arr3 = [(q, r) for q in range(-2, 3) for r in range(-2, 3) if max(abs(q), abs(r), abs(q+r)) <= 2][:12]
    candidates.append(arr3)
    
    # Candidate 4: more compact - 2x6 rectangle
    arr4 = [(q, r) for r in range(-2, 2) for q in range(-3, 3)][:12]
    candidates.append(arr4)
    
    # Candidate 5: 3x4 with offset
    arr5 = []
    for r in range(-2, 2):
        for q in range(-2, 2):
            arr5.append((q, r))
    candidates.append(arr5[:12])
    
    best_L = float('inf')
    best_result = None
    
    for arr in candidates:
        if len(arr) != N:
            continue
        
        # Convert to cartesian
        centers = []
        for q, r in arr:
            x, y = axial_to_cartesian(q, r)
            centers.append((x, y))
        
        # Compute centroid
        cx = sum(c[0] for c in centers) / N
        cy = sum(c[1] for c in centers) / N
        
        # Translate so centroid is at origin
        centers = [(x - cx, y - cy) for x, y in centers]
        
        # Try different rotations
        for phi_deg in range(0, 60, 10):
            phi = math.radians(phi_deg)
            
            # All inner hexagons aligned
            angles = [0.0] * N
            L = min_L_for_arrangement(centers, angles, phi)
            if L < best_L:
                best_L = L
                best_result = (centers, angles, phi)
        
        # Try rotating inner hexagons
        for phi_deg in range(0, 60, 10):
            phi = math.radians(phi_deg)
            for inner_rot_deg in [0, 30]:
                inner_rot = math.radians(inner_rot_deg)
                angles = [inner_rot] * N
                L = min_L_for_arrangement(centers, angles, phi)
                if L < best_L:
                    best_L = L
                    best_result = (centers, angles, phi)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Check deadline
    if time.time() > deadline or best_result is None:
        # Return a safe default - linear arrangement
        inner = [[2.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 25.0, 0.0
    
    centers, angles, phi = best_result
    
    # Build the inner list
    inner = []
    for (x, y), ang in zip(centers, angles):
        inner.append([x, y, math.degrees(ang)])
    
    return inner, (0.0, 0.0), best_L, math.degrees(phi)
