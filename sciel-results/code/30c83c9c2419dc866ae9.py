import math
import time

N = 12

def construct():
    # Start time
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Static packing for N=12
    # We use a subset of the 19-hexagon cluster (side 3) which fits in L=4.
    # The 19-hex cluster centers are at axial coordinates (q, r) such that max(|q|, |r|, |q+r|) <= 2.
    # We select 12 of these centers.
    
    # Generate all 19 centers
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert axial to cartesian
                # x = sqrt(3) * (q + r/2)
                # y = 1.5 * r
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                centers.append((x, y))
                
    # We have 19 centers. We need 12.
    # To minimize the bounding box (and thus L), we should pick the most central ones.
    # Sort by distance from origin
    centers.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first 12
    selected_centers = centers[:12]
    
    # Scale to ensure no overlap (distance > sqrt(3))
    scale = 1.001
    
    # Rotate by 30 degrees to align "flat" side centers with outer hex vertices
    # This allows L=4 to work (eroded hex vertex dist = L-1 = 3)
    angle_rad = math.pi / 6  # 30 degrees
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    rotated_centers = []
    for x, y in selected_centers:
        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a
        x_new *= scale
        y_new *= scale
        rotated_centers.append((x_new, y_new))
        
    # Construct the inner list
    # Format: [x, y, angle_degrees]
    inner = []
    for x, y in rotated_centers:
        inner.append([x, y, 0.0])
        
    # The outer hexagon center is (0,0)
    # Side length L.
    # Max distance of centers is approx 3 * scale = 3.003
    # Eroded hex vertex distance is L - 1.
    # We need L - 1 >= 3.003 => L >= 4.003
    # Let's use 4.01 to be safe.
    
    outer_center = (0.0, 0.0)
    side = 4.01
    angle = 0.0
    
    return inner, outer_center, side, angle

# EVOLVE-BLOCK-END
