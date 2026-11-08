import numpy as np
import math
import time

N = 11

def get_hex_centers_shape_A():
    # 7-hex flower
    centers = [(0.0, 0.0)]
    # 6 neighbors
    for i in range(6):
        angle = i * math.pi / 3
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        centers.append((x, y))
    return centers

def get_hex_centers_shape_B():
    # 7-hex flower + 4 ring 2 hexes at 30, 90, 150, 210
    centers = get_hex_centers_shape_A()
    # Ring 2 "close" ones are at 30, 90, 150, 210, 270, 330
    # Let's add 4 of them
    angles = [30, 90, 150, 210]
    for deg in angles:
        rad = math.radians(deg)
        # Distance is 3
        x = 3 * math.cos(rad)
        y = 3 * math.sin(rad)
        centers.append((x, y))
    return centers

def get_hex_centers_shape_C():
    # 7-hex flower + 4 ring 2 hexes at 0, 60, 120, 180 (the far ones)
    centers = get_hex_centers_shape_A()
    angles = [0, 60, 120, 180]
    for deg in angles:
        rad = math.radians(deg)
        # Distance is 2*sqrt(3)
        x = 2 * math.sqrt(3) * math.cos(rad)
        y = 2 * math.sqrt(3) * math.sin(rad)
        centers.append((x, y))
    return centers

def support_hex(theta):
    # Support function of unit hexagon, flat top/bottom, vertices at 0, 60...
    # h(theta) = sqrt(3)/2 * sec(theta mod pi/3 - pi/6)
    # Careful with domain
    t = theta % (math.pi / 3)
    if t > math.pi / 6:
        t = t - math.pi / 3
    return (math.sqrt(3) / 2) / math.cos(t)

def compute_min_side(centers):
    # Sample theta
    thetas = np.linspace(0, 2 * math.pi, 720, endpoint=False)
    # Precompute center vectors
    cx = [c[0] for c in centers]
    cy = [c[1] for c in centers]
    
    best_s = float('inf')
    best_alpha = 0
    
    # Optimize alpha
    # The function S(alpha) = max_theta h_union(theta) / h_hex(theta - alpha)
    # is periodic with period pi/3?
    # h_hex is periodic with pi/3.
    # h_union is not necessarily, but the cluster has 6-fold symmetry?
    # Shape A has 6-fold symmetry.
    # Shape B (30, 90, 150, 210) has 2-fold symmetry?
    # Let's just scan alpha from 0 to pi/3.
    
    alphas = np.linspace(0, math.pi / 3, 36, endpoint=False)
    
    for alpha in alphas:
        max_ratio = 0
        for theta in thetas:
            # h_union(theta)
            # h_hex(theta) is constant for all inner hexes if they are aligned
            h_hex_val = support_hex(theta)
            max_proj = -float('inf')
            for x, y in zip(cx, cy):
                # Projection of center onto normal theta
                proj = x * math.cos(theta) + y * math.sin(theta)
                val = h_hex_val + proj
                if val > max_proj:
                    max_proj = val
            
            h_outer_val = support_hex(theta - alpha)
            if h_outer_val < 1e-9:
                continue
            ratio = max_proj / h_outer_val
            if ratio > max_ratio:
                max_ratio = ratio
        
        if max_ratio < best_s:
            best_s = max_ratio
            best_alpha = alpha
            
    return best_s, best_alpha

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Try different shapes
    shapes = {
        'A': get_hex_centers_shape_A(),
        'B': get_hex_centers_shape_B(),
        'C': get_hex_centers_shape_C(),
    }
    
    best_result = None
    best_score = float('inf')
    
    for name, centers in shapes.items():
        if time.time() - start_time > 100:
            break
        s, alpha = compute_min_side(centers)
        if s < best_score:
            best_score = s
            best_result = (centers, alpha)
            
    if best_result is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    centers, alpha = best_result
    
    # Construct inner list
    # The problem asks for inner rows (x, y, angle_degrees)
    # We assumed inner hexagons are aligned (angle 0)
    # The outer hexagon is rotated by alpha
    # Wait, the return signature is:
    # inner, center, side, angle
    # inner: rows of (x, y, angle_degrees)
    # center: outer hexagon center
    # side: outer hexagon side length
    # angle: outer hexagon rotation
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    # Outer hexagon center is (0,0)
    # Side is best_score
    # Angle is alpha (radians) -> degrees
    
    return inner, (0.0, 0.0), best_score, math.degrees(alpha)
# EVOLVE-BLOCK-END
