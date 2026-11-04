import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # We search for the minimum side length S of the outer hexagon
    # such that 12 unit hexagons (aligned, on a hex grid) fit inside.
    
    # Lower bound from area: S >= sqrt(12) ~ 3.46
    # Upper bound: S=5 is safe (fits 19 hexagons)
    
    best_side = None
    best_inner = None
    
    # Iterate S from 3.47 to 5.0
    # Step 0.01 is fine for 120s limit
    S = 3.47
    while S < 5.0:
        inner = find_packing(S)
        if inner is not None:
            best_side = S
            best_inner = inner
            break
        S += 0.01
        
    if best_side is None:
        # Fallback to a known valid packing (e.g. S=5)
        best_side = 5.0
        best_inner = find_packing(5.0)
        
    return best_inner, (0.0, 0.0), best_side, 0.0

def is_point_in_hex(x, y, S):
    # Outer hexagon centered at (0,0), side S, flat-top
    # Inequalities:
    # |x| <= S
    # |x + sqrt(3)y| <= sqrt(3)S
    # |x - sqrt(3)y| <= sqrt(3)S
    if abs(x) > S:
        return False
    if abs(x + math.sqrt(3) * y) > math.sqrt(3) * S:
        return False
    if abs(x - math.sqrt(3) * y) > math.sqrt(3) * S:
        return False
    return True

def is_hex_in_hex(cx, cy, S):
    # Check if unit hexagon at (cx, cy) is inside outer hex of side S
    # Vertices of unit flat-top hexagon at origin:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    verts = [
        (1.0, 0.0),
        (0.5, math.sqrt(3)/2),
        (-0.5, math.sqrt(3)/2),
        (-1.0, 0.0),
        (-0.5, -math.sqrt(3)/2),
        (0.5, -math.sqrt(3)/2)
    ]
    
    for vx, vy in verts:
        px = cx + vx
        py = cy + vy
        if not is_point_in_hex(px, py, S):
            return False
    return True

def find_packing(S):
    # Generate hex grid candidates
    # Axial coords q, r
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    
    # Determine range
    # Max x ~ S, so 1.5*q ~ S => q ~ 2*S/3
    R = int(2.5 * S) + 2
    
    candidates = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            cx = 1.5 * q
            cy = math.sqrt(3) * (r + 0.5 * q)
            
            # Quick reject: if center is too far out
            # Approximate check
            if abs(cx) > S + 1 or abs(cy) > S * math.sqrt(3):
                continue
                
            if is_hex_in_hex(cx, cy, S):
                candidates.append((cx, cy, 0.0))
                
    if len(candidates) >= N:
        # Return first N candidates
        # To be deterministic and compact, we might want to sort them by distance from center
        # But any valid set works.
        # Sorting by distance ensures we pick the "central" ones first, which is good for packing density
        candidates.sort(key=lambda c: c[0]**2 + c[1]**2)
        return candidates[:N]
        
    return None
# EVOLVE-BLOCK-END
