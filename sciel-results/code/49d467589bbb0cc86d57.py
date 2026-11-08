import math

def construct():
    N = 11  # This value is hardcoded in the seed, but the task says "n unit hexagons".
             # The seed has N=11. The prompt says "inner rows ... of n unit hexagons".
             # The seed code has N=11. I should probably keep N=11 or make it generic?
             # The seed code defines N=11 globally. I will keep N=11 to match the seed's specific instance
             # unless the task implies N is variable. The task says "inner rows ... of n unit hexagons".
             # The seed returns a list of N items. I will assume N is fixed at 11 for this specific evaluation
             # or I should check if N is a parameter. The signature is construct() -> ...
             # The seed uses global N=11. I will stick to N=11.
    
    # Generate hex grid points
    # We need enough points to cover N
    # Estimate max ring k
    k_est = int(math.ceil((math.sqrt(1 + 12*N) - 1) / 6))
    
    points = []
    # Iterate through hex distance rings
    for q in range(-k_est, k_est + 1):
        for r in range(-k_est, k_est + 1):
            # Hex distance in axial coords is max(|q|, |r|, |q+r|)? No.
            # Distance from (0,0) in axial is (|q| + |r| + |q+r|) / 2
            dist_hex = (abs(q) + abs(r) + abs(q + r)) // 2
            if dist_hex > k_est:
                continue
            # Convert to Cartesian
            # x = sqrt(3) * (q + r/2)
            # y = 1.5 * r
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            dist_sq = x*x + y*y
            points.append((dist_sq, x, y))
            
    # Sort by distance
    points.sort(key=lambda p: p[0])
    
    # Take first N
    selected = points[:N]
    
    # Calculate max distance
    max_dist_sq = max(p[0] for p in selected)
    max_dist = math.sqrt(max_dist_sq)
    
    # Calculate L
    # L = (2/sqrt(3)) * max_dist + 1
    L = (2.0 / math.sqrt(3)) * max_dist + 1.0
    
    # Format output
    inner = [[x, y, 0.0] for (_, x, y) in selected]
    
    return inner, (0.0, 0.0), L, 0.0
