"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # Ring structure: 1 center, 5 inner ring, 6 outer ring
    # Inner ring: 5 hexagons at angles 72° apart, radius r1
    # Outer ring: 6 hexagons at angles 60° apart, radius r2
    
    # Support function of a unit regular hexagon (side=1) with rotation alpha in direction theta
    def hex_support(theta, alpha=0.0):
        # Max of cos(theta - alpha - k*pi/3) for k=0..5
        angles = alpha + np.arange(6) * np.pi / 3.0
        return np.max(np.cos(theta - angles))
    
    # Compute the required outer side length for given parameters
    def compute_side(r1, r2, global_rot):
        # Center hexagon at origin
        # Inner ring: 5 hexagons at radius r1, angles offset by global_rot + k*72°
        # Outer ring: 6 hexagons at radius r2, angles offset by global_rot + k*60° + 30° (to interleave)
        
        # Positions and rotations of all 12 hexagons
        # All inner hexagons have the same orientation (global_rot)
        
        # Sample many directions
        thetas = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        
        # Cluster support in each direction
        # For each hexagon: cx*cos(theta) + cy*sin(theta) + hex_support(theta, hex_rot)
        
        # Center hexagon
        sup = hex_support(thetas, global_rot)
        
        # Inner ring: 5 hexagons
        for k in range(5):
            ang = global_rot + k * 2 * np.pi / 5.0
            cx = r1 * np.cos(ang)
            cy = r1 * np.sin(ang)
            sup = np.maximum(sup, cx * np.cos(thetas) + cy * np.sin(thetas) + hex_support(thetas, global_rot))
        
        # Outer ring: 6 hexagons
        for k in range(6):
            ang = global_rot + k * 2 * np.pi / 6.0 + np.pi / 6.0  # offset by 30° to interleave
            cx = r2 * np.cos(ang)
            cy = r2 * np.sin(ang)
            sup = np.maximum(sup, cx * np.cos(thetas) + cy * np.sin(thetas) + hex_support(thetas, global_rot))
        
        # Outer hexagon support: s * hex_support(theta, beta)
        # We can choose beta (outer rotation) optimally
        # s = max_theta sup(theta) / hex_support(theta, beta)
        # To minimize s, we want to maximize hex_support(theta, beta) where sup is large
        # Try several beta values
        best_s = np.inf
        for beta in np.linspace(0, np.pi/3, 13):
            outer_sup = hex_support(thetas, beta)
            # Avoid division by zero (hex_support is always >= 0.5 for regular hex)
            ratio = sup / np.maximum(outer_sup, 1e-10)
            s = np.max(ratio)
            if s < best_s:
                best_s = s
        
        return best_s
    
    # Check feasibility: do the hexagons overlap?
    def check_overlap(r1, r2, global_rot):
        # Two unit hexagons with the same orientation at distance d apart
        # They overlap if d < 2 * inradius = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732
        # Actually, two hexagons with same orientation: minimum distance to not overlap is 2*apothem = sqrt(3)
        # Wait, apothem of unit hex = sqrt(3)/2, so two side-by-side need distance >= 2*sqrt(3)/2 = sqrt(3)
        
        # Get all positions
        positions = [(0.0, 0.0)]
        for k in range(5):
            ang = global_rot + k * 2 * np.pi / 5.0
            positions.append((r1 * np.cos(ang), r1 * np.sin(ang)))
        for k in range(6):
            ang = global_rot + k * 2 * np.pi / 6.0 + np.pi / 6.0
            positions.append((r2 * np.cos(ang), r2 * np.sin(ang)))
        
        # Check all pairs
        min_dist = np.inf
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                d = np.sqrt(dx*dx + dy*dy)
                min_dist = min(min_dist, d)
        
        # Minimum distance for non-overlapping unit hexagons with same orientation
        # The "width" of a hexagon in any direction is 2*apothem = sqrt(3) for the direction perpendicular to a side
        # But for same orientation, the critical distance is 2 * apothem = sqrt(3)
        # Actually, let me think more carefully. Two identical regular hexagons with the same orientation:
        # The minimum translation to separate them is 2 * apothem in the direction perpendicular to a side.
        # But if they're offset at an angle, the required distance depends on the direction.
        # For safety, use 2 * apothem = sqrt(3) as minimum distance in any direction.
        # Actually that's not right either. Let me use a more conservative bound.
        # The inradius is sqrt(3)/2. Two hexagons don't overlap if the distance between centers
        # is at least 2 * inradius = sqrt(3) in the worst case (when the line connecting centers
        # is perpendicular to a side).
        
        threshold = np.sqrt(3) - 0.01  # small tolerance
        return min_dist >= threshold
    
    # Optimize r1, r2, global_rot
    best_side = np.inf
    best_params = (1.75, 3.5, 0.0)
    
    # Grid search over r1, r2, global_rot
    # r1 should be around sqrt(3) ≈ 1.732 (just enough to not overlap with center)
    # r2 should be larger
    
    r1_vals = np.linspace(1.73, 2.2, 20)
    r2_vals = np.linspace(3.0, 4.5, 20)
    rot_vals = np.linspace(0, np.pi/3, 13)
    
    for r1 in r1_vals:
        if time.time() > deadline:
            break
        for r2 in r2_vals:
            if time.time() > deadline:
                break
            # Check overlap quickly
            if not check_overlap(r1, r2, 0.0):
                continue
            for rot in rot_vals:
                if time.time() > deadline:
                    break
                s = compute_side(r1, r2, rot)
                if s < best_side:
                    best_side = s
                    best_params = (r1, r2, rot)
    
    # Local refinement around best
    r1_best, r2_best, rot_best = best_params
    
    for step in [0.05, 0.02, 0.01, 0.005]:
        if time.time() > deadline:
            break
        improved = True
        while improved:
            improved = False
            for dr1 in [-step, step]:
                for dr2 in [-step, step]:
                    for drot in [-step, step]:
                        nr1 = r1_best + dr1
                        nr2 = r2_best + dr2
                        nrot = rot_best + drot
                        if nr1 < 1.7 or nr2 < 2.5 or nrot < -0.5 or nrot > 2.0:
                            continue
                        if not check_overlap(nr1, nr2, nrot):
                            continue
                        s = compute_side(nr1, nr2, nrot)
                        if s < best_side - 1e-6:
                            best_side = s
                            best_params = (nr1, nr2, nrot)
                            r1_best, r2_best, rot_best = nr1, nr2, nrot
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    r1_f, r2_f, rot_f = best_params
    
    # Build the inner hexagons
    inner = []
    # Center
    inner.append([0.0, 0.0, np.degrees(rot_f)])
    # Inner ring
    for k in range(5):
        ang = rot_f + k * 2 * np.pi / 5.0
        inner.append([r1_f * np.cos(ang), r1_f * np.sin(ang), np.degrees(rot_f)])
    # Outer ring
    for k in range(6):
        ang = rot_f + k * 2 * np.pi / 6.0 + np.pi / 6.0
        inner.append([r2_f * np.cos(ang), r2_f * np.sin(ang), np.degrees(rot_f)])
    
    # Outer hexagon: center at origin, side = best_side, rotation = optimal beta
    # Find optimal beta
    thetas = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    
    def hex_support(theta, alpha=0.0):
        angles = alpha + np.arange(6) * np.pi / 3.0
        return np.max(np.cos(theta - angles))
    
    sup = hex_support(thetas, rot_f)
    for k in range(5):
        ang = rot_f + k * 2 * np.pi / 5.0
        cx = r1_f * np.cos(ang)
        cy = r1_f * np.sin(ang)
        sup = np.maximum(sup, cx * np.cos(thetas) + cy * np.sin(thetas) + hex_support(thetas, rot_f))
    for k in range(6):
        ang = rot_f + k * 2 * np.pi / 6.0 + np.pi / 6.0
        cx = r2_f * np.cos(ang)
        cy = r2_f * np.sin(ang)
        sup = np.maximum(sup, cx * np.cos(thetas) + cy * np.sin(thetas) + hex_support(thetas, rot_f))
    
    best_s = np.inf
    best_beta = 0.0
    for beta in np.linspace(0, np.pi/3, 37):
        outer_sup = hex_support(thetas, beta)
        ratio = sup / np.maximum(outer_sup, 1e-10)
        s = np.max(ratio)
        if s < best_s:
            best_s = s
            best_beta = beta
    
    # Add small margin for safety
    side = best_s + 0.01
    outer_rot = np.degrees(best_beta)
    
    return inner, (0.0, 0.0), side, outer_rot
# EVOLVE-BLOCK-END
