"""Packing 11 unit hexagons inside a regular hexagon."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: return quickly for this direct construction
    t0 = time.time()
    
    s3 = math.sqrt(3)
    
    # "Flower" packing: 1 center + 6 inner ring + 4 outer ring
    # Inner hexagons all at angle 0 (aligned)
    inner = []
    
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # 6 inner ring at distance sqrt(3) from center, angles 0, 60, 120, 180, 240, 300
    for k in range(6):
        ang = math.radians(60.0 * k)
        x = s3 * math.cos(ang)
        y = s3 * math.sin(ang)
        inner.append([x, y, 0.0])
    
    # 4 outer ring at distance 3, angles 30, 150, 210, 330
    for k in range(4):
        ang = math.radians(30.0 + 120.0 * k)
        x = 3.0 * math.cos(ang)
        y = 3.0 * math.sin(ang)
        inner.append([x, y, 0.0])
    
    center = (0.0, 0.0)
    
    # Compute the minimal enclosing regular hexagon side length
    # A regular hexagon with side length S, center (cx, cy), rotation theta has:
    # - Vertices at distance S from center, at angles theta + 60*k
    # - Sides at distance S*sqrt(3)/2 from center, with normals at angles theta + 30 + 60*k
    # A point (px, py) is inside if for all 6 directions:
    #   (px-cx)*cos(n) + (py-cy)*sin(n) <= S*sqrt(3)/2
    #   where n is the outward normal angle for each side
    
    # The cluster has 6-fold symmetry (approximately, since the 4 outer hexagons break it to 4-fold)
    # Actually the 4 outer hexagons at 30, 150, 210, 330 have 4-fold symmetry (90 deg rotations)
    # But combined with the 6-fold inner, the full cluster has only 2-fold symmetry
    
    # Let's compute the required S by checking all points against all 6 sides
    # for a given rotation of the container.
    
    # For a regular hexagon container with side S and rotation theta:
    # The 6 side normals are at angles theta + 30 + 60*k for k=0..5
    # Constraint: max over all points p of (p - c) . n_k <= S * sqrt(3)/2 for all k
    # So S >= max_k [max_p (p-c).n_k] / (sqrt(3)/2) = 2/sqrt(3) * max_k max_p (p-c).n_k
    
    # We want to minimize over theta.
    
    # Get all vertices of all inner hexagons
    # For a unit hexagon at center (hx, hy) with rotation phi (degrees):
    # vertices are at (hx + cos(phi + 60*j), hy + sin(phi + 60*j)) for j=0..5
    # (vertices at angles phi, phi+60, ..., phi+300)
    
    all_points = []
    for (hx, hy, phi_deg) in inner:
        phi = math.radians(phi_deg)
        for j in range(6):
            vx = hx + math.cos(phi + math.radians(60.0 * j))
            vy = hy + math.sin(phi + math.radians(60.0 * j))
            all_points.append((vx, vy))
    
    # Search over container rotation theta
    # Due to symmetry, we can restrict to a small range
    # The cluster has symmetry: if we rotate the cluster by 60 deg, the inner 7 hexagons map to themselves
    # but the 4 outer ones don't (they go to different positions). Actually the 4 outer at 30,150,210,330
    # rotated by 60 become 90, 210, 270, 30 - wait, 30+60=90, 150+60=210, 210+60=270, 330+60=30+360=30
    # So {30,150,210,330} -> {90,210,270,30} = {30,90,210,270} which is different.
    # So the cluster only has 2-fold symmetry (180 deg rotation).
    
    # Let's try a range of theta values. The optimal theta should be where the 
    # projections are balanced.
    
    best_S = float('inf')
    best_theta = 0.0
    
    # Try theta from 0 to 30 degrees in fine steps (by symmetry of hexagon, period is 60 deg)
    # Actually the hexagon container has 6-fold symmetry, so theta mod 60 suffices.
    # But the cluster breaks some of that symmetry.
    
    # Let's sample theta in [0, 30] degrees finely
    n_samples = 1801
    for i in range(n_samples):
        theta_deg = 30.0 * i / (n_samples - 1)
        theta = math.radians(theta_deg)
        
        max_proj = 0.0
        for k in range(6):
            n_ang = theta + math.radians(30.0 + 60.0 * k)
            nx = math.cos(n_ang)
            ny = math.sin(n_ang)
            
            for (px, py) in all_points:
                proj = px * nx + py * ny
                if proj > max_proj:
                    max_proj = proj
        
        S = max_proj / (s3 / 2.0)
        if S < best_S:
            best_S = S
            best_theta = theta_deg
    
    # Refine around best_theta with finer search
    if 0 < best_theta < 30:
        lo = max(0, best_theta - 0.1)
        hi = min(30, best_theta + 0.1)
        n_refine = 1001
        for i in range(n_refine):
            theta_deg = lo + (hi - lo) * i / (n_refine - 1)
            theta = math.radians(theta_deg)
            
            max_proj = 0.0
            for k in range(6):
                n_ang = theta + math.radians(30.0 + 60.0 * k)
                nx = math.cos(n_ang)
                ny = math.sin(n_ang)
                
                for (px, py) in all_points:
                    proj = px * nx + py * ny
                    if proj > max_proj:
                        max_proj = proj
            
            S = max_proj / (s3 / 2.0)
            if S < best_S:
                best_S = S
                best_theta = theta_deg
    
    # Add a tiny safety margin
    side = best_S + 1e-9
    
    return inner, center, side, best_theta
# EVOLVE-BLOCK-END
