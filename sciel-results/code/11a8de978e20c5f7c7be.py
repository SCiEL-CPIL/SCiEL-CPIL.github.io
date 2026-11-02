"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    def hexagon_contains(s, cx, cy, theta_deg, x, y):
        """Check if point (x,y) is inside regular hexagon of side s centered at (cx,cy) rotated by theta_deg."""
        dx = x - cx
        dy = y - cy
        theta = math.radians(theta_deg)
        xr = dx * math.cos(-theta) - dy * math.sin(-theta)
        yr = dx * math.sin(-theta) + dy * math.cos(-theta)
        
        inradius = s * math.sqrt(3) / 2.0
        for edge_angle_deg in [30, 90, 150, 210, 270, 330]:
            a = math.radians(edge_angle_deg)
            val = xr * math.cos(a) + yr * math.sin(a)
            if val > inradius + 1e-9:
                return False
        return True
    
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = float('inf')
    best_angle = 0.0
    
    # Try hexagonal grid arrangements with different radii
    for r_max in range(0, 20):
        if time.time() > deadline:
            break
        
        # Generate all positions in hexagonal cluster of radius r_max
        positions = []
        for q in range(-r_max, r_max + 1):
            for rr in range(-r_max, r_max + 1):
                if abs(q + rr) <= r_max:
                    x = math.sqrt(3) * (q + rr / 2.0)
                    y = 1.5 * rr
                    positions.append((x, y))
        
        if len(positions) < N:
            continue
        
        # Sort by distance from center, pick N closest
        positions_sorted = sorted(positions, key=lambda p: p[0]**2 + p[1]**2)
        selected = positions_sorted[:N]
        
        xs = [p[0] for p in selected]
        ys = [p[1] for p in selected]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        
        # Binary search on outer side length S
        lo, hi = 0.0, 100.0
        for _ in range(50):
            if time.time() > deadline:
                break
            mid = (lo + hi) / 2.0
            s_inner = mid - 1.0
            if s_inner <= 0:
                lo = mid
                continue
            all_inside = True
            for (px, py) in selected:
                if not hexagon_contains(s_inner, cx, cy, 0.0, px, py):
                    all_inside = False
                    break
            if all_inside:
                hi = mid
            else:
                lo = mid
        
        S_candidate = hi
        
        if S_candidate < best_side:
            best_side = S_candidate
            best_center = (cx, cy)
            best_inner = [[px, py, 0.0] for (px, py) in selected]
            best_angle = 0.0
    
    if best_inner is None:
        inner = [[math.sqrt(3) * i, 0.0, 0.0] for i in range(N)]
        cx = math.sqrt(3) * (N - 1) / 2.0
        cy = 0.0
        side = (N - 1) + 2.0 / math.sqrt(3)
        return inner, (cx, cy), side, 0.0
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
