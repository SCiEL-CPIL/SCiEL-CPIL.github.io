"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
PERIMETER = 4.0
START = time.time()
DEADLINE = START + 138.0


def _validity(circles, W, H):
    """Check disjoint interiors and bounds. Returns (ok, perimeter, sum_radii)."""
    n = len(circles)
    if n != N:
        return False, 0.0, 0.0
    minx = min(c[0] - c[2] for c in circles)
    miny = min(c[1] - c[2] for c in circles)
    maxx = max(c[0] + c[2] for c in circles)
    maxy = max(c[1] + c[2] for c in circles)
    bw = maxx - minx
    bh = maxy - miny
    per = 2.0 * (bw + bh)
    if per > PERIMETER + 1e-6:
        return False, per, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(circles[i][0] - circles[j][0], circles[i][1] - circles[j][1])
            if d < circles[i][2] + circles[j][2] - 1e-7:
                return False, per, 0.0
    s = sum(c[2] for c in circles)
    return True, per, s


def _scale_to_perimeter(circles, target_per=PERIMETER):
    """Scale the whole packing so bounding-box perimeter equals target_per."""
    minx = min(c[0] - c[2] for c in circles)
    miny = min(c[1] - c[2] for c in circles)
    maxx = max(c[0] + c[2] for c in circles)
    maxy = max(c[1] + c[2] for c in circles)
    bw = maxx - minx
    bh = maxy - miny
    per = 2.0 * (bw + bh)
    if per <= 0:
        return circles
    s = target_per / per
    out = []
    for (x, y, r) in circles:
        nx = (x - minx) * s
        ny = (y - miny) * s
        nr = r * s
        out.append((nx, ny, nr))
    return out

def _force_directed_pack(W, H, seed=0):
    """
    Pack N circles in a W x H rectangle using force-directed relaxation.
    Returns list of (x, y, r) or None if failed.
    """
    rng = random.Random(seed)
    
    # Initial configuration: Hexagonal lattice
    points = []
    # Estimate radius based on area
    r_est = np.sqrt(W * H / N) * 0.8
    r = r_est
    
    y = r
    row = 0
    while y < H - r:
        x = r
        if row % 2 == 1:
            x += r
        while x < W - r:
            points.append((x, y, r))
            x += 2 * r
        y += r * np.sqrt(3)
        row += 1
        
    # Adjust to exactly N points
    if len(points) > N:
        # Keep closest to center
        center_x, center_y = W/2, H/2
        points.sort(key=lambda p: (p[0]-center_x)**2 + (p[1]-center_y)**2)
        points = points[:N]
    elif len(points) < N:
        # Add random points
        while len(points) < N:
            points.append((rng.uniform(r, W-r), rng.uniform(r, H-r), r))
            
    circles = [list(p) for p in points]
    
    # Simulation parameters
    dt = 0.05
    max_iter = 500
    repulsion_strength = 5.0
    boundary_strength = 10.0
    growth_rate = 0.01 # How fast radii grow
    
    for step in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
            
        forces = [[0.0, 0.0] for _ in range(N)]
        
        # 1. Repulsion between circles
        for i in range(N):
            for j in range(i+1, N):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                d = np.hypot(dx, dy)
                min_d = circles[i][2] + circles[j][2]
                
                if d < 1e-6:
                    dx = 1e-3
                    dy = 1e-3
                    d = np.hypot(dx, dy)
                    
                if d < min_d:
                    # Force is proportional to overlap
                    overlap = min_d - d
                    f = repulsion_strength * overlap
                    fx = f * dx / d
                    fy = f * dy / d
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy
                    
        # 2. Boundary forces
        for i in range(N):
            x, y, r = circles[i]
            
            # Left wall
            if x < r:
                forces[i][0] += boundary_strength * (r - x)
            # Right wall
            if x > W - r:
                forces[i][0] -= boundary_strength * (x - (W - r))
            # Bottom wall
            if y < r:
                forces[i][1] += boundary_strength * (r - y)
            # Top wall
            if y > H - r:
                forces[i][1] -= boundary_strength * (y - (H - r))
                
        # 3. Apply forces (simple Euler integration)
        for i in range(N):
            circles[i][0] += forces[i][0] * dt
            circles[i][1] += forces[i][1] * dt
            
            # Clamp to bounds to ensure stability
            circles[i][0] = max(circles[i][2], min(W - circles[i][2], circles[i][0]))
            circles[i][1] = max(circles[i][2], min(H - circles[i][2], circles[i][1]))
            
        # 4. Grow radii slightly to maximize sum
        # Only grow if system is relatively stable (low force magnitude)
        total_force_mag = sum(np.hypot(f[0], f[1]) for f in forces)
        if total_force_mag < 1.0:
             for i in range(N):
                 circles[i][2] += growth_rate
             
    # Final relaxation to remove any remaining overlaps
    # Use a stricter solver-like approach for the last few steps
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                d = np.hypot(dx, dy)
                min_d = circles[i][2] + circles[j][2]
                if d < min_d - 1e-8:
                    if d < 1e-8:
                        dx = 1.0
                        dy = 0.0
                        d = 1.0
                    push = (min_d - d) / 2.0 + 1e-4
                    nx_i = circles[i][0] + (dx/d)*push
                    ny_i = circles[i][1] + (dy/d)*push
                    nx_j = circles[j][0] - (dx/d)*push
                    ny_j = circles[j][1] - (dy/d)*push
                    
                    if nx_i >= circles[i][2] and nx_i <= W-circles[i][2] and ny_i >= circles[i][2] and ny_i <= H-circles[i][2]:
                        circles[i][0] = nx_i
                        circles[i][1] = ny_i
                    if nx_j >= circles[j][2] and nx_j <= W-circles[j][2] and ny_j >= circles[j][2] and ny_j <= H-circles[j][2]:
                        circles[j][0] = nx_j
                        circles[j][1] = ny_j
                    changed = True
        if not changed:
            break
            
    return circles

def construct():
    global DEADLINE
    t0 = time.time()
    DEADLINE = t0 + 138.0
    
    best = None
    best_val = -1.0
    
    # Aspect ratio candidates: W + H = 2
    # Try a range of aspects
    aspect_choices = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.15, 0.85]
    random.shuffle(aspect_choices)
    
    for W in aspect_choices:
        if time.time() > DEADLINE - 10:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        circles = _force_directed_pack(W, H, seed=int(W * 100))
        if circles is not None:
            # Scale to exact perimeter 4
            circles_scaled = _scale_to_perimeter(circles, PERIMETER)
            ok, per, s = _validity(circles_scaled, 2.0, 2.0)
            if ok and s > best_val:
                best_val = s
                best = circles_scaled
    
    if best is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = cells[:N]
        best = _scale_to_perimeter(best, PERIMETER)
    
    return np.array(best)


if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(res[:3])
