"""Packing 11 unit hexagons into a minimal outer hexagon."""
import math
import time
import numpy as np

N = 11
T_START = time.time()
DEADLINE = 140.0

def hex_vertices(cx, cy, r, ang_deg):
    a = math.radians(ang_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    return np.array(verts)

def compute_min_S(inner):
    """Minimum side length S of centered, axis-aligned outer hex containing all inner hexes."""
    apothem_needed = 0.0
    for (x, y, ang) in inner:
        verts = hex_vertices(x, y, 1.0, ang)
        for v in verts:
            max_proj = 0.0
            for k in range(6):
                theta = k * math.pi / 3.0
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = abs(v[0] * nx + v[1] * ny)
                if proj > max_proj:
                    max_proj = proj
            if max_proj > apothem_needed:
                apothem_needed = max_proj
    return apothem_needed * 2.0 / math.sqrt(3)

def construct():
    t0 = time.time()
    
    # Initial: alternating orientation tight row, spacing 1.5
    inner = []
    for i in range(N):
        x = (i - (N-1)/2.0) * 1.5
        y = 0.0
        ang = 0.0 if i % 2 == 0 else 30.0
        inner.append([x, y, ang])
    
    best_S = compute_min_S(inner)
    best_inner = [list(x) for x in inner]
    
    # Local search: try to improve
    if time.time() - t0 < DEADLINE - 10:
        np.random.seed(123)
        current_S = best_S
        current_inner = [list(x) for x in best_inner]
        step = 0.02
        
        for it in range(10000):
            if time.time() - t0 > DEADLINE - 3:
                break
            # Random perturbation
            dx = np.random.uniform(-step, step, 3*N)
            new_inner = []
            for i in range(N):
                new_inner.append([
                    current_inner[i][0] + dx[3*i],
                    current_inner[i][1] + dx[3*i+1],
                    current_inner[i][2] + dx[3*i+2]
                ])
            S = compute_min_S(new_inner)
            if S < current_S - 1e-13:
                current_S = S
                current_inner = new_inner
                if S < best_S:
                    best_S = S
                    best_inner = [list(x) for x in current_inner]
                step = min(step * 1.05, 0.3)
            else:
                step = max(step * 0.98, 0.001)
    
    return best_inner, (0.0, 0.0), best_S, 0.0

if __name__ == "__main__":
    inner, center, S, angle = construct()
    print(f"S = {S:.8f}")
