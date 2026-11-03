import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def hex_vertices(cx, cy, angle_deg, side=1.0):
        angle_rad = math.radians(angle_deg)
        return np.array([(cx + side * math.cos(angle_rad + i * math.pi / 3),
                         cy + side * math.sin(angle_rad + i * math.pi / 3)) for i in range(6)])
    
    def min_side(inner):
        """Min side length of outer hexagon (centered at origin, angle 0) to contain all inner."""
        min_s = 0.0
        for (x, y, a) in inner:
            v = hex_vertices(x, y, a)
            for k in range(6):
                theta = k * math.pi / 3
                proj = np.abs(v[:, 0] * math.cos(theta) + v[:, 1] * math.sin(theta))
                s = proj.max() * 2.0 / math.sqrt(3)
                if s > min_s:
                    min_s = s
        return min_s
    
    def overlap_penalty(inner):
        """Penalty for overlapping hexagons."""
        penalty = 0.0
        verts = [hex_vertices(x, y, a) for x, y, a in inner]
        for i in range(N):
            for j in range(i + 1, N):
                v1, v2 = verts[i], verts[j]
                # SAT
                sep = False
                for v in (v1, v2):
                    for k in range(6):
                        k2 = (k + 1) % 6
                        ex = v[k2, 0] - v[k, 0]
                        ey = v[k2, 1] - v[k, 1]
                        nx, ny = -ey, ex
                        p1 = v1[:, 0] * nx + v1[:, 1] * ny
                        p2 = v2[:, 0] * nx + v2[:, 1] * ny
                        if p1.max() < p2.min() + 1e-9 or p2.max() < p1.min() + 1e-9:
                            sep = True
                            break
                    if sep:
                        break
                if not sep:
                    # Overlapping - estimate penetration
                    # Find min distance between vertices
                    min_d = float('inf')
                    for p in v1:
                        for q in v2:
                            d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                            if d < min_d:
                                min_d = d
                    min_d = math.sqrt(min_d)
                    # If overlapping, min_d is small; use a large penalty
                    penalty += 500.0
        return penalty
    
    def objective(x):
        if time.time() > deadline:
            return 1e6
        inner = [(x[3*i], x[3*i+1], x[3*i+2]) for i in range(N)]
        s = x[-1]
        
        # Containment
        pen = 0.0
        for (x, y, a) in inner:
            v = hex_vertices(x, y, a)
            for k in range(6):
                theta = k * math.pi / 3
                proj = np.abs(v[:, 0] * math.cos(theta) + v[:, 1] * math.sin(theta))
                inradius = s * math.sqrt(3) / 2
                if proj.max() > inradius:
                    pen += (proj.max() - inradius) ** 2
        
        # Overlap
        pen += overlap_penalty(inner)
        
        return s + pen
    
    # Multiple initial guesses
    d = math.sqrt(3)
    guesses = []
    
    # Guess 1: center + 6 + 5
    pos = [(0, 0)]
    for i in range(6):
        a = math.radians(i * 60)
        pos.append((d * math.cos(a), d * math.sin(a)))
    for i in range(5):
        a = math.radians(i * 60 + 30)
        pos.append((2 * d * math.cos(a), 2 * d * math.sin(a)))
    guesses.append(pos)
    
    # Guess 2: 3-4-3-2 rows
    pos = []
    # Row 1: 3 hexagons at y = 0
    for i in range(3):
        pos.append(((i - 1) * d, 0))
    # Row 2: 4 hexagons at y = d*sqrt(3)/2, offset
    for i in range(4):
        pos.append(((i - 1.5) * d, d * math.sqrt(3) / 2))
    # Row 3: 3 hexagons at y = d*sqrt(3)
    for i in range(3):
        pos.append(((i - 1) * d, d * math.sqrt(3)))
    # Row 4: 2 hexagons at y = d*sqrt(3)*1.5, offset
    for i in range(2):
        pos.append(((i - 0.5) * d, d * math.sqrt(3) * 1.5))
    guesses.append(pos)
    
    # Guess 3: 2-3-4-3 rows
    pos = []
    for i in range(2):
        pos.append(((i - 0.5) * d, 0))
    for i in range(3):
        pos.append(((i - 1) * d, d * math.sqrt(3) / 2))
    for i in range(4):
        pos.append(((i - 1.5) * d, d * math.sqrt(3)))
    for i in range(3):
        pos.append(((i - 1) * d, d * math.sqrt(3) * 1.5))
    guesses.append(pos)
    
    best_x = None
    best_obj = float('inf')
    
    for pos in guesses:
        if time.time() > deadline - 10:
            break
        inner_init = [[x, y, 0.0] for x, y in pos]
        s_init = min_side(inner_init)
        x0 = []
        for x, y, a in inner_init:
            x0.extend([x, y, a])
        x0.append(s_init)
        x0 = np.array(x0)
        
        obj0 = objective(x0)
        if obj0 < best_obj:
            best_obj = obj0
            best_x = x0.copy()
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    if best_x is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = [[float(best_x[3*i]), float(best_x[3*i+1]), float(best_x[3*i+2])] for i in range(N)]
    s = min_side(inner)
    
    return inner, (0.0, 0.0), s, 0.0

# EVOLVE-BLOCK-END
