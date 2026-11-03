import numpy as np
import time
import math

N = 11

def hex_vertices(cx, cy, angle_deg, side=1.0):
    angle_rad = math.radians(angle_deg)
    return [(cx + side * math.cos(angle_rad + k * math.pi / 3),
             cy + side * math.sin(angle_rad + k * math.pi / 3)) for k in range(6)]

def polygons_overlap(v1, v2):
    for verts in [v1, v2]:
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i+1) % 6]
            nx, ny = -(y2-y1), (x2-x1)
            proj1 = [vx*nx+vy*ny for vx,vy in v1]
            proj2 = [vx*nx+vy*ny for vx,vy in v2]
            if max(proj1) < min(proj2) or max(proj2) < min(proj1):
                return False
    return True

def enclosing_hex_side(pos, angles):
    normals = [(math.cos(math.radians(30+k*60)), math.sin(math.radians(30+k*60))) for k in range(6)]
    max_p = 0.0
    for i in range(N):
        for vx, vy in hex_vertices(pos[i][0], pos[i][1], angles[i]):
            for nx, ny in normals:
                p = abs(vx*nx + vy*ny)
                if p > max_p: max_p = p
    return 2.0 * max_p / math.sqrt(3)

def construct():
    t0 = time.time()
    dl = t0 + 140.0
    d = math.sqrt(3)
    
    # Honeycomb: center + 6 + 4
    pos = [[0.0, 0.0]]
    for k in range(6):
        th = k * math.pi / 3
        pos.append([d*math.cos(th), d*math.sin(th)])
    for k in [0, 1, 2, 3]:
        th = k * math.pi / 3
        pos.append([2*d*math.cos(th), 2*d*math.sin(th)])
    angles = [0.0] * N
    
    best = enclosing_hex_side(pos, angles)
    bp, ba = [p[:] for p in pos], angles[:]
    
    T = 0.3
    while T > 1e-7 and time.time() < dl:
        i = np.random.randint(N)
        op, oa = pos[i][:], angles[i]
        if np.random.random() < 0.7:
            pos[i][0] += np.random.normal(0, 0.05)
            pos[i][1] += np.random.normal(0, 0.05)
        else:
            angles[i] += np.random.normal(0, 5.0)
        
        vi = hex_vertices(pos[i][0], pos[i][1], angles[i])
        ok = all(not polygons_overlap(vi, hex_vertices(pos[j][0], pos[j][1], angles[j])) for j in range(N) if j != i)
        
        if ok:
            ns = enclosing_hex_side(pos, angles)
            if ns < best or np.random.random() < math.exp(-(ns-best)/T):
                best, bp, ba = ns, [p[:] for p in pos], angles[:]
            else:
                pos[i], angles[i] = op, oa
        else:
            pos[i], angles[i] = op, oa
        T *= 0.9995
    
    return [[bp[i][0], bp[i][1], ba[i]] for i in range(N)], (0.0, 0.0), best, 0.0
