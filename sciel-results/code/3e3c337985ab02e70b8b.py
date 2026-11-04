import numpy as np
import time
import math

N = 12
_DEADLINE = time.time() + 110.0

def _hex_verts(cx, cy, a_deg):
    a = math.radians(a_deg)
    return np.array([[cx + math.cos(a + k*math.pi/3), cy + math.sin(a + k*math.pi/3)] for k in range(6)])

def _enclosing_side(verts, cx, cy, rot_deg):
    """verts: (M,2). Returns min side of hexagon (center, rot) containing all verts."""
    rot = math.radians(rot_deg)
    normals = np.array([[math.cos(rot + k*math.pi/3), math.sin(rot + k*math.pi/3)] for k in range(6)])
    rel = verts - np.array([cx, cy])
    proj = rel @ normals.T
    return np.max(np.abs(proj)) / (math.sqrt(3)/2)

def _no_overlap(pos, angs, eps=1e-6):
    """Fast overlap check using center distances and bounding boxes."""
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            dist = math.sqrt(dx*dx + dy*dy)
            # Two unit hexagons can overlap if centers are closer than sqrt(3)
            # Minimum distance between centers of non-overlapping unit hexagons is sqrt(3)
            if dist < math.sqrt(3) - eps:
                return False
    return True

def construct():
    t0 = time.time()
    sqrt3 = math.sqrt(3.0)
    
    # Initial hexagonal cluster of 12
    pos = np.zeros((N, 2))
    angs = np.zeros(N)
    
    # Hex lattice positions
    hex_positions = []
    # Ring 0: 1
    hex_positions.append((0, 0))
    # Ring 1: 6
    for k in range(6):
        a = k * math.pi / 3
        hex_positions.append((sqrt3 * math.cos(a), sqrt3 * math.sin(a)))
    # Ring 2: 12, take 5 to make 12 total
    for k in range(6):
        a = k * math.pi / 3
        hex_positions.append((2*sqrt3 * math.cos(a), 2*sqrt3 * math.sin(a)))
    hex_positions = hex_positions[:N]
    
    for i in range(N):
        pos[i] = hex_positions[i]
    
    # Compute initial
    verts = np.vstack([_hex_verts(pos[i,0], pos[i,1], angs[i]) for i in range(N)])
    cx = np.mean(verts[:,0])
    cy = np.mean(verts[:,1])
    side = _enclosing_side(verts, cx, cy, 0.0)
    
    best = (pos.copy(), angs.copy(), cx, cy, 0.0, side)
    
    # SA
    T = 0.3
    T_min = 1e-9
    alpha = 0.9998
    step = 0.4
    
    np.random.seed(42)
    
    for it in range(100000):
        if time.time() > _DEADLINE:
            break
        
        i = np.random.randint(N)
        dp = np.random.normal(0, step, 2)
        da = np.random.normal(0, step*5)
        
        old_p = pos[i].copy()
        old_a = angs[i]
        
        pos[i] += dp
        angs[i] = (old_a + da) % 360
        
        if not _no_overlap(pos, angs):
            pos[i] = old_p
            angs[i] = old_a
            continue
        
        verts = np.vstack([_hex_verts(pos[j,0], pos[j,1], angs[j]) for j in range(N)])
        new_cx = np.mean(verts[:,0])
        new_cy = np.mean(verts[:,1])
        new_side = _enclosing_side(verts, new_cx, new_cy, 0.0)
        
        delta = new_side - best[5]
        if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
            best = (pos.copy(), angs.copy(), new_cx, new_cy, 0.0, new_side)
        else:
            pos[i] = old_p
            angs[i] = old_a
        
        T = max(T * alpha, T_min)
    
    # Polish: optimize center and rotation
    bp, ba, bcx, bcy, brot, bside = best
    verts = np.vstack([_hex_verts(bp[i,0], bp[i,1], ba[i]) for i in range(N)])
    
    # Refine center
    for _ in range(30):
        if time.time() > _DEADLINE:
            break
        best_cx, best_cy = bcx, bcy
        best_s = bside
        for dcx in np.linspace(-0.2, 0.2, 5):
            for dcy in np.linspace(-0.2, 0.2, 5):
                s = _enclosing_side(verts, bcx+dcx, bcy+dcy, brot)
                if s < best_s:
                    best_s = s
                    best_cx, best_cy = bcx+dcx, bcy+dcy
        bcx, bcy, bside = best_cx, best_cy, best_s
    
    # Refine rotation
    for _ in range(30):
        if time.time() > _DEADLINE:
            break
        best_r = brot
        best_s = bside
        for dr in np.linspace(-3, 3, 7):
            s = _enclosing_side(verts, bcx, bcy, brot+dr)
            if s < best_s:
                best_s = s
                best_r = brot+dr
        brot, bside = best_r, best_s
    
    inner = [[bp[i,0], bp[i,1], ba[i]] for i in range(N)]
    return inner, (bcx, bcy), bside, brot
