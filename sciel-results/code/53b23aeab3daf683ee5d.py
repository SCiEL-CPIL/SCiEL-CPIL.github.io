import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, angle_deg=0.0):
    a = np.radians(angle_deg)
    return np.array([[cx + np.cos(a + np.pi*i/3), cy + np.sin(a + np.pi*i/3)] for i in range(6)])

def _min_outer_side(positions, angles, outer_angle=0.0):
    all_v = np.vstack([_hex_vertices(p[0], p[1], a) for p, a in zip(positions, angles)])
    theta = np.radians(outer_angle)
    normals = np.array([[np.cos(theta + np.pi/6 + np.pi*k/3), np.sin(theta + np.pi/6 + np.pi*k/3)] for k in range(6)])
    dots = all_v @ normals.T
    return max(2.0 * np.max(dots) / np.sqrt(3.0), 0.0)

def _hex_cluster(r):
    pos = []
    for q in range(-r, r+1):
        for rr in range(-r, r+1):
            s = -q - rr
            if max(abs(q), abs(rr), abs(s)) <= r:
                pos.append((np.sqrt(3)*(q + rr/2.0), 1.5*rr))
    return pos

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    best = (float('inf'), None)
    
    # Generate candidate initial arrangements
    cands = []
    for r in [1, 2, 3]:
        p = _hex_cluster(r)
        if len(p) >= N:
            cands.append(np.array(p[:N]))
    
    # Also try compact rectangular-ish
    p2 = []
    for i in range(N):
        row, col = divmod(i, 6)
        p2.append((np.sqrt(3)*(col + 0.5*(row%2)), 1.5*row))
    cands.append(np.array(p2))
    
    for pos0 in cands:
        if time.time() > deadline:
            break
        pos0 = pos0[:N].copy()
        
        def obj(x):
            p = x[:2*N].reshape(N, 2)
            a = x[2*N:]
            s = _min_outer_side(p.tolist(), a.tolist(), 0.0)
            pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(p[i]-p[j])
                    if d < np.sqrt(3) - 0.02:
                        pen += 1000*(np.sqrt(3)-d)**2
            return s + pen
        
        x0 = np.concatenate([pos0.ravel(), np.zeros(N)])
        try:
            res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 300, 'xatol':1e-3, 'fatol':1e-3})
            if res.fun < best[0]:
                p = res.x[:2*N].reshape(N,2)
                a = res.x[2*N:]
                # Verify
                ok = all(np.linalg.norm(p[i]-p[j]) >= np.sqrt(3)-0.01 for i in range(N) for j in range(i+1,N))
                if ok:
                    s = _min_outer_side(p.tolist(), a.tolist(), 0.0)
                    if s < best[0]:
                        inner = [[float(p[i,0]), float(p[i,1]), float(a[i])] for i in range(N)]
                        best = (s, (inner, (0.0, 0.0), float(s), 0.0)))
        except:
            pass
    
    if best[1] is None:
        p = np.array(_hex_cluster(2)[:N])
        inner = [[float(p[i,0]), float(p[i,1]), 0.0] for i in range(N)]
        s = _min_outer_side(p.tolist(), [0.0]*N, 0.0)
        return (inner, (0.0, 0.0), float(s), 0.0)
    return best[1]
# EVOLVE-BLOCK-END
