import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(np.arange(0, 360, 60))
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack([xs, ys])

def _check_overlap(h1, h2):
    axes = []
    for h in [h1, h2]:
        for i in range(6):
            e = h[(i + 1) % 6] - h[i]
            n = np.array([-e[1], e[0]])
            nn = np.linalg.norm(n)
            if nn > 1e-12:
                axes.append(n / nn)
    min_ov = float('inf')
    for ax in axes:
        p1 = np.dot(h1, ax)
        p2 = np.dot(h2, ax)
        m1, M1 = np.min(p1), np.max(p1)
        m2, M2 = np.min(p2), np.max(p2)
        if M1 <= m2 or M2 <= m1:
            return 0.0
        ov = min(M1, M2) - max(m1, m2)
        if ov < min_ov:
            min_ov = ov
    return min_ov

def _objective(params):
    n = N
    ip = params[:3 * n]
    op = params[3 * n:]
    ox, oy, os, oa = op
    if os < 1.0:
        return 1e6
    iv = []
    for i in range(n):
        iv.append(_hex_vertices(ip[3*i], ip[3*i+1], 1.0, ip[3*i+2]))
    ov = _hex_vertices(ox, oy, os, oa)
    pen = 0.0
    for i in range(n):
        for v in iv[i]:
            for j in range(6):
                v1 = ov[j]
                v2 = ov[(j+1)%6]
                e = v2 - v1
                nm = np.array([-e[1], e[0]])
                nn = np.linalg.norm(nm)
                if nn < 1e-12:
                    continue
                nm /= nn
                d = np.dot(v - v1, nm)
                if d < 0:
                    pen += d * d
    for i in range(n):
        for j in range(i+1, n):
            ov2 = _check_overlap(iv[i], iv[j])
            if ov2 > 0:
                pen += ov2 * ov2 * 100
    return os + pen

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    best_score = float('inf')
    best_result = None
    
    arrangements = []
    
    # Line
    lp = []
    for i in range(N):
        lp.extend([i * 2.0, 0.0, 0.0])
    lp.extend([0.0, 0.0, 15.0, 0.0])
    arrangements.append(lp)
    
    # Hex cluster
    hp = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-1,2)]
    hpx = []
    for q,r in hp:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2)
        hpx.extend([x, y, 0.0])
    hpx.extend([0.0, 0.0, 6.0, 0.0])
    arrangements.append(hpx)
    
    np.random.seed(42)
    for _ in range(5):
        rp = []
        for i in range(N):
            rp.extend([np.random.uniform(-3,3), np.random.uniform(-3,3), np.random.uniform(0,60)])
        rp.extend([0.0, 0.0, 8.0, 0.0])
        arrangements.append(rp)
    
    for arr in arrangements:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(_objective, arr, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-5})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except:
            pass
    
    if best_result is not None:
        inner = []
        for i in range(N):
            inner.append([best_result[3*i], best_result[3*i+1], best_result[3*i+2]])
        ox, oy, os, oa = best_result[3*N], best_result[3*N+1], best_result[3*N+2], best_result[3*N+3]
        if os < 1.0:
            os = 1.0
        return inner, (ox, oy), os, oa
    else:
        inner = [[i*2.0, 0.0, 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 15.0, 0.0
# EVOLVE-BLOCK-END
