"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

N = 11

def barycentric_to_cartesian(bary):
    """Convert barycentric coords (u,v) with w=1-u-v to Cartesian."""
    u = bary[:, 0]
    v = bary[:, 1]
    x = 0.5 + v - 0.5 * u
    y = (3 ** 0.5) / 2.0 * (1.0 - u - v)
    return np.column_stack([x, y])

def min_triangle_area(pts):
    """Minimum area of any triangle formed by 3 of the N points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            pij = pts[j] - pi
            for k in range(j + 1, n):
                pik = pts[k] - pi
                area = 0.5 * abs(pij[0] * pik[1] - pij[1] * pik[0])
                if area < min_area:
                    min_area = area
    return min_area

def objective(bary_flat):
    """Minimize negative min triangle area."""
    bary = bary_flat.reshape(N, 2)
    u = np.clip(bary[:, 0], 0.0, 1.0)
    v = np.clip(bary[:, 1], 0.0, 1.0)
    w = 1.0 - u - v
    mask = w < 0
    if np.any(mask):
        s = u[mask] + v[mask]
        s[s == 0] = 1.0
        u[mask] /= s
        v[mask] /= s
    pts = barycentric_to_cartesian(np.column_stack([u, v]))
    return -min_triangle_area(pts)

def make_symmetric_init():
    """Symmetric 11-point configuration in barycentric coords."""
    verts = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    mids = np.array([[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]])
    cx, cy = 1.0/3, 1.0/3
    r = 1.0/6
    interior = np.array([
        [cx, cy],
        [cx + r, cy],
        [cx, cy + r],
        [cx - r, cy],
        [cx, cy - r],
    ])
    return np.vstack([verts, mids, interior])

def make_random_init(seed):
    rng = np.random.RandomState(seed)
    pts = []
    while len(pts) < N:
        u = rng.random()
        v = rng.random()
        if u + v < 1.0 and u > 0.03 and v > 0.03:
            pts.append([u, v])
    return np.array(pts)

def construct():
    t0 = time.time()
    deadline = t0 + 138.0
    
    best_score = 0.0
    best_bary = None
    
    inits = [make_symmetric_init()]
    for s in range(8):
        inits.append(make_random_init(1000 + s * 7919))
    
    for bary0 in inits:
        if time.time() > deadline - 8.0:
            break
        try:
            res = minimize(objective, bary0.flatten(), method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            if time.time() > deadline - 3.0:
                break
            sc = -res.fun
            if sc > best_score:
                best_score = sc
                best_bary = res.x.reshape(N, 2)
        except Exception:
            pass
    
    if best_bary is None:
        best_bary = make_symmetric_init()
    
    u = np.clip(best_bary[:, 0], 0.0, 1.0)
    v = np.clip(best_bary[:, 1], 0.0, 1.0)
    w = 1.0 - u - v
    mask = w < 0
    if np.any(mask):
        s = u[mask] + v[mask]
        s[s == 0] = 1.0
        u[mask] /= s
        v[mask] /= s
    
    return barycentric_to_cartesian(np.column_stack([u, v]))

# EVOLVE-BLOCK-END
