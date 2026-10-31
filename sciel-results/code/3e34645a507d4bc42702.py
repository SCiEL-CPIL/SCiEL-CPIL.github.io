import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 140.0

def _is_valid(arr, tol=1e-7):
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x < -tol) or np.any(x > 1+tol) or np.any(y < -tol) or np.any(y > 1+tol):
        return False
    if np.any(r > x + tol) or np.any(r > 1-x + tol) or np.any(r > y + tol) or np.any(r > 1-y + tol):
        return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _penalty_obj(v, lb, lo):
    x = v[:N]; y = v[N:2*N]; r = v[2*N:3*N]
    obj = -r.sum()
    b = np.maximum(0, r-x) + np.maximum(0, r-(1-x)) + np.maximum(0, r-y) + np.maximum(0, r-(1-y))
    obj += lb * b.sum()
    dx = x[:,None]-x[None,:]; dy = y[:,None]-y[None,:]
    dist = np.sqrt(dx*dx+dy*dy)
    rs = r[:,None]+r[None,:]
    ov = np.maximum(0, rs - dist)
    np.fill_diagonal(ov, 0)
    obj += lo * (ov*ov).sum()
    return obj

def _hex_init():
    pts = []
    for row in range(5):
        for col in range(5):
            ox = 0.1 if row % 2 else 0.0
            pts.append([0.1+col*0.18+ox, 0.1+row*0.2])
    pts.append([0.5, 0.5])
    pts = np.array(pts[:N])
    xs, ys = pts[:,0], pts[:,1]
    md = np.inf
    for i in range(N):
        for j in range(i+1, N):
            md = min(md, np.hypot(xs[i]-xs[j], ys[i]-ys[j]))
    r0 = min(md/2*0.85, 0.5)
    r0 = min(r0, xs.min(), 1-xs.max(), ys.min(), 1-ys.max())
    return np.column_stack([xs, ys, np.full(N, r0)])

def _grid_init():
    s = 5
    pts = np.array([(i+0.5)/s for i in range(s)] + [(j+0.5)/s for j in range(s)]).reshape(-1,2)[:N]
    r0 = 0.5/s*0.85
    r0 = min(r0, pts[:,0].min(), 1-pts[:,0].max(), pts[:,1].min(), 1-pts[:,1].max())
    return np.column_stack([pts[:,0], pts[:,1], np.full(N, r0)])

def _rand_init(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.05, 0.95, N)
    y = rng.uniform(0.05, 0.95, N)
    md = np.full(N, 0.4)
    for i in range(N):
        for j in range(N):
            if i != j:
                md[i] = min(md[i], np.hypot(x[i]-x[j], y[i]-y[j]))
    r = 0.45*md
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    r = np.maximum(r, 0.02)
    return np.column_stack([x, y, r])

def _opt(arr, lb, lo, mit=300):
    v0 = arr.flatten()
    bounds = [(0,1)]*N + [(0,1)]*N + [(0,0.5)]*N
    try:
        res = minimize(_penalty_obj, v0, args=(lb,lo), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': mit, 'ftol':1e-14})
        return res.x.reshape(N,3)
    except:
        return arr

def construct():
    t0 = time.time()
    best, bv = None, -np.inf
    inits = [_hex_init(), _grid_init()]
    for s in [1,7,13,42,99,200,333,555]:
        inits.append(_rand_init(s))
    lams = [(5,50),(20,200),(50,500)]
    for lb, lo in lams:
        if time.time()-t0 > DEADLINE-15: break
        for ai in inits:
            if time.time()-t0 > DEADLINE-5: break
            a = _opt(ai, lb, lo)
            if _is_valid(a):
                v = a[:,2].sum()
                if v > bv:
                    bv, best = v, a.copy()
    if best is None:
        best = _grid_init()
    if not _is_valid(best):
        x,y,r = best[:,0], best[:,1], best[:,2]
        r = np.minimum(r, np.minimum(np.minimum(x,1-x), np.minimum(y,1-y)))
        for _ in range(100):
            ch = False
            for i in range(N):
                for j in range(i+1,N):
                    d = np.hypot(x[i]-x[j], y[i]-y[j])
                    if d < r[i]+r[j]:
                        ex = r[i]+r[j]-d
                        r[i] = max(0, r[i]-ex/2)
                        r[j] = max(0, r[j]-ex/2)
                        ch = True
            if not ch: break
        best = np.column_stack([x,y,r])
    if not _is_valid(best):
        best = _grid_init()
    return best
