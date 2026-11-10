# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hv(x, y):
    """Unit hexagon vertices at (x,y), angle 0."""
    th = np.radians(60 * np.arange(6))
    return np.column_stack([x + np.cos(th), y + np.sin(th)])

def _oside(pos):
    """Min outer hex side (angle 0, center 0) to contain unit hexes at pos."""
    on = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in (30,90,150,210,270,330)])
    m = 0.0
    for i in range(len(pos)):
        v = _hv(pos[i,0], pos[i,1])
        m = max(m, (v @ on.T).max())
    return m / (np.sqrt(3)/2)

def _sep(i, j, pos, tol=1e-6):
    v1 = _hv(pos[i,0], pos[i,1])
    v2 = _hv(pos[j,0], pos[j,1])
    ns = []
    for v in (v1, v2):
        for k in range(6):
            e = v[(k+1)%6] - v[k]
            n = np.array([e[1], -e[0]])
            n /= np.linalg.norm(n)
            ns.append(n)
    for n in ns:
        if (v1 @ n).max() < (v2 @ n).min() - tol or (v2 @ n).max() < (v1 @ n).min() - tol:
            return True
    return False

def _spen(pos, tol=1e-6):
    pen = 0.0
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            if not _sep(i, j, pos, tol):
                v1 = _hv(pos[i,0], pos[i,1])
                v2 = _hv(pos[j,0], pos[j,1])
                ns = []
                for v in (v1, v2):
                    for k in range(6):
                        e = v[(k+1)%6] - v[k]
                        n = np.array([e[1], -e[0]])
                        n /= np.linalg.norm(n)
                        ns.append(n)
                mg = 1e10
                for n in ns:
                    mg = min(mg, (v2 @ n).min() - (v1 @ n).max(), (v1 @ n).min() - (v2 @ n).max())
                pen += max(0.0, -mg)**2 * 100
    return pen

def hexagon_packing_11():
    t0 = time.time(); dl = t0 + 600
    s3 = np.sqrt(3); ep = 0.01
    d1, d2 = s3+ep, 3.0+ep
    a6 = np.radians(np.arange(0,360,60))
    
    def make_pos(a4):
        p = np.array([[0.0,0.0]])
        for a in a6: p = np.vstack([p,[[d1*np.cos(a),d1*np.sin(a)]]])
        for a in a4: p = np.vstack([p,[[d2*np.cos(a),d2*np.sin(a)]]])
        return p
    
    def obj(x):
        if time.time() > dl-30: return 1e10
        p = x.reshape(11,2)
        return _oside(p) + _spen(p)
    
    # Multiple restarts with different 4-position selections
    a4_options = [
        np.radians([30,90,150,210]),
        np.radians([90,270,0,180]),
        np.radians([30,150,270,330]),
        np.radians([90,210,30,330]),
        np.radians([0,120,240,60]),
    ]
    
    bx = None
    bs = 1e10
    
    for a4 in a4_options:
        if time.time() > dl-60: break
        p0 = make_pos(a4)
        x0 = p0.flatten()
        try:
            r = minimize(obj, x0, method='L-BFGS-B', options={'maxiter':500,'ftol':1e-12})
            p = r.x.reshape(11,2)
            if _spen(p) < 1e-3:
                s = _oside(p)
                if s < bs:
                    bs = s; bx = r.x.copy()
        except: pass
    
    if bx is None:
        # Fallback
        p0 = make_pos(a4_options[0])
        bx = p0.flatten()
        bs = _oside(p0)
    
    inner = np.column_stack([bx.reshape(11,2), np.zeros(11)])
    outer = np.array([0.0,0.0,0.0])
    side = bs + 1e-5
    
    # Validate
    ok = np.all(np.isfinite(inner)) and np.all(np.isfinite(outer)) and side >= np.sqrt(11)
    if ok:
        ov = _hv(0,0) * side
        for i in range(11):
            iv = _hv(inner[i,0], inner[i,1])
            for p in iv:
                for j in range(6):
                    p1v, p2v = ov[j], ov[(j+1)%6]
                    if (p2v[0]-p1v[0])*(p[1]-p1v[1]) - (p2v[1]-p1v[1])*(p[0]-p1v[0]) < -1e-6:
                        ok = False; break
                if not ok: break
            if not ok: break
    if ok:
        pf = bx.reshape(11,2)
        for i in range(11):
            for j in range(i+1,11):
                if not _sep(i,j,pf): ok = False; break
            if not ok: break
    
    if not ok:
        inner = np.array([[0,0,0],[-2.5,0,0],[2.5,0,0],[-1.25,2.17,0],[1.25,2.17,0],
                         [-1.25,-2.17,0],[1.25,-2.17,0],[-3.75,2.17,0],[3.75,2.17,0],
                         [-3.75,-2.17,0],[3.75,-2.17,0]])
        outer = np.array([0,0,0]); side = 8.0
    
    return inner, outer, side

# EVOLVE-BLOCK-END
