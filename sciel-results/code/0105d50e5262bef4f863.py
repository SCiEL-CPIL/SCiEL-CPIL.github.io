import numpy as np
from scipy.optimize import differential_evolution, minimize

def _vertex_scores(pts):
    n = len(pts)
    c = pts.mean(axis=0)
    p = pts - c
    dcent = np.linalg.norm(p, axis=1)
    mean_d = dcent.mean()
    if mean_d <= 0.01:
        return 0.0
    p = p / mean_d
    scores = []
    for i in range(n):
        diff = p - p[i]
        dists = np.linalg.norm(diff, axis=1)
        dists = np.delete(dists, i)
        dists = np.sort(dists)
        if dists.min() < 0.05:
            scores.append(0.0)
            continue
        gaps = np.diff(dists)
        k = int(np.argmin(gaps))
        D = 0.5*(dists[k] + dists[k+1])
        if D < 0.1:
            scores.append(0.0)
            continue
        if k == 0:
            four = dists[0:4]
        elif k == 17:
            four = dists[15:19]
        else:
            four = dists[k-1:k+3]
        ratios = np.maximum(four/D, D/four)
        raw = ratios.mean()
        scores.append(min(1.0, 1.0/raw))
    return min(scores)

def _crosses(pts):
    n = len(pts)
    cr = []
    for i in range(n):
        a = pts[i]; b = pts[(i+1)%n]; c = pts[(i+2)%n]
        v1 = b-a; v2 = c-b
        cr.append(v1[0]*v2[1]-v1[1]*v2[0])
    return np.array(cr)

def _convex_violation(pts):
    cr = _crosses(pts)
    if np.any(np.abs(cr) < 1e-7):
        return 1.0 + np.abs(cr).min()
    # mixed signs?
    if np.all(cr > 0) or np.all(cr < 0):
        return 0.0
    # magnitude of inconsistency
    s = np.sign(cr[0])
    bad = cr * s
    return 1.0 + np.abs(bad[bad < 0]).max()

def _pts_from_params(params):
    n = 20
    M = 5
    theta = 2*np.pi*np.arange(n)/n
    r = np.ones(n)
    idx = 0
    for m in range(1, M+1):
        a = params[idx]; b = params[idx+1]; idx += 2
        r = r + a*np.cos(m*theta) + b*np.sin(m*theta)
    x = r*np.cos(theta); y = r*np.sin(theta)
    return np.stack([x,y],axis=1)

def _objective(params):
    pts = _pts_from_params(params)
    # repeated points check
    for i in range(20):
        for j in range(i+1,20):
            if np.linalg.norm(pts[i]-pts[j]) < 1e-6:
                return 5.0
    viol = _convex_violation(pts)
    score = _vertex_scores(pts)
    if viol > 0:
        return 1.0 + 10.0*viol + (1.0-score)
    return 1.0 - score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    M = 5
    np.random.seed(0)
    bounds = [(-0.12,0.12)]*(2*M)
    res = differential_evolution(_objective, bounds, seed=0, maxiter=80,
                                  popsize=25, tol=1e-6, polish=True,
                                  init='latinhypercube', mutation=(0.4,1.0),
                                  recombination=0.8)
    best_params = res.x
    best_pts = _pts_from_params(best_params)
    best_score = _vertex_scores(best_pts) if _convex_violation(best_pts)==0 else 0.0

    # local polish with Nelder-Mead on valid region
    def obj_local(params):
        pts = _pts_from_params(params)
        viol = _convex_violation(pts)
        if viol > 0:
            return 1.0 + 10*viol
        return 1.0 - _vertex_scores(pts)
    try:
        lm = minimize(obj_local, best_params, method='Nelder-Mead',
                      options={'maxiter':2000,'xatol':1e-9,'fatol':1e-9})
        if _convex_violation(_pts_from_params(lm.x)) == 0:
            s2 = _vertex_scores(_pts_from_params(lm.x))
            if s2 > best_score:
                best_pts = _pts_from_params(lm.x)
                best_score = s2
    except Exception:
        pass

    # fallback to regular if invalid
    if _convex_violation(best_pts) != 0:
        angles = np.linspace(0,2*np.pi,n,endpoint=False)
        best_pts = np.stack([np.cos(angles),np.sin(angles)],axis=1)
    return best_pts
