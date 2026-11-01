import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 14
D = 3
K = N * (N - 1) // 2  # 91

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    min_sq = np.min(sq_dists)
    if min_sq < 1e-12:
        return np.inf
    max_sq = np.max(sq_dists)
    return (max_sq / min_sq)

def softmin_log(s, tau):
    s = np.maximum(s, 1e-18)
    m = np.min(s)
    return m - np.log(np.sum(np.exp(tau * (s - m)))) / tau

def softmax_log(s, tau):
    s = np.maximum(s, 1e-18)
    m = np.max(s)
    return m + np.log(np.sum(np.exp(-tau * (s - m)))) / tau

def objective_smooth(x, tau=100.0):
    points = x.reshape(N, D)
    points = points - np.mean(points, axis=0, keepdims=True)
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq = np.mean(sq_dists)
    if mean_sq < 1e-14:
        return 1e10
    points = points / np.sqrt(mean_sq)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    log_sq = np.log(sq_dists)
    smax = softmax_log(log_sq, tau)
    smin = softmin_log(log_sq, tau)
    return smax - smin

def construct():
    t0 = time.time()
    deadline = 143.0
    rng = np.random.default_rng(20240615)
    
    best_score = np.inf
    best_points = None

    def update(pts):
        nonlocal best_score, best_points
        if pts is None or not np.all(np.isfinite(pts)):
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, D):
            return
        sqd = get_pairwise_sq_dists(pts)
        if np.min(sqd) < 1e-10:
            return
        sc = raw_score(pts)
        if sc < best_score:
            best_score = sc
            best_points = pts.copy()

    # Build initial diverse candidates
    cands = []
    
    # Fibonacci sphere
    def fib_sphere(n):
        pts = []
        ga = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - 2 * i / (n - 1)
            r = np.sqrt(max(0, 1 - y*y))
            th = ga * i
            pts.append([np.cos(th)*r, y, np.sin(th)*r])
        return np.array(pts)
    cands.append(fib_sphere(N))
    
    # Icosahedron + center + 2
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0,1,phi],[0,1,-phi],[0,-1,phi],[0,-1,-phi],
        [1,phi,0],[1,-phi,0],[-1,phi,0],[-1,-phi,0],
        [phi,0,1],[phi,0,-1],[-phi,0,1],[-phi,0,-1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    c = np.array([0.,0.,0.])
    v1 = rng.normal(size=3); v1 /= np.linalg.norm(v1)
    v2 = rng.normal(size=3); v2 /= np.linalg.norm(v2)
    cands.append(np.vstack([ico, c, v1*0.8, v2*0.8]))
    
    # D_{3d} symmetric configs
    for (r1,r2,r3) in [(1.0,1.0,0.5),(1.2,0.9,0.4),(0.8,1.1,0.6),(1.0,1.0,0.8),(1.5,0.7,0.3)]:
        a1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
        p1 = np.column_stack([r1*np.cos(a1), r1*np.sin(a1), np.full(6, -r3)])
        a2 = a1 + np.pi/6
        p2 = np.column_stack([r2*np.cos(a2), r2*np.sin(a2), np.full(6, r3)])
        p3 = np.array([[0,0,-1.5*r3],[0,0,1.5*r3]])
        cands.append(np.vstack([p1,p2,p3]))
    
    # Random sphere
    for _ in range(4):
        p = rng.normal(size=(N, D))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        cands.append(p)
    
    # Random cube
    for _ in range(4):
        cands.append(rng.uniform(-1, 1, size=(N, D)))
    
    # Two rings
    for h in [0.3, 0.6, 1.0]:
        a1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a2 = a1 + np.pi/7
        r1 = np.column_stack([np.cos(a1), np.sin(a1), np.full(7, -h)])
        r2 = np.column_stack([np.cos(a2), np.sin(a2), np.full(7, h)])
        cands.append(np.vstack([r1, r2]))

    # Coordinate-space multi-stage optimization with progressive tau
    tau_sched = [10.0, 50.0, 200.0, 800.0, 3000.0]
    n_restarts = min(len(cands), 12)
    
    for i in range(n_restarts):
        if time.time() - t0 > deadline - 25:
            break
        x0 = cands[i].flatten()
        cur = x0
        for tau in tau_sched:
            if time.time() - t0 > deadline - 5:
                break
            try:
                res = minimize(objective_smooth, cur, method='L-BFGS-B',
                               args=(tau,),
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    cur = res.x
                    pts = cur.reshape(N, D)
                    update(pts)
            except Exception:
                pass

    # Additional random restarts
    for _ in range(20):
        if time.time() - t0 > deadline - 10:
            break
        x0 = rng.normal(size=(N * D))
        x0 /= np.linalg.norm(x0)
        cur = x0
        for tau in [50.0, 300.0, 2000.0]:
            if time.time() - t0 > deadline - 3:
                break
            try:
                res = minimize(objective_smooth, cur, method='L-BFGS-B',
                               args=(tau,),
                               options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    cur = res.x
                    update(cur.reshape(N, D))
            except Exception:
                pass

    # Final high-precision polish
    if best_points is not None and time.time() - t0 < deadline - 2:
        try:
            res = minimize(objective_smooth, best_points.flatten(), method='L-BFGS-B',
                           args=(5000.0,),
                           options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-11})
            if res.x is not None and np.all(np.isfinite(res.x)):
                update(res.x.reshape(N, D))
        except Exception:
            pass

    if best_points is None:
        best_points = cands[0]
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {raw_score(pts)}")
