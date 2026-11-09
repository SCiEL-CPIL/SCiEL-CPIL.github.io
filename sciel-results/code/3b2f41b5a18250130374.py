"""Pack 50 equal circles in the unit disk via radius-based optimization + SA polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE_MARGIN = 5.0  # seconds of safety margin


def _score(pts):
    """Compute the packing score: min(min pair dist/2, min (1-|c_i|))."""
    n = len(pts)
    if n == 0:
        return 0.0
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    radii = np.sqrt(np.sum(pts * pts, axis=1))
    min_boundary = np.min(1.0 - radii)
    return min(min_pair, min_boundary)


def _soft_score(pts, beta=200.0):
    """Smooth approximation of the minimum distance using log-sum-exp."""
    n = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    radii = np.sqrt(np.sum(pts * pts, axis=1))
    boundary_dists = 1.0 - radii
    
    all_dists = np.concatenate([dists[~np.eye(n, dtype=bool)], boundary_dists])
    
    min_val = -np.log(np.sum(np.exp(-beta * all_dists))) / beta
    return min_val


def _hex_lattice_init():
    """Generate a hexagonal lattice initialization and select N best points."""
    best_pts = None
    best_score = 0.0
    
    for spacing in np.linspace(0.22, 0.30, 8):
        k = int(2.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.sqrt(x * x + y * y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append([x, y])
        pts = np.array(rows)
        if len(pts) >= N:
            radii = np.sqrt(np.sum(pts * pts, axis=1))
            idx = np.argsort(radii)[:N]
            pts_sel = pts[idx]
            score = _score(pts_sel)
            if score > best_score:
                best_score = score
                best_pts = pts_sel
    
    if best_pts is None:
        best_pts = np.random.randn(N, 2) * 0.8
        best_score = _score(best_pts)
    
    return best_pts, best_score


def _violation(pts, r):
    """Check if packing with radius r is valid. Returns max violation (0 if valid)."""
    n = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair = np.min(dists) / 2.0
    radii = np.sqrt(np.sum(pts * pts, axis=1))
    min_boundary = np.min(1.0 - radii)
    
    return max(0.0, r - min_pair, r - min_boundary)


def _optimize_for_radius(pts, r, max_time=10.0):
    """Try to find a valid packing with radius r using SLSQP."""
    start_time = time.time()
    n = len(pts)
    
    def objective(x):
        pts = x.reshape(n, 2)
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.sqrt(np.sum(pts * pts, axis=1))
        boundary_dists = 1.0 - radii
        
        all_dists = np.concatenate([dists[~np.eye(n, dtype=bool)], boundary_dists])
        
        # Penalize distances below r
        violations = np.maximum(0.0, r - all_dists)
        return np.sum(violations**2)
    
    def jac(x):
        pts = x.reshape(n, 2)
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.sqrt(np.sum(pts * pts, axis=1))
        boundary_dists = 1.0 - radii
        
        all_dists = np.concatenate([dists[~np.eye(n, dtype=bool)], boundary_dists])
        violations = np.maximum(0.0, r - all_dists)
        
        grad = np.zeros(2 * n)
        
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = dists[i, j]
                if d > 1e-10:
                    v = violations[pair_idx]
                    if v > 0:
                        dir_vec = (pts[i] - pts[j]) / d
                        grad[2*i:2*i+2] += 2 * v * dir_vec
                        grad[2*j:2*j+2] -= 2 * v * dir_vec
                pair_idx += 1
        
        for i in range(n):
            v = violations[n*(n-1)//2 + i]
            if v > 0:
                rad = radii[i]
                if rad > 1e-10:
                    grad[2*i:2*i+2] += 2 * v * (pts[i] / rad)
        
        return grad
    
    x0 = pts.flatten()
    
    def constraint(i):
        def c(x):
            pts = x.reshape(n, 2)
            return 1.0 - np.sqrt(np.sum(pts[i]**2)) - 0.0001
        return c
    
    cons = [{'type': 'ineq', 'fun': constraint(i)} for i in range(n)]
    
    result = minimize(objective, x0, method='SLSQP', jac=jac,
                      constraints=cons, options={'maxiter': 100, 'ftol': 1e-10})
    
    if result.fun < 1e-8:
        return result.x.reshape(n, 2), True
    return pts, False


def _find_max_radius(pts, r_low, r_high, max_time=30.0):
    """Binary search for maximum feasible radius."""
    start_time = time.time()
    best_pts = pts.copy()
    best_r = r_low
    
    for _ in range(10):
        if time.time() - start_time > max_time:
            break
        r_mid = (r_low + r_high) / 2
        new_pts, success = _optimize_for_radius(pts, r_mid, max_time=min(5.0, max_time - (time.time() - start_time)))
        if success:
            best_pts = new_pts
            best_r = r_mid
            r_low = r_mid
        else:
            r_high = r_mid
    
    return best_pts, best_r


def _simulated_annealing(pts, max_time=60.0):
    """Simulated annealing with improved parameters."""
    start_time = time.time()
    n = len(pts)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    T = 0.03
    T_min = 0.0005
    cooling_rate = 0.998
    
    current_pts = pts.copy()
    current_score = best_score
    
    while time.time() - start_time < max_time:
        if T < T_min:
            break
        
        i = np.random.randint(n)
        delta = np.random.randn(2) * T
        new_pts = current_pts.copy()
        new_pts[i] += delta
        
        r_new = np.sqrt(np.sum(new_pts[i]**2))
        if r_new > 1.0 - 0.0001:
            new_pts[i] *= (1.0 - 0.0001) / r_new
        
        new_score = _score(new_pts)
        
        if new_score >= current_score:
            current_pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        else:
            delta_e = (new_score - current_score) / T
            if np.random.random() < np.exp(delta_e):
                current_pts = new_pts
                current_score = new_score
        
        T *= cooling_rate
    
    return best_pts, best_score


def _local_polish(pts, max_time=15.0):
    """Local polish using SLSQP with soft-min objective."""
    start_time = time.time()
    n = len(pts)
    
    def objective(x):
        pts = x.reshape(n, 2)
        return -_soft_score(pts, beta=500.0)
    
    x0 = pts.flatten()
    
    def constraint(i):
        def c(x):
            pts = x.reshape(n, 2)
            return 1.0 - np.sqrt(np.sum(pts[i]**2)) - 0.0001
        return c
    
    cons = [{'type': 'ineq', 'fun': constraint(i)} for i in range(n)]
    
    result = minimize(objective, x0, method='SLSQP',
                      constraints=cons, options={'maxiter': 200, 'ftol': 1e-12})
    
    if result.fun < objective(x0):
        return result.x.reshape(n, 2)
    return pts


def _final_polish(pts, max_time=10.0):
    """Final high-precision polish."""
    start_time = time.time()
    n = len(pts)
    
    def objective(x):
        pts = x.reshape(n, 2)
        return -_soft_score(pts, beta=1000.0)
    
    x0 = pts.flatten()
    
    def constraint(i):
        def c(x):
            pts = x.reshape(n, 2)
            return 1.0 - np.sqrt(np.sum(pts[i]**2)) - 0.00001
        return c
    
    cons = [{'type': 'ineq', 'fun': constraint(i)} for i in range(n)]
    
    result = minimize(objective, x0, method='SLSQP',
                      constraints=cons, options={'maxiter': 300, 'ftol': 1e-14})
    
    if result.fun < objective(x0):
        return result.x.reshape(n, 2)
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Step 1: Structured initialization
    pts, score = _hex_lattice_init()
    
    # Step 2: Local polish
    if time.time() < deadline - 90.0:
        pts = _local_polish(pts, max_time=min(15.0, deadline - time.time() - 90.0))
    
    # Step 3: Radius-based optimization
    if time.time() < deadline - 60.0:
        r_low = _score(pts)
        r_high = min(0.13, r_low * 1.1)
        pts, r_found = _find_max_radius(pts, r_low, r_high, max_time=min(30.0, deadline - time.time() - 60.0))
    
    # Step 4: SA polish
    if time.time() < deadline - 30.0:
        pts, _ = _simulated_annealing(pts, max_time=min(40.0, deadline - time.time() - 30.0))
    
    # Step 5: Final local polish
    if time.time() < deadline - 15.0:
        pts = _local_polish(pts, max_time=min(10.0, deadline - time.time() - 15.0))
    
    # Step 6: Final high-precision polish
    if time.time() < deadline - 5.0:
        pts = _final_polish(pts, max_time=min(8.0, deadline - time.time() - 5.0))
    
    # Ensure validity
    radii = np.sqrt(np.sum(pts * pts, axis=1))
    for i in range(len(pts)):
        if radii[i] > 1.0:
            pts[i] *= (1.0 - 1e-9) / radii[i]
    
    return pts
