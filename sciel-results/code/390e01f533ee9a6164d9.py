import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5

def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r

def _violation_and_grad(pts, r_target):
    """
    Compute violation and gradient for a given radius r_target.
    Violation is sum of squared violations.
    """
    pts = np.array(pts).reshape(N, 2)
    n_vars = 2 * N
    
    violation = 0.0
    grad = np.zeros(n_vars)
    
    # Pairwise violations
    for i in range(N):
        for j in range(i + 1, N):
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 2 * r_target:
                if dist < 1e-12:
                    # If coincident, push apart randomly or along x
                    # This is a singular point, but rare
                    diff = np.array([1.0, 0.0])
                    dist = 1.0
                overlap = 2 * r_target - dist
                # v = overlap^2
                violation += overlap * overlap
                
                # grad w.r.t pts[i]
                # d(v^2)/d(pts[i]) = 2 * v * d(2r - dist)/d(pts[i]) = 2 * v * (-diff/dist)
                factor = 2 * overlap * (-1.0 / dist)
                grad[2*i] += factor * diff[0]
                grad[2*i+1] += factor * diff[1]
                
                # grad w.r.t pts[j]
                # d(v^2)/d(pts[j]) = 2 * v * d(2r - dist)/d(pts[j]) = 2 * v * (diff/dist)
                factor_j = 2 * overlap * (1.0 / dist)
                grad[2*j] += factor_j * diff[0]
                grad[2*j+1] += factor_j * diff[1]
    
    # Boundary violations
    for i in range(N):
        for k in range(2):
            coord = pts[i, k]
            # Distance to nearest edge
            if coord < 0:
                dist_edge = HALF + coord
            else:
                dist_edge = HALF - coord
            
            if dist_edge < r_target:
                overlap = r_target - dist_edge
                violation += overlap * overlap
                
                # grad w.r.t coord
                # if coord < 0, dist_edge = 0.5 + coord, d(dist_edge)/d(coord) = 1
                # if coord >= 0, dist_edge = 0.5 - coord, d(dist_edge)/d(coord) = -1
                if coord < 0:
                    d_dist_dc = 1.0
                else:
                    d_dist_dc = -1.0
                
                # d(v^2)/d(coord) = 2 * v * d(r - dist_edge)/d(coord) = 2 * v * (-d_dist_dc)
                factor = 2 * overlap * (-d_dist_dc)
                grad[2*i+k] += factor
    
    return violation, grad

def _lbfgs_minimize_violation(pts0, r_target, deadline):
    """Minimize violation for a fixed radius using L-BFGS."""
    x0 = pts0.ravel()
    
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        v, _ = _violation_and_grad(p, r_target)
        return v
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        _, g = _violation_and_grad(p, r_target)
        return g
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', jac=grad,
                       bounds=[(-HALF, HALF)] * (2 * N),
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000})
        pts = res.x.reshape(N, 2)
        pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)
        return pts, float(res.fun)
    except Exception:
        return pts0, float('inf')

def _simulated_annealing(pts0, r_target, deadline, max_steps=50000):
    """Simulated annealing to minimize violation for a fixed radius."""
    pts = pts0.copy()
    current_viol, _ = _violation_and_grad(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    
    T = 0.01
    T_min = 1e-6
    cooling = 0.999
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Pick a random circle
        i = np.random.randint(N)
        # Perturb position
        delta = np.random.normal(0, 0.01, 2)
        new_pts = pts.copy()
        new_pts[i] += delta
        
        # Keep within bounds (roughly)
        new_pts = np.clip(new_pts, -HALF + 1e-9, HALF - 1e-9)
        
        new_viol, _ = _violation_and_grad(new_pts, r_target)
        
        if new_viol < current_viol:
            pts = new_pts
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_pts = pts.copy()
        else:
            # Metropolis criterion
            delta_v = new_viol - current_viol
            if np.random.rand() < np.exp(-delta_v / T):
                pts = new_pts
                current_viol = new_viol
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_pts, best_viol

def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice, jittered."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF + 1e-9, HALF - 1e-9)

def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin for 150s limit
    rng = np.random.default_rng(12345)

    best_pts = _hex_init(rng, jitter=0.01)
    best_r = _compute_r(best_pts)

    def try_update(pts):
        nonlocal best_pts, best_r
        r = _compute_r(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Binary search on radius
    r_low = best_r
    r_high = 0.08  # Upper bound guess
    
    # First, try to improve r_low with SA and L-BFGS
    for i in range(5):
        if time.time() > deadline - 30.0:
            break
        
        # Simulated Annealing
        if time.time() < deadline - 10.0:
            sa_pts, sa_viol = _simulated_annealing(best_pts, r_low, deadline, max_steps=20000)
            try_update(sa_pts)
        
        # L-BFGS Polish
        if time.time() < deadline - 5.0:
            lbfgs_pts, lbfgs_viol = _lbfgs_minimize_violation(best_pts, r_low, deadline)
            try_update(lbfgs_pts)
        
        # Check if we can increase r
        current_r = _compute_r(best_pts)
        if current_r > r_low:
            r_low = current_r
            r_high = max(r_high, r_low * 1.05)
        else:
            break

    # Now binary search for higher r
    for i in range(15):
        if time.time() > deadline - 20.0:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to pack with r_mid
        # Use SA first to find a good configuration
        if time.time() < deadline - 10.0:
            sa_pts, sa_viol = _simulated_annealing(best_pts, r_mid, deadline, max_steps=30000)
            try_update(sa_pts)
        
        # Then polish with L-BFGS
        if time.time() < deadline - 5.0:
            lbfgs_pts, lbfgs_viol = _lbfgs_minimize_violation(best_pts, r_mid, deadline)
            try_update(lbfgs_pts)
        
        # Check if r_mid is feasible
        current_r = _compute_r(best_pts)
        if current_r >= r_mid - 1e-6:
            r_low = r_mid
        else:
            r_high = r_mid
    
    # Final polish with best_r
    if time.time() < deadline - 5.0:
        lbfgs_pts, lbfgs_viol = _lbfgs_minimize_violation(best_pts, best_r, deadline)
        try_update(lbfgs_pts)

    return best_pts
