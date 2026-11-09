import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    def compute_radius(pts):
        """Compute the packing radius for given points."""
        if pts.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms > 1.0 + 1e-10):
            return 0.0
        b_min = np.min(1.0 - norms)
        # Pairwise distances
        d_min = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < d_min:
                    d_min = d
        r = min(d_min / 2.0, b_min)
        return max(r, 0.0)

    def constraint_violation(pts, r):
        """Return total constraint violation for radius r (0 means feasible)."""
        if pts.shape != (N, 2):
            return 1e10
        if not np.all(np.isfinite(pts)):
            return 1e10
        # Boundary: 1 - |c_i| >= r  =>  |c_i| <= 1 - r
        norms = np.linalg.norm(pts, axis=1)
        boundary_viol = np.maximum(0, norms - (1.0 - r))
        # Pairwise: |c_i - c_j| >= 2r
        pair_viol = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 2.0 * r:
                    pair_viol += (2.0 * r - d) ** 2
        return np.sum(boundary_viol ** 2) + pair_viol

    def smooth_objective(pts, r, soft_beta=20.0):
        """Smooth surrogate: minimize soft-min of all constraints."""
        if pts.shape != (N, 2):
            return 1e10
        if not np.all(np.isfinite(pts)):
            return 1e10
        
        norms = np.linalg.norm(pts, axis=1)
        # Boundary margins: 1 - |c_i| - r (want these >= 0)
        b_margins = 1.0 - norms - r
        
        # Pairwise margins: |c_i - c_j|/2 - r (want these >= 0)
        p_margins = []
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                p_margins.append(d / 2.0 - r)
        p_margins = np.array(p_margins)
        
        all_margins = np.concatenate([b_margins, p_margins])
        
        # Soft-min via log-sum-exp (negative for min)
        # soft_min(x) = -logsumexp(-beta * x) / beta
        neg_beta_x = -soft_beta * all_margins
        max_val = np.max(neg_beta_x)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
        soft_min_val = -log_sum_exp / soft_beta
        
        # We want to maximize soft_min, so minimize -soft_min
        return -soft_min_val

    def smooth_grad(pts, r, soft_beta=20.0):
        """Gradient of smooth_objective."""
        if pts.shape != (N, 2):
            return np.zeros_like(pts)
        
        norms = np.linalg.norm(pts, axis=1)
        b_margins = 1.0 - norms - r
        
        p_margins = []
        p_indices = []
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                p_margins.append(d / 2.0 - r)
                p_indices.append((i, j))
        p_margins = np.array(p_margins)
        
        all_margins = np.concatenate([b_margins, p_margins])
        
        neg_beta_x = -soft_beta * all_margins
        max_val = np.max(neg_beta_x)
        exp_vals = np.exp(neg_beta_x - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp  # softmax weights
        
        # Gradient of -soft_min w.r.t. pts
        # d(-soft_min)/d(margin_k) = -weights[k] * (-beta) * (-1) = -weights[k] * beta... 
        # Actually: soft_min = -logsumexp(-beta * m) / beta
        # d(soft_min)/d(m_k) = -1/beta * d/d(m_k)[-logsumexp(-beta*m)]
        # = -1/beta * (-1) * (beta * exp(-beta*m_k) / sum(exp(-beta*m)))
        # = weights[k]
        # So d(-soft_min)/d(m_k) = -weights[k]
        
        # Now chain rule: d(m_k)/d(pts)
        grad = np.zeros_like(pts)
        
        # Boundary margins: m_i = 1 - |c_i| - r
        # d(m_i)/d(c_i) = -c_i / |c_i|
        for i in range(N):
            if norms[i] > 1e-12:
                grad[i] += -weights[i] * (-pts[i] / norms[i])
            # else: gradient is 0 (at origin, use subgradient 0)
        
        # Pairwise margins: m_{ij} = |c_i - c_j|/2 - r
        # d(m_{ij})/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
        # d(m_{ij})/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
        for idx, (i, j) in enumerate(p_indices):
            w = weights[N + idx]
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d > 1e-12:
                grad[i] += -w * (diff / (2.0 * d))
                grad[j] += -w * (-diff / (2.0 * d))
        
        return grad

    def hexagonal_init(spacing):
        """Generate points on a hexagonal lattice clipped to unit circle, pick N closest to origin."""
        points = []
        k = int(1.0 / spacing) + 3
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            return None
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:N]
        return points[indices]

    def ring_init(n_rings, r_start, r_end):
        """Initialize with concentric rings."""
        pts = [[0.0, 0.0]]
        for ring in range(1, n_rings + 1):
            r = r_start + (r_end - r_start) * (ring - 1) / max(n_rings - 1, 1)
            if ring == 1:
                n_pts = 6
            else:
                n_pts = ring * 6
            n_pts = min(n_pts, N - len(pts))
            if n_pts <= 0:
                break
            angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
            for a in angles:
                pts.append([r * np.cos(a), r * np.sin(a)])
        pts = np.array(pts[:N])
        if len(pts) < N:
            return None
        return pts

    # Generate initial candidates
    candidates = []
    
    # Hexagonal grids
    for s in np.linspace(0.10, 0.25, 20):
        pts = hexagonal_init(s)
        if pts is not None:
            candidates.append(pts)
    
    # Ring-based inits
    for n_rings in [5, 6, 7]:
        for r_start in [0.05, 0.1, 0.15]:
            pts = ring_init(n_rings, r_start, 0.9)
            if pts is not None:
                candidates.append(pts)

    # Random inits
    for _ in range(15):
        pts = np.random.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, None] * 0.95
        candidates.append(pts)

    # Evaluate all candidates to find best starting point
    best_pts = None
    best_r = 0.0
    for c in candidates:
        r = compute_radius(c)
        if r > best_r:
            best_r = r
            best_pts = c.copy()

    if best_pts is None:
        best_pts = np.random.rand(N, 2) * 0.8
        best_r = 0.01

    # Binary search on radius with L-BFGS-B optimization
    # Start with a radius we know is feasible, then increase
    
    def optimize_for_radius(r_target, initial_pts, max_iter=2000, soft_beta=20.0):
        """Optimize points to be feasible for radius r_target using smooth objective."""
        x0 = initial_pts.flatten().copy()
        
        def obj(x):
            pts = x.reshape(N, 2)
            return smooth_objective(pts, r_target, soft_beta)
        
        def jac(x):
            pts = x.reshape(N, 2)
            return smooth_grad(pts, r_target, soft_beta)
        
        try:
            result = minimize(obj, x0, method='L-BFGS-B', jac=jac,
                             options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            opt_pts = result.x.reshape(N, 2)
            return opt_pts
        except Exception:
            return initial_pts

    def is_feasible(pts, r, tol=1e-8):
        """Check if pts are feasible for radius r."""
        if pts.shape != (N, 2):
            return False
        if not np.all(np.isfinite(pts)):
            return False
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms > 1.0 - r + tol):
            return False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 2.0 * r - tol:
                    return False
        return True

    # Binary search on radius
    r_low = best_r  # known feasible
    r_high = 0.135  # upper bound (best known is ~0.1258, but let's try higher)
    
    # First, verify r_low is feasible
    if not is_feasible(best_pts, r_low):
        # Adjust r_low down
        r_low = 0.05
        # Find a feasible starting point
        for _ in range(5):
            opt_pts = optimize_for_radius(r_low, best_pts, max_iter=500)
            if is_feasible(opt_pts, r_low):
                best_pts = opt_pts
                break
            r_low *= 0.9
    
    # Binary search
    for _ in range(12):  # ~12 iterations of binary search
        if time.time() > deadline - 15:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible packing for r_mid
        opt_pts = optimize_for_radius(r_mid, best_pts, max_iter=1500, soft_beta=20.0)
        
        if is_feasible(opt_pts, r_mid):
            r_low = r_mid
            best_pts = opt_pts.copy()
            best_r = compute_radius(opt_pts)
            # Increase soft_beta for tighter optimization
        else:
            r_high = r_mid
            # Try with different soft_beta
            opt_pts2 = optimize_for_radius(r_mid, best_pts, max_iter=1000, soft_beta=40.0)
            if is_feasible(opt_pts2, r_mid):
                r_low = r_mid
                best_pts = opt_pts2.copy()
                best_r = compute_radius(opt_pts2)
            else:
                # Try another soft_beta
                opt_pts3 = optimize_for_radius(r_mid, best_pts, max_iter=800, soft_beta=10.0)
                if is_feasible(opt_pts3, r_mid):
                    r_low = r_mid
                    best_pts = opt_pts3.copy()
                    best_r = compute_radius(opt_pts3)
                else:
                    r_high = r_mid
    
    # Final polish: use the best radius found and do a high-precision optimization
    if time.time() < deadline - 10:
        # Do a final optimization at the best found radius with tighter tolerances
        r_final = r_low
        # Try to push slightly higher
        for dr in [0.0001, 0.00005, 0.00002]:
            if time.time() > deadline - 5:
                break
            r_try = r_final + dr
            opt_pts = optimize_for_radius(r_try, best_pts, max_iter=2000, soft_beta=50.0)
            if is_feasible(opt_pts, r_try):
                best_pts = opt_pts
                r_final = r_try
                best_r = compute_radius(opt_pts)
        
        # One more polish at the final radius
        opt_pts = optimize_for_radius(r_final, best_pts, max_iter=3000, soft_beta=100.0)
        if compute_radius(opt_pts) >= best_r - 1e-10:
            best_pts = opt_pts
            best_r = compute_radius(best_pts)

    # Ensure final result is valid
    final_r = compute_radius(best_pts)
    if final_r <= 0:
        # Fallback
        best_pts = np.random.rand(N, 2) * 0.7
        best_pts[0] = [0.0, 0.0]
    
    return best_pts
