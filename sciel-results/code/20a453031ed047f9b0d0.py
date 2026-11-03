import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def compute_radius(C):
        """Compute the actual packing radius for configuration C."""
        if len(C) != N:
            return 0.0
        if not np.all(np.isfinite(C)):
            return 0.0
        
        # Boundary: min_i (1 - |c_i|)
        rad = np.linalg.norm(C, axis=1)
        min_boundary = np.min(1.0 - rad)
        
        # Pairwise: min_{i<j} |c_i - c_j| / 2
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair = np.sqrt(np.min(dists_sq)) / 2.0
        
        return min(min_boundary, min_pair)

    def soft_min_obj(x, beta=50.0):
        """
        Smooth surrogate for maximizing the minimum distance.
        Uses log-sum-exp style soft-min to approximate min of distances.
        
        We want to maximize r = min(pairwise_dists/2, boundary_dists)
        Equivalent to minimizing -r, but we use a smooth version.
        
        For each constraint value v_i (distances), we compute:
        soft_min = -log(sum(exp(-beta * v_i))) / beta
        
        This is a smooth approximation to min(v_i) that becomes exact as beta -> inf.
        """
        C = x.reshape(N, 2)
        
        # Boundary distances: 1 - |c_i|
        rad = np.linalg.norm(C, axis=1)
        boundary_vals = 1.0 - rad
        
        # Pairwise distances / 2
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq) / 2.0
        # Remove diagonal (inf values)
        pair_vals = dists[~np.eye(N, dtype=bool)].flatten()
        
        # Combine all constraint values
        all_vals = np.concatenate([boundary_vals, pair_vals])
        
        # Soft-min using log-sum-exp for numerical stability
        # soft_min = -log(sum(exp(-beta * v))) / beta
        # = -(log(sum(exp(-beta * v)))) / beta
        # For stability: let m = min(all_vals), then:
        # soft_min = m - log(sum(exp(-beta * (v - m)))) / beta
        
        m = np.min(all_vals)
        if m <= 0:
            # If any constraint is violated, penalize heavily
            return -m + 0.01 * np.sum(np.maximum(0, -all_vals)**2)
        
        shifted = -beta * (all_vals - m)
        # Clip for numerical stability
        shifted = np.clip(shifted, -500, 500)
        log_sum = np.log(np.sum(np.exp(shifted)))
        soft_min_val = m - log_sum / beta
        
        # We want to MAXIMIZE soft_min_val, so minimize negative
        return -soft_min_val

    def ring_init(num_rings, r_target):
        """Generate a ring-based initial configuration."""
        pts = []
        
        # Center point
        pts.append([0.0, 0.0])
        
        # Distribute remaining points in rings
        remaining = N - 1
        if num_rings <= 0:
            # All in one ring
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            radius = r_target
            pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
            return np.array(pts)
        
        for ring_idx in range(num_rings):
            # How many points in this ring?
            if ring_idx == num_rings - 1:
                n_ring = remaining
            else:
                # Distribute roughly evenly, outer rings get more
                n_ring = int(remaining * (ring_idx + 1) / num_rings * 1.2)
                n_ring = max(1, min(n_ring, remaining))
            
            if n_ring <= 0:
                continue
                
            # Ring radius: approximate to fit points
            # For n points on a ring of radius R, min distance ~ R * sin(pi/n)
            # We want this to be ~ 2*r_target, so R ~ 2*r_target / sin(pi/n)
            R = min(0.95, 2 * r_target / np.sin(np.pi / n_ring) if n_ring > 2 else 0.5)
            R = max(R, 0.1)
            
            angles = np.linspace(0, 2*np.pi, n_ring, endpoint=False)
            ring_pts = np.column_stack([R * np.cos(angles), R * np.sin(angles)])
            pts.append(ring_pts)
            remaining -= n_ring
            
            if remaining <= 0:
                break
        
        if pts[0] is None or len(pts) == 1:
            # Fallback
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            radius = 0.5
            pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
            return np.array(pts)
        
        # Stack all points
        all_pts = []
        for p in pts:
            if isinstance(p, np.ndarray) and p.ndim == 2:
                all_pts.append(p)
            else:
                all_pts.append(np.array([p]))
        
        result = np.vstack(all_pts)
        
        # Trim or pad to N points
        if len(result) > N:
            result = result[:N]
        elif len(result) < N:
            # Add points near center
            extra = N - len(result)
            angles = np.linspace(0, 2*np.pi, extra, endpoint=False)
            small_r = 0.05
            extra_pts = np.column_stack([small_r * np.cos(angles), small_r * np.sin(angles)])
            result = np.vstack([result, extra_pts])
        
        return result

    def hex_init(r_target):
        """Generate hex lattice initial configuration."""
        s = 2 * r_target
        k = int(1.0 / s) + 2
        pts = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - r_target)**2:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        if len(pts) > N:
            d = np.linalg.norm(pts, axis=1)
            idx = np.argsort(d)[:N]
            pts = pts[idx]
        elif len(pts) < N:
            missing = N - len(pts)
            angles = np.linspace(0, 2*np.pi, missing, endpoint=False)
            radius = 0.3
            extra = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
            pts = np.vstack([pts, extra])
        
        return pts

    def optimize_from_init(C_init, max_iter=300):
        """Optimize starting from C_init."""
        x0 = C_init.flatten()
        
        def obj(x):
            return soft_min_obj(x, beta=50.0)
        
        def grad(x):
            # Numerical gradient
            eps = 1e-7
            g = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                g[i] = (obj(x_plus) - obj(x_minus)) / (2 * eps)
            return g
        
        try:
            res = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                options={
                    'maxiter': max_iter,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 30
                }
            )
            return res.x.reshape(N, 2), res.fun
        except Exception:
            return C_init, np.inf

    def polish(C, iterations=200):
        """High-precision polish with tighter tolerances."""
        x0 = C.flatten()
        
        def obj(x):
            return soft_min_obj(x, beta=100.0)
        
        try:
            res = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                options={
                    'maxiter': iterations,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 50
                }
            )
            return res.x.reshape(N, 2)
        except Exception:
            return C

    # Try multiple initializations
    best_C = None
    best_r = 0.0
    
    inits = []
    
    # Ring-based inits with different structures
    for num_rings in [3, 4, 5]:
        for r_target in [0.12, 0.125, 0.13]:
            C_init = ring_init(num_rings, r_target)
            inits.append(C_init)
    
    # Hex init
    C_hex = hex_init(0.125)
    inits.append(C_hex)
    
    # Random inits
    for _ in range(3):
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.random.uniform(0, 0.8, N)
        C_rand = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(C_rand)
    
    # Optimize from each init
    for i, C_init in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        C_opt, _ = optimize_from_init(C_init, max_iter=200)
        r_opt = compute_radius(C_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_C = C_opt.copy()
    
    # Polish the best result
    if best_C is not None and time.time() < deadline - 5:
        best_C = polish(best_C, iterations=150)
        best_r_final = compute_radius(best_C)
        if best_r_final > best_r:
            best_r = best_r_final
    
    return best_C
