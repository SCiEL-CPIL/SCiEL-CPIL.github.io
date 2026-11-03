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
        rad = np.linalg.norm(C, axis=1)
        min_boundary = np.min(1.0 - rad)
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        return min(min_boundary, min_pair)

    def soft_violation(C, r):
        """Returns sum of squared violations for radius r."""
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_viol = np.maximum(0, 2.0 * r - dist)
        pair_cost = np.sum(pair_viol ** 2)
        rad = np.linalg.norm(C, axis=1)
        bound_viol = np.maximum(0, rad - (1.0 - r))
        bound_cost = np.sum(bound_viol ** 2)
        return pair_cost + bound_cost

    def get_ring_init():
        """Generate ring-based initialization matching known packing topology: 1+6+12+18+13"""
        # Ring radii (tuned to give good initial spacing)
        ring_radii = [0.0, 0.35, 0.65, 0.85, 0.98]
        ring_counts = [1, 6, 12, 18, 13]
        
        pts = []
        for ri, (radius, count) in enumerate(zip(ring_radii, ring_counts)):
            if count == 1:
                pts.append([0.0, 0.0])
            else:
                angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
                # Add slight angular offset for alternating rings to break symmetry
                offset = 0.0 if ri % 2 == 0 else np.pi / count
                angles = angles + offset
                x = radius * np.cos(angles)
                y = radius * np.sin(angles)
                for xi, yi in zip(x, y):
                    pts.append([xi, yi])
        
        pts = np.array(pts)
        return pts

    def get_hex_init():
        """Generate hex lattice initialization."""
        s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
        k = int(1.0 / s) + 2
        pts = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    pts.append((x, y))
        pts = np.array(pts)
        if len(pts) > N:
            d = np.linalg.norm(pts, axis=1)
            idx = np.argsort(d)[:N]
            pts = pts[idx]
        elif len(pts) < N:
            missing = N - len(pts)
            angles = np.linspace(0, 2 * np.pi, missing, endpoint=False)
            extra = np.c_[0.95 * np.cos(angles), 0.95 * np.sin(angles)]
            pts = np.vstack([pts, extra])
        return pts

    def polish(C, r, max_iter=2000, ftol=1e-12):
        """Polish configuration C for radius r using L-BFGS-B."""
        x0 = C.flatten()
        limit = 1.0 - r
        bounds = [(-limit, limit)] * (2 * N)
        
        def obj(x):
            Cx = x.reshape(N, 2)
            return soft_violation(Cx, r)
        
        try:
            res = minimize(
                obj, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-10, 'maxls': 30}
            )
            return res.x.reshape(N, 2), res.fun
        except Exception:
            return C, np.inf

    def try_radius(r, init_C):
        """Try to find a valid configuration for radius r."""
        C_opt, val = polish(init_C, r)
        # Check feasibility with a relaxed threshold
        if val < 1e-8:
            # Verify actual radius
            actual_r = compute_radius(C_opt)
            if actual_r >= r - 1e-9:
                return C_opt, actual_r
        return None, 0.0

    # Generate initial configurations
    C_ring = get_ring_init()
    C_hex = get_hex_init()
    
    # Start with the better initialization
    r_test = 0.12
    C_ring_opt, r_ring = try_radius(r_test, C_ring)
    C_hex_opt, r_hex = try_radius(r_test, C_hex)
    
    if C_ring_opt is not None and C_hex_opt is not None:
        C_best = C_ring_opt if r_ring >= r_hex else C_hex_opt
    elif C_ring_opt is not None:
        C_best = C_ring_opt
    elif C_hex_opt is not None:
        C_best = C_hex_opt
    else:
        C_best = C_ring  # Fallback
    
    best_r = compute_radius(C_best)
    
    # Binary search for max radius
    r_low = 0.12
    r_high = 0.126
    best_r = compute_radius(C_best)
    
    num_steps = 25
    for i in range(num_steps):
        if time.time() > deadline:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try both initializations
        C_try1, r1 = try_radius(r_mid, C_best)
        if C_try1 is not None:
            C_best = C_try1
            best_r = r1
            r_low = r_mid
        else:
            C_try2, r2 = try_radius(r_mid, get_ring_init())
            if C_try2 is not None:
                C_best = C_try2
                best_r = r2
                r_low = r_mid
            else:
                r_high = r_mid
    
    # Final high-precision polish on the best configuration
    if time.time() < deadline - 5.0:
        final_r = compute_radius(C_best)
        C_best, _ = polish(C_best, final_r, max_iter=5000, ftol=1e-14)
        final_r2 = compute_radius(C_best)
        if final_r2 > best_r:
            best_r = final_r2
    
    return C_best
