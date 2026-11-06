import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 8
    
    def compute_score(pts):
        """Compute Delta/n^n where Delta is product of ordered pair distances
        after scaling to diameter 2. Returns -inf for invalid inputs."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, 2):
            return -np.inf
        
        # Check distinctness
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(n, dtype=bool)
        d_sq = dists_sq[mask]
        
        if np.any(d_sq < 1e-12):
            return -np.inf
        
        diam_sq = np.max(d_sq)
        if diam_sq < 1e-12:
            return -np.inf
        
        # Scale to diameter 2: multiply by 2/sqrt(diam_sq)
        scale = 2.0 / np.sqrt(diam_sq)
        pts_scaled = pts * scale
        
        # Recalculate squared distances after scaling
        diff_s = pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :]
        dists_sq_s = np.sum(diff_s**2, axis=2)
        d_sq_s = dists_sq_s[mask]
        
        # Product of all ordered pair distances
        # log(Delta) = sum_{i!=j} log(d_ij) = sum_{i!=j} 0.5*log(d_ij^2)
        log_delta = 0.5 * np.sum(np.log(d_sq_s))
        score = np.exp(log_delta) / (n ** n)
        
        return score
    
    def neg_log_score(pts):
        """
        Negative log of the score, for minimization.
        Score = Delta / n^n = (prod d_ij) / n^n
        After scaling to diameter 2, d_ij_scaled = d_ij * 2/D
        So Delta = prod (d_ij * 2/D) = (prod d_ij) * (2/D)^{n(n-1)}
        log(Delta) = sum log(d_ij) + n(n-1)*log(2) - n(n-1)*log(D)
        Since D = max(d_ij), and we want to maximize Delta, we minimize -log(Delta).
        But note: the scaling is fixed by the diameter.
        
        Actually, let's re-derive carefully.
        Let D = max_{i<j} d_ij.
        Scaled points: p_i' = p_i * (2/D).
        Scaled distances: d_ij' = d_ij * (2/D).
        Delta = prod_{i!=j} d_ij' = prod_{i!=j} (d_ij * 2/D) = (prod_{i!=j} d_ij) * (2/D)^{n(n-1)}
        log(Delta) = sum_{i!=j} log(d_ij) + n(n-1)*log(2) - n(n-1)*log(D)
        
        We want to maximize log(Delta), i.e., minimize -log(Delta).
        -log(Delta) = -sum_{i!=j} log(d_ij) - n(n-1)*log(2) + n(n-1)*log(D)
        
        Since n(n-1)*log(2) is constant, we minimize:
        f = -sum_{i!=j} log(d_ij) + n(n-1)*log(D)
        
        sum_{i!=j} log(d_ij) = 2 * sum_{i<j} log(d_ij)
        
        So f = -2 * sum_{i<j} log(d_ij) + n(n-1)*log(D)
        
        Note: log(D) = 0.5 * log(D^2). And D^2 = max_{i<j} d_ij^2.
        
        This function is not smooth at points where the max distance changes,
        but it's well-defined.
        """
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, 2):
            return 1e10
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(n, dtype=bool)
        d_sq = dists_sq[mask]
        
        if np.any(d_sq < 1e-12):
            return 1e10
        
        diam_sq = np.max(d_sq)
        if diam_sq < 1e-12:
            return 1e10
        
        # sum_{i<j} log(d_ij) = 0.5 * sum_{i<j} log(d_ij^2)
        # sum_{i!=j} log(d_ij) = 2 * sum_{i<j} log(d_ij) = sum_{i<j} log(d_ij^2)
        # Wait: sum_{i!=j} log(d_ij) = sum_{i<j} [log(d_ij) + log(d_ji)] = sum_{i<j} 2*log(d_ij)
        # So -sum_{i!=j} log(d_ij) = -2 * sum_{i<j} log(d_ij) = -sum_{i<j} log(d_ij^2)
        
        log_sum_d_sq = np.sum(np.log(d_sq))  # This is sum_{i!=j} log(d_ij^2) = 2 * sum_{i!=j} log(d_ij)
        # Actually, d_sq has n*(n-1) entries (ordered pairs).
        # sum_{i!=j} log(d_ij^2) = log_sum_d_sq
        # sum_{i!=j} log(d_ij) = 0.5 * log_sum_d_sq
        
        # f = -sum_{i!=j} log(d_ij) + n(n-1)*log(D)
        #   = -0.5 * log_sum_d_sq + n(n-1) * 0.5 * log(diam_sq)
        #   = 0.5 * [-log_sum_d_sq + n(n-1)*log(diam_sq)]
        
        val = 0.5 * (-log_sum_d_sq + n*(n-1)*np.log(diam_sq))
        return val
    
    def regular_octagon():
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def cage_config(r1, r2, theta):
        """Two concentric squares with radii r1, r2 and relative rotation theta."""
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = theta + angles1
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([pts1, pts2])
    
    # Generate diverse initial configurations
    def generate_starts():
        starts = []
        # Regular octagon
        starts.append(regular_octagon())
        
        # Various cage configurations
        r1_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
        r2_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
        theta_vals = [0, np.pi/16, np.pi/12, np.pi/8, np.pi/6, np.pi/5, np.pi/4, np.pi/3, np.pi/2]
        
        for r1 in r1_vals:
            for r2 in r2_vals:
                for theta in theta_vals:
                    starts.append(cage_config(r1, r2, theta))
        
        # Add some random perturbations of the octagon
        rng = np.random.default_rng(12345)
        for _ in range(30):
            base = regular_octagon()
            noise = rng.normal(0, 0.5, size=(n, 2))
            starts.append(base + noise)
        
        # Add configurations with different symmetries
        # 4 points on inner ring, 4 on outer, but not necessarily squares
        for inner_r in [0.3, 0.5, 0.7, 0.9]:
            for outer_r in [0.8, 1.0, 1.2, 1.5, 1.8]:
                for rot in [0, np.pi/8, np.pi/6, np.pi/4, np.pi/3]:
                    inner_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2]) + rot
                    outer_angles = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
                    p_in = np.column_stack([inner_r * np.cos(inner_angles), inner_r * np.sin(inner_angles)])
                    p_out = np.column_stack([outer_r * np.cos(outer_angles), outer_r * np.sin(outer_angles)])
                    starts.append(np.vstack([p_in, p_out]))
        
        return starts
    
    def polish(pts0, maxiter=10000, ftol=1e-15, method='L-BFGS-B'):
        """Polish a configuration."""
        x0 = pts0.flatten()
        
        def obj(x):
            pts = x.reshape(n, 2)
            return neg_log_score(pts)
        
        try:
            if method == 'L-BFGS-B':
                res = minimize(obj, x0, method='L-BFGS-B',
                              options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-12})
            elif method == 'SLSQP':
                res = minimize(obj, x0, method='SLSQP',
                              options={'maxiter': maxiter, 'ftol': ftol})
            elif method == 'BFGS':
                res = minimize(obj, x0, method='BFGS',
                              options={'maxiter': maxiter, 'gtol': 1e-14})
            else:
                return pts0
            
            if res.fun < 1e9 and np.isfinite(res.fun):
                return res.x.reshape(n, 2)
        except Exception:
            pass
        return pts0
    
    def try_polish(pts, current_best_score, label=""):
        """Try multiple polish methods and return best result."""
        best_local = pts.copy()
        best_local_score = current_best_score
        
        for method in ['L-BFGS-B', 'SLSQP', 'BFGS']:
            if time.time() > deadline - 3:
                break
            polished = polish(pts, maxiter=8000, ftol=1e-15, method=method)
            s = compute_score(polished)
            if s > best_local_score:
                best_local_score = s
                best_local = polished.copy()
        
        return best_local, best_local_score
    
    # Generate all starting configurations
    all_starts = generate_starts()
    
    # Evaluate all starts quickly
    best_pts = all_starts[0]
    best_score = compute_score(best_pts)
    
    scored_starts = []
    for cfg in all_starts:
        if time.time() > deadline - 40:
            break
        s = compute_score(cfg)
        scored_starts.append((s, cfg))
        if s > best_score:
            best_score = s
            best_pts = cfg.copy()
    
    # Sort starts by score and polish the top ones
    scored_starts.sort(key=lambda x: -x[0])
    
    # Polish top 20 starts
    for i, (s, cfg) in enumerate(scored_starts[:20]):
        if time.time() > deadline - 25:
            break
        polished, ps = try_polish(cfg, max(best_score, s))
        if ps > best_score:
            best_score = ps
            best_pts = polished
    
    # Differential evolution for global search
    if time.time() < deadline - 30:
        def de_obj(x):
            pts = x.reshape(n, 2)
            return neg_log_score(pts)
        
        bounds = [(-3, 3)] * (2 * n)
        
        try:
            de_res = differential_evolution(de_obj, bounds, maxiter=150,
                                            popsize=25, tol=1e-12,
                                            seed=42, polish=False,
                                            updating='deferred',
                                            mutation=(0.5, 1.5),
                                            recombination=0.8,
                                            init='sobol')
            
            if de_res.fun < 1e9 and np.isfinite(de_res.fun):
                pts_de = de_res.x.reshape(n, 2)
                score_de = compute_score(pts_de)
                if score_de > best_score:
                    best_score = score_de
                    best_pts = pts_de
        except Exception:
            pass
    
    # Final polish of the best result found
    if time.time() < deadline - 5:
        polished, ps = try_polish(best_pts, best_score)
        if ps > best_score:
            best_score = ps
            best_pts = polished
    
    # Final fallback
    if best_pts is None:
        best_pts = regular_octagon()
        best_score = compute_score(best_pts)
    
    # Ensure coordinates are within bounds
    best_pts = np.clip(best_pts, -1e6, 1e6)
    
    return best_pts.tolist()
