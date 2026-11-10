# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin

    def compute_score(points):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def compute_score_smooth(points, beta=50.0):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        # Use log-sum-exp trick for numerical stability
        neg_d = -d
        max_neg_d = np.max(neg_d)
        soft_min = -1.0 / beta * (max_neg_d + np.log(np.sum(np.exp(neg_d - max_neg_d))))
        max_d = np.max(d)
        soft_max = 1.0 / beta * (max_d + np.log(np.sum(np.exp(d - max_d))))
        if soft_max <= 0:
            return 0.0
        ratio = soft_min / soft_max
        return ratio * ratio

    def coordinate_ascent_optimize(points, max_iter=1000, tol=1e-12, time_limit=None):
        """Iteratively optimize each point while keeping others fixed."""
        pts = points.copy()
        n = len(pts)
        
        for iteration in range(max_iter):
            if time_limit is not None and time.time() > time_limit:
                break
            
            improved = False
            for i in range(n):
                # Fix all other points, optimize point i
                others = np.delete(pts, i, axis=0)
                
                # Try to find a better position for point i
                # Start from current position and do local search
                best_pos = pts[i].copy()
                best_score = compute_score(pts)
                
                # Try small perturbations in different directions
                step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
                for step in step_sizes:
                    # Try 8 directions + stay
                    directions = np.array([
                        [1, 0], [-1, 0], [0, 1], [0, -1],
                        [1, 1], [1, -1], [-1, 1], [-1, -1]
                    ]) * step / np.sqrt(2)
                    
                    for d in directions:
                        new_pos = pts[i] + d
                        new_pts = pts.copy()
                        new_pts[i] = new_pos
                        new_score = compute_score(new_pts)
                        if new_score > best_score + tol:
                            best_score = new_score
                            best_pos = new_pos
                            improved = True
                    
                    # Also try radial moves toward/away from centroid
                    centroid = np.mean(others, axis=0)
                    direction_to_center = centroid - pts[i]
                    norm_dir = np.linalg.norm(direction_to_center)
                    if norm_dir > 1e-10:
                        direction_to_center = direction_to_center / norm_dir
                        for factor in [step, -step]:
                            new_pos = pts[i] + factor * direction_to_center
                            new_pts = pts.copy()
                            new_pts[i] = new_pos
                            new_score = compute_score(new_pts)
                            if new_score > best_score + tol:
                                best_score = new_score
                                best_pos = new_pos
                                improved = True
                
                pts[i] = best_pos
            
            if not improved:
                break
        
        return pts

    def simulated_annealing_optimize(points, max_iter=5000, initial_temp=0.1, cooling_rate=0.9995, time_limit=None):
        """Simulated annealing to escape local optima."""
        pts = points.copy()
        current_score = compute_score(pts)
        best_pts = pts.copy()
        best_score = current_score
        temp = initial_temp
        n = len(pts)
        
        for iteration in range(max_iter):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Random perturbation
            i = np.random.randint(n)
            perturbation = np.random.randn(2) * 0.05
            new_pts = pts.copy()
            new_pts[i] = pts[i] + perturbation
            new_score = compute_score(new_pts)
            
            # Accept or reject
            delta = new_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
            
            temp *= cooling_rate
        
        return best_pts, best_score

    # Generate diverse initial configurations
    configs = []

    # 1. 4x4 grid
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    grid = (grid - 1.5) / 1.5
    configs.append(grid)

    # 2. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))

    # 3. Two concentric squares
    inner_r = 0.5
    outer_r = 1.0
    inner = np.array([[inner_r, inner_r], [-inner_r, inner_r],
                      [-inner_r, -inner_r], [inner_r, -inner_r]])
    outer = []
    for r in [outer_r]:
        for a in np.linspace(0, 2*np.pi, 12, endpoint=False):
            outer.append([r*np.cos(a), r*np.sin(a)])
    configs.append(np.vstack([inner, np.array(outer)]))

    # 4. Two concentric octagons
    inner_a = np.linspace(0, 2*np.pi, 8, endpoint=False)
    outer_a = inner_a + np.pi/8
    inner_r4 = 0.55
    outer_r4 = 1.0
    inner4 = np.column_stack([inner_r4*np.cos(inner_a), inner_r4*np.sin(inner_a)])
    outer4 = np.column_stack([outer_r4*np.cos(outer_a), outer_r4*np.sin(outer_a)])
    configs.append(np.vstack([inner4, outer4]))

    # 5. Center + 5-ring + 10-ring
    c = np.array([[0.0, 0.0]])
    r5 = 0.5
    a5 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    ring5 = np.column_stack([r5*np.cos(a5), r5*np.sin(a5)])
    r10 = 1.0
    a10 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    ring10 = np.column_stack([r10*np.cos(a10), r10*np.sin(a10)])
    configs.append(np.vstack([c, ring5, ring10]))

    # 6. Hexagonal lattice selection
    s = 1.0
    h = s * np.sqrt(3) / 2
    lattice = []
    for row in range(-3, 4):
        for col in range(-3, 4):
            x = col * s + (row % 2) * s / 2
            y = row * h
            lattice.append([x, y])
    lattice = np.array(lattice)
    centroid = np.mean(lattice, axis=0)
    dists = np.sum((lattice - centroid) ** 2, axis=1)
    idx = np.argsort(dists)[:16]
    cfg6 = lattice[idx]
    cfg6 = (cfg6 - np.mean(cfg6, axis=0))
    norm = np.max(np.max(np.abs(cfg6), axis=0))
    if norm > 0:
        cfg6 = cfg6 / norm
    configs.append(cfg6)

    # 7. Two concentric hexagons + extras
    hex_a = np.linspace(0, 2*np.pi, 6, endpoint=False)
    inner_hex_r = 0.5
    outer_hex_r = 1.0
    inner_hex = np.column_stack([inner_hex_r*np.cos(hex_a), inner_hex_r*np.sin(hex_a)])
    outer_hex = np.column_stack([outer_hex_r*np.cos(hex_a + np.pi/6), outer_hex_r*np.sin(hex_a + np.pi/6)])
    extra = np.array([[0, 1.0], [1.0, 0], [0, -1.0], [-1.0, 0.0]])
    configs.append(np.vstack([inner_hex, outer_hex, extra]))

    # 8. Random perturbations of grid
    for seed in range(3):
        np.random.seed(seed * 7 + 1)
        pert = grid + 0.1 * np.random.randn(16, 2)
        pert = (pert - np.mean(pert, axis=0))
        std = np.std(pert)
        if std > 0:
            pert = pert / std
        configs.append(pert)

    # 9. Fibonacci spiral
    n = 16
    golden = (1 + np.sqrt(5)) / 2
    indices = np.arange(n)
    r = np.sqrt(indices / n)
    theta = 2 * np.pi * indices / golden
    fib_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    configs.append(fib_pts)

    # 10. Ring + inner ring
    outer_ring_a = np.linspace(0, 2*np.pi, 12, endpoint=False)
    inner_ring_a = outer_ring_a + np.pi/12
    outer_ring_r = 1.0
    inner_ring_r = 0.45
    outer_ring = np.column_stack([outer_ring_r*np.cos(outer_ring_a), outer_ring_r*np.sin(outer_ring_a)])
    inner_ring = np.column_stack([inner_ring_r*np.cos(inner_ring_a), inner_ring_r*np.sin(inner_ring_a)])
    configs.append(np.vstack([inner_ring, outer_ring]))

    # 11. 3 concentric rings: 4 + 6 + 6
    ring1_a = np.linspace(0, 2*np.pi, 4, endpoint=False)
    ring2_a = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    ring3_a = np.linspace(0, 2*np.pi, 6, endpoint=False)
    r1, r2, r3 = 0.3, 0.6, 1.0
    ring1 = np.column_stack([r1*np.cos(ring1_a), r1*np.sin(ring1_a)])
    ring2 = np.column_stack([r2*np.cos(ring2_a), r2*np.sin(ring2_a)])
    ring3 = np.column_stack([r3*np.cos(ring3_a), r3*np.sin(ring3_a)])
    configs.append(np.vstack([ring1, ring2, ring3]))

    # 12. 2+4+10 configuration
    c2 = np.array([[0, 0], [0.3, 0]])
    ring4_a = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    ring10_a = np.linspace(0, 2*np.pi, 10, endpoint=False)
    ring4 = np.column_stack([0.55*np.cos(ring4_a), 0.55*np.sin(ring4_a)])
    ring10 = np.column_stack([1.0*np.cos(ring10_a), 1.0*np.sin(ring10_a)])
    configs.append(np.vstack([c2, ring4, ring10]))

    # Evaluate initial configs
    best_score = -np.inf
    best_points = None
    for cfg in configs:
        score = compute_score(cfg)
        if score > best_score:
            best_score = score
            best_points = cfg.copy()

    from scipy.optimize import minimize

    # Phase 1: Coordinate ascent from best config
    if best_points is not None and time.time() < deadline - 200:
        try:
            ca_pts = coordinate_ascent_optimize(best_points, max_iter=500, time_limit=deadline - 180)
            ca_score = compute_score(ca_pts)
            if ca_score > best_score:
                best_score = ca_score
                best_points = ca_pts.copy()
        except Exception:
            pass

    # Phase 2: Smooth Nelder-Mead from best config
    if best_points is not None and time.time() < deadline - 180:
        def objective_smooth(pts_flat, beta=50.0):
            pts = pts_flat.reshape(16, 2)
            s = compute_score_smooth(pts, beta)
            if not np.isfinite(s):
                return 1e10
            return -s

        try:
            result = minimize(objective_smooth, best_points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 5000, 'fatol': 1e-14, 'xatol': 1e-14})
            opt_pts = result.x.reshape(16, 2)
            opt_score = compute_score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    # Phase 3: Simulated annealing to escape local optima
    if best_points is not None and time.time() < deadline - 150:
        try:
            sa_pts, sa_score = simulated_annealing_optimize(best_points, max_iter=10000, 
                                                            initial_temp=0.05, cooling_rate=0.9998,
                                                            time_limit=deadline - 130)
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_pts.copy()
        except Exception:
            pass

    # Phase 4: Coordinate ascent after SA
    if best_points is not None and time.time() < deadline - 120:
        try:
            ca_pts = coordinate_ascent_optimize(best_points, max_iter=500, time_limit=deadline - 100)
            ca_score = compute_score(ca_pts)
            if ca_score > best_score:
                best_score = ca_score
                best_points = ca_pts.copy()
        except Exception:
            pass

    # Phase 5: Coarse-to-fine beta annealing
    if best_points is not None and time.time() < deadline - 100:
        betas = [20.0, 40.0, 80.0, 150.0, 300.0]
        for beta in betas:
            if time.time() > deadline - 30:
                break
            def objective_beta(pts_flat):
                pts = pts_flat.reshape(16, 2)
                s = compute_score_smooth(pts, beta)
                if not np.isfinite(s):
                    return 1e10
                return -s

            try:
                result = minimize(objective_beta, best_points.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 3000, 'fatol': 1e-14, 'xatol': 1e-14})
                opt_pts = result.x.reshape(16, 2)
                opt_score = compute_score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
            except Exception:
                pass

    # Phase 6: Random restarts with coordinate ascent
    if time.time() < deadline - 60:
        for seed in range(10):
            if time.time() > deadline - 30:
                break
            np.random.seed(seed * 31 + 17)
            rand_pts = np.random.randn(16, 2)
            rand_pts = rand_pts - np.mean(rand_pts, axis=0)
            std = np.std(rand_pts)
            if std > 0:
                rand_pts = rand_pts / std

            try:
                ca_pts = coordinate_ascent_optimize(rand_pts, max_iter=300, time_limit=deadline - 25)
                ca_score = compute_score(ca_pts)
                if ca_score > best_score:
                    best_score = ca_score
                    best_points = ca_pts.copy()
            except Exception:
                pass

    # Phase 7: Gradient-based refinement using L-BFGS-B with smooth objective
    if best_points is not None and time.time() < deadline - 40:
        def objective_lbfgs(pts_flat, beta=200.0):
            pts = pts_flat.reshape(16, 2)
            s = compute_score_smooth(pts, beta)
            if not np.isfinite(s):
                return 1e10
            return -s

        try:
            result = minimize(objective_lbfgs, best_points.flatten(), method='L-BFGS-B',
                             options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-15})
            opt_pts = result.x.reshape(16, 2)
            opt_score = compute_score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    # Phase 8: High-precision polish with coordinate ascent
    if best_points is not None and time.time() < deadline - 20:
        try:
            ca_pts = coordinate_ascent_optimize(best_points, max_iter=2000, tol=1e-14, time_limit=deadline - 10)
            ca_score = compute_score(ca_pts)
            if ca_score > best_score:
                best_score = ca_score
                best_points = ca_pts.copy()
        except Exception:
            pass

    # Phase 9: Final Nelder-Mead polish
    if best_points is not None and time.time() < deadline - 5:
        def objective_polish(pts_flat):
            pts = pts_flat.reshape(16, 2)
            s = compute_score(pts)
            if not np.isfinite(s):
                return 1e10
            return -s

        try:
            result = minimize(objective_polish, best_points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 10000, 'fatol': 1e-18, 'xatol': 1e-18})
            opt_pts = result.x.reshape(16, 2)
            opt_score = compute_score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = configs[0]

    return best_points

# EVOLVE-BLOCK-END
