import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 10.0
    N = 16

    def get_ratio(pts):
        """Calculate d_min / d_max. Higher is better."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return d_min / d_max

    def get_cost(pts):
        """Negative ratio for minimization. Lower is better."""
        return -get_ratio(pts)

    def hex_init():
        """Hexagonal arrangement - structurally closer to optimal packing."""
        pts = []
        a = 1.0
        b = np.sqrt(3) / 2
        # Center
        pts.append([0, 0])
        # Ring 1: 6 points
        for i in range(6):
            angle = 2 * np.pi * i / 6
            pts.append([a * np.cos(angle), a * np.sin(angle)])
        # Ring 2: 9 points
        ring2_pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * a + j * 0.5 * a
                y = j * b
                r = np.sqrt(x*x + y*y)
                if 1.5 * a - 0.1 < r < 2.5 * a + 0.1:
                    ring2_pts.append([x, y])
        ring2_pts = np.array(ring2_pts)
        if len(ring2_pts) >= 9:
            r_target = 2.0 * a
            dists_to_target = np.abs(np.linalg.norm(ring2_pts, axis=1) - r_target)
            idx = np.argsort(dists_to_target)[:9]
            ring2_pts = ring2_pts[idx]
        pts = np.array(pts)
        pts = np.vstack([pts, ring2_pts])
        pts = pts - np.mean(pts, axis=0)
        return pts

    def ring_init(n_inner, n_outer, r_inner, r_outer):
        """Points on two concentric rings."""
        pts = []
        if n_inner > 0:
            for i in range(n_inner):
                angle = 2 * np.pi * i / n_inner
                pts.append([r_inner * np.cos(angle), r_inner * np.sin(angle)])
        if n_outer > 0:
            for i in range(n_outer):
                angle = 2 * np.pi * i / n_outer + np.pi / n_outer
                pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
        return np.array(pts)

    def grid_init():
        """4x4 square grid."""
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        pts = np.array(pts, dtype=float)
        pts = pts - np.mean(pts, axis=0)
        return pts

    def circle_init(n=16):
        """Points on a circle."""
        pts = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            pts.append([np.cos(angle), np.sin(angle)])
        return np.array(pts)

    def generate_candidates():
        candidates = []
        candidates.append(ring_init(1, 15, 0, 1.0))
        candidates.append(ring_init(4, 12, 0.5, 1.0))
        candidates.append(ring_init(7, 9, 0.4, 1.0))
        candidates.append(ring_init(9, 7, 0.4, 1.0))
        candidates.append(ring_init(1, 3, 0, 0.5))
        for i, c in enumerate(candidates):
            c = c - np.mean(c, axis=0)
            max_r = np.max(np.linalg.norm(c, axis=1))
            if max_r > 0:
                c = c / max_r
            candidates[i] = c
        candidates.append(grid_init())
        candidates.append(hex_init())
        candidates.append(circle_init(16))
        return candidates

    best_pts = None
    best_ratio = 0.0

    for c in generate_candidates():
        if len(c) != 16:
            continue
        ratio = get_ratio(c)
        if ratio > best_ratio:
            best_ratio = ratio
            best_pts = c.copy()

    # Simulated Annealing - can escape local minima
    def simulated_annealing(pts, t_start, time_budget):
        """Simulated annealing to maximize d_min/d_max."""
        pts = pts.copy()
        current_ratio = get_ratio(pts)
        best_ratio_local = current_ratio
        best_pts_local = pts.copy()
        
        T = 0.1  # Initial temperature
        T_min = 1e-6
        cooling_rate = 0.9995
        step_size = 0.1
        
        t_end = t_start + time_budget
        while time.time() < t_end and T > T_min:
            # Generate perturbation
            i = np.random.randint(N)
            delta = np.random.randn(2) * step_size
            new_pts = pts.copy()
            new_pts[i] += delta
            new_pts = new_pts - np.mean(new_pts, axis=0)
            
            new_ratio = get_ratio(new_pts)
            
            # Accept if better, or with probability exp((new_ratio - current_ratio) / T)
            if new_ratio > current_ratio:
                pts = new_pts
                current_ratio = new_ratio
                if current_ratio > best_ratio_local:
                    best_ratio_local = current_ratio
                    best_pts_local = pts.copy()
            else:
                delta_ratio = new_ratio - current_ratio
                if np.random.rand() < np.exp(delta_ratio / T):
                    pts = new_pts
                    current_ratio = new_ratio
            
            T *= cooling_rate
            # Occasionally increase step size to explore more
            if np.random.rand() < 0.01:
                step_size *= 1.5
            else:
                step_size *= 0.999
        
        return best_pts_local, best_ratio_local

    # Nelder-Mead polish
    def nelder_mead_polish(pts, t_start, time_budget):
        """Polish with Nelder-Mead."""
        def objective(flat_pts):
            pts_2d = flat_pts.reshape(N, 2)
            return -get_ratio(pts_2d)
        
        t_end = t_start + time_budget
        flat = pts.flatten()
        
        try:
            result = minimize(objective, flat, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.success or time.time() > t_end:
                polished = result.x.reshape(N, 2)
                polished = polished - np.mean(polished, axis=0)
                ratio = get_ratio(polished)
                return polished, ratio
        except Exception:
            pass
        
        return pts, get_ratio(pts)

    # Run simulated annealing from best candidate
    np.random.seed(42)
    time_elapsed = time.time() - start_time
    sa_budget = min(180.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN - 30.0)
    if sa_budget > 10:
        sa_pts, sa_ratio = simulated_annealing(best_pts, time.time(), sa_budget)
        if sa_ratio > best_ratio:
            best_pts = sa_pts
            best_ratio = sa_ratio

    # Nelder-Mead polish
    time_elapsed = time.time() - start_time
    nm_budget = min(120.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN - 20.0)
    if nm_budget > 5:
        nm_pts, nm_ratio = nelder_mead_polish(best_pts, time.time(), nm_budget)
        if nm_ratio > best_ratio:
            best_pts = nm_pts
            best_ratio = nm_ratio

    # Multi-start: SA from different initializations
    time_elapsed = time.time() - start_time
    remaining = TIME_LIMIT - time_elapsed - SAFETY_MARGIN - 20.0
    if remaining > 30:
        for c in generate_candidates():
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            if len(c) != 16:
                continue
            sa_pts, sa_ratio = simulated_annealing(c, time.time(), min(30.0, remaining / 5))
            if sa_ratio > best_ratio:
                best_pts = sa_pts
                best_ratio = sa_ratio
                # Polish
                nm_pts, nm_ratio = nelder_mead_polish(best_pts, time.time(), 30.0)
                if nm_ratio > best_ratio:
                    best_pts = nm_pts
                    best_ratio = nm_ratio

    # Final polish
    time_elapsed = time.time() - start_time
    if TIME_LIMIT - time_elapsed - SAFETY_MARGIN > 10:
        nm_pts, nm_ratio = nelder_mead_polish(best_pts, time.time(), 
                                               min(60.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN))
        if nm_ratio > best_ratio:
            best_pts = nm_pts
            best_ratio = nm_ratio

    # Ensure finite and correct shape
    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)) or final_pts.shape != (16, 2):
        final_pts = hex_init()

    return final_pts
