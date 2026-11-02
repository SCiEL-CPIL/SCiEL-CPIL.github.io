import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def get_dims(t):
        t = np.clip(t, 0.15, 0.85)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def compute_penalty_vectorized(t, circles, mu=10.0):
        """
        Vectorized penalty computation using only Delaunay-relevant pairs.
        Returns (penalty, score, is_valid)
        """
        w, h = get_dims(t)
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        
        # Boundary violations
        viol_left = np.maximum(0, r - x)
        viol_right = np.maximum(0, x - (w - r))
        viol_bottom = np.maximum(0, r - y)
        viol_top = np.maximum(0, y - (h - r))
        
        boundary_penalty = np.sum(viol_left**2) + np.sum(viol_right**2) + np.sum(viol_bottom**2) + np.sum(viol_top**2)
        
        # Use Delaunay triangulation to find relevant pairs
        # Add small jitter to avoid degenerate triangulations
        pts = np.column_stack([x, y]) + np.random.randn(N, 2) * 1e-6
        
        try:
            tri = Delaunay(pts)
            # Extract unique edges from triangles
            simplices = tri.simplices
            edges = set()
            for simplex in simplices:
                for i in range(3):
                    a = simplex[i]
                    b = simplex[(i + 1) % 3]
                    if a < b:
                        edges.add((a, b))
                    else:
                        edges.add((b, a))
        except Exception:
            # Fallback: use all pairs
            edges = {(i, j) for i in range(N) for j in range(i + 1, N)}
        
        # Compute overlap penalty only for Delaunay-relevant pairs
        overlap_penalty = 0.0
        for (i, j) in edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            sum_r = r[i] + r[j]
            if dist_sq < sum_r**2:
                dist = np.sqrt(max(dist_sq, 1e-12))
                overlap = sum_r - dist
                overlap_penalty += overlap**2
        
        penalty = boundary_penalty + overlap_penalty
        score = np.sum(r)
        is_valid = penalty < 1e-7
        return penalty, score, is_valid

    def objective(vars):
        """
        Objective function for scipy minimize.
        vars: [t, x_0...x_N-1, y_0...y_N-1, r_0...r_N-1]
        Minimize penalty - mu * sum(r)
        """
        t = vars[0]
        circles = vars[1:].reshape(N, 3)
        
        # Ensure radii are positive
        circles[:, 2] = np.maximum(0.01, circles[:, 2])
        
        penalty, score, _ = compute_penalty_vectorized(t, circles)
        
        return penalty - 10.0 * score

    def physics_relax(t, circles, n_iter=2000, dt=0.005):
        """
        Physics-based relaxation using Lennard-Jones-like forces.
        Returns relaxed circles.
        """
        w, h = get_dims(t)
        circles = circles.copy()
        
        for iteration in range(n_iter):
            if time.time() > deadline - 5:
                break
                
            forces = np.zeros_like(circles)
            x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
            
            # Compute forces from overlaps and boundaries
            for i in range(N):
                # Boundary forces
                if x[i] < r[i]:
                    forces[i, 0] += (r[i] - x[i]) * 100
                if x[i] > w - r[i]:
                    forces[i, 0] -= (x[i] - (w - r[i])) * 100
                if y[i] < r[i]:
                    forces[i, 1] += (r[i] - y[i]) * 100
                if y[i] > h - r[i]:
                    forces[i, 1] -= (y[i] - (h - r[i])) * 100
                
                # Repulsion from other circles
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(max(dist_sq, 1e-12))
                    sum_r = r[i] + r[j]
                    
                    if dist < sum_r:
                        # Strong repulsion when overlapping
                        force_mag = (sum_r - dist) * 50 / max(dist, 0.01)
                        fx = force_mag * dx / max(dist, 0.01)
                        fy = force_mag * dy / max(dist, 0.01)
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
            
            # Update positions
            circles[:, :2] += forces[:, :2] * dt
            
            # Clamp to valid range
            circles[:, 0] = np.clip(circles[:, 0], 0, w)
            circles[:, 1] = np.clip(circles[:, 1], 0, h)
            circles[:, 2] = np.maximum(0.01, circles[:, 2])
            
            # Decay time step
            dt *= 0.999
            
            # Check convergence
            if iteration % 100 == 0:
                penalty, score, is_valid = compute_penalty_vectorized(t, circles)
                if is_valid:
                    break
        
        return circles

    def initial_grid(t):
        w, h = get_dims(t)
        circles = np.zeros((N, 3))
        # Use a 5x5 grid, fill 21
        side = 5
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                cx = (i + 0.5) * (w / side)
                cy = (j + 0.5) * (h / side)
                r = 0.45 * min(w, h) / side
                circles[count, 0] = cx
                circles[count, 1] = cy
                circles[count, 2] = r
                count += 1
        return circles

    best_score = -np.inf
    best_t = 0.5
    best_circles = None
    
    # Try multiple aspect ratios
    t_values = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    
    random.seed(42)
    np.random.seed(42)
    
    for t in t_values:
        if time.time() > deadline - 15:
            break
            
        circles = initial_grid(t)
        
        # Phase 1: Physics-based relaxation
        relaxed_circles = physics_relax(t, circles, n_iter=1000)
        
        # Phase 2: Scipy optimization from relaxed state
        init_vars = np.concatenate(([t], relaxed_circles.flatten()))
        
        try:
            result = minimize(
                objective,
                init_vars,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 3000}
            )
        except Exception:
            result = None
        
        if result is not None and result.success:
            opt_vars = result.x
            opt_t = opt_vars[0]
            opt_circles = opt_vars[1:].reshape(N, 3)
            opt_circles[:, 2] = np.maximum(0.01, opt_circles[:, 2])
            
            penalty, score, is_valid = compute_penalty_vectorized(opt_t, opt_circles)
            
            if is_valid and score > best_score:
                best_score = score
                best_t = opt_t
                best_circles = opt_circles.copy()
        
        # Phase 3: Try physics relaxation alone (sometimes works better)
        penalty, score, is_valid = compute_penalty_vectorized(t, relaxed_circles)
        if is_valid and score > best_score:
            best_score = score
            best_t = t
            best_circles = relaxed_circles.copy()
        
        # Phase 4: Polish with more iterations if time permits
        if best_score > 1.0 and time.time() < deadline - 20:
            polished = physics_relax(t, best_circles, n_iter=500, dt=0.002)
            penalty, score, is_valid = compute_penalty_vectorized(t, polished)
            if is_valid and score > best_score:
                best_score = score
                best_t = t
                best_circles = polished.copy()
        
        # Phase 5: Final scipy polish
        if best_circles is not None and time.time() < deadline - 10:
            init_vars = np.concatenate(([best_t], best_circles.flatten()))
            try:
                result = minimize(
                    objective,
                    init_vars,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 2000}
                )
                if result.success:
                    opt_vars = result.x
                    opt_t = opt_vars[0]
                    opt_circles = opt_vars[1:].reshape(N, 3)
                    opt_circles[:, 2] = np.maximum(0.01, opt_circles[:, 2])
                    
                    penalty, score, is_valid = compute_penalty_vectorized(opt_t, opt_circles)
                    if is_valid and score > best_score:
                        best_score = score
                        best_t = opt_t
                        best_circles = opt_circles.copy()
            except Exception:
                pass
    
    # Fallback if we didn't find a valid solution
    if best_circles is None or best_score < 0.5:
        t = 0.5
        circles = initial_grid(t)
        relaxed = physics_relax(t, circles, n_iter=500)
        penalty, score, is_valid = compute_penalty_vectorized(t, relaxed)
        if is_valid:
            best_score = score
            best_t = t
            best_circles = relaxed
        else:
            # Last resort: use initial grid
            penalty, score, is_valid = compute_penalty_vectorized(t, circles)
            if is_valid:
                best_score = score
                best_t = t
                best_circles = circles
            else:
                # Return something valid even if not optimal
                best_circles = circles
                best_t = t

    return best_circles
