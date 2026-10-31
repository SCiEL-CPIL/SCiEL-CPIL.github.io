"""
Optimized 13-point configuration to maximize minimum triangle area / hull area.
Combines soft-min annealing (inspiration) with strict deadline management and 
multi-restart polishing (current program).
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish by 145s.
    deadline = start_time + 145.0

    # Precompute all triples for vectorized area calculation
    triples = list(combinations(range(N), 3))
    idx_i = np.array([t[0] for t in triples])
    idx_j = np.array([t[1] for t in triples])
    idx_k = np.array([t[2] for t in triples])
    
    def _compute_areas(x_flat):
        """Vectorized triangle area computation."""
        x = x_flat.reshape(-1, 2)
        x_i = x[idx_i]
        x_j = x[idx_j]
        x_k = x[idx_k]
        v1 = x_j - x_i
        v2 = x_k - x_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)

    def _hull_area(x_flat):
        """Compute convex hull area."""
        x = x_flat.reshape(-1, 2)
        try:
            hull = ConvexHull(x)
            return hull.volume
        except Exception:
            return 0.0

    def _hard_score(x_flat):
        """Exact score: min triangle area / hull area."""
        hull_area = _hull_area(x_flat)
        if hull_area < 1e-10:
            return 0.0
        areas = _compute_areas(x_flat)
        min_area = np.min(areas)
        return min_area / hull_area

    def _soft_score(x_flat, beta):
        """
        Soft-min score using log-sum-exp.
        As beta -> 0, this approaches min(areas).
        This smooths the objective landscape for optimization.
        """
        hull_area = _hull_area(x_flat)
        if hull_area < 1e-10:
            return 0.0
        areas = _compute_areas(x_flat)
        # Avoid log(0)
        areas = np.maximum(areas, 1e-12)
        scaled = -areas / beta
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min_val = -beta * log_sum_exp
        return soft_min_val / hull_area

    def _objective_soft(x_flat, beta):
        """Negative soft score for minimization."""
        return -_soft_score(x_flat, beta)

    def _objective_hard(x_flat):
        """Negative hard score for minimization."""
        return -_hard_score(x_flat)

    def regular_polygon(n=N, cx=0.0, cy=0.0, r=1.0, phase=0.0):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + phase
        return np.column_stack([cx + r*np.cos(angles), cy + r*np.sin(angles)])

    def generate_starts():
        """Generate diverse initial configurations."""
        starts = []
        
        # 1. Regular polygon
        starts.append(regular_polygon().flatten())
        
        # 2. Perturbed regular polygons
        rng = np.random.default_rng(42)
        for scale in [0.05, 0.1, 0.2]:
            base = regular_polygon()
            pert = base + rng.normal(0, scale, size=(N, 2))
            starts.append(pert.flatten())
            
        # 3. Two-ring structures (common in optimal packing)
        # e.g., 1 center, 6 inner, 6 outer? Or 0 center, 5 inner, 8 outer?
        # Let's try a few plausible distributions
        configs = [
            (0, 6, 7, 0.5),   # 0 center, 6 inner (r=0.5), 7 outer
            (1, 5, 7, 0.4),   # 1 center, 5 inner (r=0.4), 7 outer
            (0, 5, 8, 0.6),   # 0 center, 5 inner (r=0.6), 8 outer
            (1, 6, 6, 0.5),   # 1 center, 6 inner (r=0.5), 6 outer
        ]
        for n_c, n_i, n_o, r_i in configs:
            pts = []
            if n_c > 0:
                pts.append([0.0, 0.0])
            if n_i > 0:
                angles_i = np.linspace(0, 2*np.pi, n_i, endpoint=False)
                pts.append(np.column_stack([r_i*np.cos(angles_i), r_i*np.sin(angles_i)]))
            if n_o > 0:
                angles_o = np.linspace(0, 2*np.pi, n_o, endpoint=False)
                pts.append(np.column_stack([np.cos(angles_o), np.sin(angles_o)]))
            if pts:
                starts.append(np.vstack(pts).flatten())

        # 4. Random points in unit disk (sorted by angle to reduce symmetry breaking issues)
        for _ in range(5):
            angles = np.sort(rng.uniform(0, 2*np.pi, N))
            radii = np.sqrt(rng.uniform(0.1, 1.0, N))
            pts = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
            starts.append(pts.flatten())
            
        return starts

    def optimize_with_annealing(x_start, beta_start=0.5, beta_end=0.01, n_steps=10):
        """
        Anneal the beta parameter for soft-min score.
        Uses Nelder-Mead for robustness.
        """
        x_curr = x_start.copy()
        for step in range(n_steps):
            if time.time() > deadline - 10.0:
                break
            t = step / max(n_steps - 1, 1)
            # Exponential decay of beta
            beta = beta_start * (beta_end / beta_start) ** t
            
            def obj(x):
                return _objective_soft(x, beta)
                
            try:
                res = minimize(
                    obj, x_curr,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500, 
                        'xatol': 1e-8, 
                        'fatol': 1e-8, 
                        'adaptive': True
                    }
                )
                x_curr = res.x
            except Exception:
                break
        return x_curr

    def polish_hard(x_start, max_iter=1000):
        """
        Final polish using L-BFGS-B on the hard objective.
        Note: Hard objective is non-smooth, but L-BFGS-B often works well 
        if initialized close to the optimum found by annealing.
        """
        if time.time() > deadline - 5.0:
            return x_start
        try:
            res = minimize(
                _objective_hard, 
                x_start,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            return res.x
        except Exception:
            return x_start

    best_x = None
    best_score = -np.inf

    def update_best(x_candidate):
        nonlocal best_x, best_score
        s = _hard_score(x_candidate)
        if s > best_score:
            best_score = s
            best_x = x_candidate.copy()
        return s

    # Phase 1: Generate starts and optimize with annealing
    starts = generate_starts()
    
    for i, x_start in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
        
        # Anneal to find a good local optimum
        x_annealed = optimize_with_annealing(x_start)
        update_best(x_annealed)
        
        # If we have plenty of time, do a quick hard polish on this candidate
        if time.time() < deadline - 15.0:
            x_polished = polish_hard(x_annealed, max_iter=200)
            update_best(x_polished)

    # Phase 2: If we still have time, do more local restarts from the global best
    if best_x is not None and time.time() < deadline - 10.0:
        rng = np.random.default_rng(123)
        # Try 5 more restarts from perturbed best
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            pert = best_x + rng.normal(0, 0.05, size=2*N)
            # Ensure points are well-behaved (e.g., not overlapping too much)
            x_pert = optimize_with_annealing(pert, beta_start=0.2, beta_end=0.01, n_steps=5)
            update_best(x_pert)
            
            if time.time() < deadline - 5.0:
                x_polished = polish_hard(x_pert, max_iter=100)
                update_best(x_polished)

    # Final check: return the best found
    if best_x is None:
        # Fallback to regular polygon if everything failed
        return regular_polygon().flatten()
        
    return best_x

if __name__ == "__main__":
    # Simple self-test
    start = time.time()
    pts = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Score: {_hard_score(pts)}")
