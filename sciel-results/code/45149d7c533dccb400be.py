"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as optimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin against 150s limit
    
    # Precompute all triples
    triples = list(combinations(range(N), 3))
    idx_i = np.array([t[0] for t in triples])
    idx_j = np.array([t[1] for t in triples])
    idx_k = np.array([t[2] for t in triples])
    n_triples = len(triples)
    
    def _compute_areas(x):
        """Vectorized triangle area computation."""
        x_i = x[idx_i]
        x_j = x[idx_j]
        x_k = x[idx_k]
        v1 = x_j - x_i
        v2 = x_k - x_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)
    
    def _hull_area(x):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(x)
            return hull.volume
        except Exception:
            return 0.0
    
    def score(x_flat):
        """
        Returns the minimum triangle area divided by convex hull area.
        Higher is better.
        """
        x = x_flat.reshape(-1, 2)
        hull_area = _hull_area(x)
        if hull_area < 1e-10:
            return 0.0
        areas = _compute_areas(x)
        min_area = np.min(areas)
        return min_area / hull_area
    
    def soft_min_score(x_flat, beta):
        """
        Soft-min version of the score for smoother optimization.
        Uses log-sum-exp trick for numerical stability.
        beta controls sharpness: large beta -> exact min, small beta -> smoother.
        """
        x = x_flat.reshape(-1, 2)
        hull_area = _hull_area(x)
        if hull_area < 1e-10:
            return 0.0
        areas = _compute_areas(x)
        
        # Soft min using -logsumexp(-areas/beta)/beta
        scaled = -areas / beta
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -beta * log_sum_exp
        
        return soft_min / hull_area
    
    def annealed_objective(x_flat, beta):
        """
        Objective for minimization with annealing.
        """
        return -soft_min_score(x_flat, beta)
    
    def regular_polygon(n=N, cx=0.0, cy=0.0, r=1.0, phase=0.0):
        """Generate points on a regular polygon."""
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + phase
        return np.column_stack([cx + r*np.cos(angles), cy + r*np.sin(angles)])
    
    def random_ring(n=N, r_min=0.8, r_max=1.0):
        """Generate random points on a ring."""
        angles = np.sort(np.random.uniform(0, 2*np.pi, n))
        radii = np.random.uniform(r_min, r_max, n)
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    def two_ring(n=N, n_inner=5, r_inner=0.5, r_outer=1.0):
        """Generate points on two concentric rings."""
        n_outer = n - n_inner
        angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + np.pi/n_outer
        x_inner = np.column_stack([r_inner*np.cos(angles_inner), r_inner*np.sin(angles_inner)])
        x_outer = np.column_stack([r_outer*np.cos(angles_outer), r_outer*np.sin(angles_outer)])
        return np.vstack([x_inner, x_outer])
    
    def optimize_with_annealing(x_start, beta_start=1.0, beta_end=0.01, 
                                  n_anneal_steps=20, max_iter_per_step=500):
        """
        Perform optimization with annealing of the soft-min parameter.
        """
        x_curr = x_start.copy()
        beta_curr = beta_start
        
        for step in range(n_anneal_steps):
            if time.time() > deadline - 5:
                break
            
            t = step / max(n_anneal_steps - 1, 1)
            beta = beta_start * (beta_end / beta_start) ** t
            
            def obj(x_flat):
                return annealed_objective(x_flat, beta)
            
            try:
                res = optimize.minimize(
                    obj,
                    x_curr,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter_per_step, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
                )
                x_curr = res.x
            except Exception:
                break
        
        return x_curr
    
    def _affine_score(params):
        """
        Score for affine-transformed grid. params = [a, b, c, d] for matrix [[a,b],[c,d]].
        """
        base_points = np.array([
            [-0.5, -1.5], [0.5, -1.5], [1.5, -1.5],
            [-1.5, -0.5], [-0.5, -0.5], [0.5, -0.5], [1.5, -0.5],
            [-1.5, 0.5], [-0.5, 0.5], [0.5, 0.5], [1.5, 0.5],
            [-0.5, 1.5], [0.5, 1.5], [1.5, 1.5]
        ])
        
        a, b, c, d = params
        
        transformed = base_points.copy()
        transformed[:, 0] = a * base_points[:, 0] + b * base_points[:, 1]
        transformed[:, 1] = c * base_points[:, 0] + d * base_points[:, 1]
        
        det = a * d - b * c
        if abs(det) < 1e-10:
            return -1.0
        
        x_flat = transformed.flatten()
        return score(x_flat)
    
    def _optimize_affine():
        """Optimize the affine transformation of the grid configuration."""
        best_affine_score = -np.inf
        best_affine_params = None
        
        initial_guesses = [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.2, 0.0, 1.0],
            [0.9, 0.0, 0.0, 1.1],
            [1.0, 0.3, -0.2, 1.0],
            [0.8, 0.1, 0.1, 1.2],
            [1.1, -0.2, 0.2, 0.9],
        ]
        
        for guess in initial_guesses:
            if time.time() > deadline - 15:
                break
            
            try:
                res = optimize.minimize(
                    lambda p: -_affine_score(p),
                    guess,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                s = -res.fun
                if s > best_affine_score:
                    best_affine_score = s
                    best_affine_params = res.x.copy()
            except Exception:
                pass
        
        return best_affine_score, best_affine_params
    
    def _get_affine_points(params):
        """Get the 13 points from affine-transformed grid."""
        base_points = np.array([
            [-0.5, -1.5], [0.5, -1.5], [1.5, -1.5],
            [-1.5, -0.5], [-0.5, -0.5], [0.5, -0.5], [1.5, -0.5],
            [-1.5, 0.5], [-0.5, 0.5], [0.5, 0.5], [1.5, 0.5],
            [-0.5, 1.5], [0.5, 1.5], [1.5, 1.5]
        ])
        
        a, b, c, d = params
        transformed = base_points.copy()
        transformed[:, 0] = a * base_points[:, 0] + b * base_points[:, 1]
        transformed[:, 1] = c * base_points[:, 0] + d * base_points[:, 1]
        return transformed
    
    def _differential_evolution(x_start, pop_size=30, max_iter=200, F=0.8, CR=0.9):
        """
        Simple differential evolution for global search.
        x_start: initial population center
        """
        x_start = np.asarray(x_start, dtype=float).flatten()
        dim = len(x_start)
        
        # Initialize population around x_start
        pop = np.zeros((pop_size, dim))
        pop[0] = x_start
        for i in range(1, pop_size):
            pop[i] = x_start + np.random.randn(dim) * 0.1
        
        # Evaluate initial population
        pop_scores = np.array([score(p) for p in pop])
        best_idx = np.argmax(pop_scores)
        best_x = pop[best_idx].copy()
        best_score_val = pop_scores[best_idx]
        
        for gen in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            for i in range(pop_size):
                # Select three distinct random indices
                a, b, c = np.random.choice(pop_size, 3, replace=False)
                
                # Mutation
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                trial = np.zeros(dim)
                j_rand = np.random.randint(0, dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Evaluate
                trial_score = score(trial)
                
                # Selection
                if trial_score >= pop_scores[i]:
                    pop[i] = trial
                    pop_scores[i] = trial_score
                    if trial_score > best_score_val:
                        best_score_val = trial_score
                        best_x = trial.copy()
        
        return best_x, best_score_val
    
    best_x = None
    best_score = -np.inf
    
    # Phase 0: Try affine optimization of grid (fast, 4D search)
    try:
        aff_score, aff_params = _optimize_affine()
        if aff_score > best_score and aff_params is not None:
            best_score = aff_score
            best_x = _get_affine_points(aff_params).flatten()
    except Exception:
        pass
    
    # Phase 1: Annealed optimization from regular polygon
    x0 = regular_polygon(N).flatten()
    try:
        x_annealed = optimize_with_annealing(
            x0, beta_start=0.5, beta_end=0.005, 
            n_anneal_steps=15, max_iter_per_step=300
        )
        s = score(x_annealed)
        if s > best_score:
            best_score = s
            best_x = x_annealed.copy()
    except Exception:
        pass
    
    # Phase 2: Differential evolution for global search
    if time.time() < deadline - 20 and best_x is not None:
        try:
            x_de, s_de = _differential_evolution(best_x, pop_size=25, max_iter=150, F=0.8, CR=0.9)
            if s_de > best_score:
                best_score = s_de
                best_x = x_de.copy()
        except Exception:
            pass
    
    # Phase 3: Direct optimization from the best result
    if time.time() < deadline - 10 and best_x is not None:
        try:
            res = optimize.minimize(
                lambda x: -score(x),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Multiple restarts with annealing
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > deadline - 8:
            break
        
        if i == 0:
            x_start = (regular_polygon(N, phase=np.random.uniform(0, 2*np.pi/N)) + 
                       np.random.randn(N, 2) * 0.05).flatten()
        elif i == 1:
            x_start = two_ring(N, n_inner=5, r_inner=0.4, r_outer=1.0).flatten()
        elif i == 2:
            x_start = two_ring(N, n_inner=6, r_inner=0.6, r_outer=1.0).flatten()
        elif i == 3:
            x_start = random_ring(N, r_min=0.85, r_max=1.0).flatten()
        elif i == 4:
            if best_x is not None:
                x_start = (best_x.reshape(-1, 2) + np.random.randn(N, 2) * 0.03).flatten()
            else:
                continue
        elif i == 5:
            angles = np.linspace(0, 2*np.pi, N, endpoint=False) + np.random.randn(N) * 0.1
            radii = 1.0 + np.random.randn(N) * 0.05
            x_start = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)]).flatten()
        elif i == 6:
            base = regular_polygon(N)
            r = np.linalg.norm(base, axis=1)
            theta = np.arctan2(base[:, 1], base[:, 0])
            r_pert = r * (1 + 0.1 * np.sin(3*theta + np.random.uniform(0, 2*np.pi)))
            x_start = np.column_stack([r_pert*np.cos(theta), r_pert*np.sin(theta)]).flatten()
        else:
            angles = np.sort(np.random.uniform(0, 2*np.pi, N))
            radii = np.sqrt(np.random.uniform(0.3, 1.0, N))
            x_start = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)]).flatten()
        
        try:
            x_annealed = optimize_with_annealing(
                x_start, beta_start=0.3, beta_end=0.01,
                n_anneal_steps=10, max_iter_per_step=200
            )
            s = score(x_annealed)
            if s > best_score:
                best_score = s
                best_x = x_annealed.copy()
        except Exception:
            pass
        
        if time.time() < deadline - 5 and best_x is not None and s > best_score - 0.01:
            try:
                res = optimize.minimize(
                    lambda x: -score(x),
                    x_annealed,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                s2 = -res.fun
                if s2 > best_score:
                    best_score = s2
                    best_x = res.x.copy()
            except Exception:
                pass
    
    # Final polish
    if time.time() < deadline - 3 and best_x is not None:
        try:
            res = optimize.minimize(
                lambda x: -score(x),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    if best_x is None:
        best_x = regular_polygon(N).flatten()
    
    return best_x.reshape(-1, 2)

# EVOLVE-BLOCK-END
