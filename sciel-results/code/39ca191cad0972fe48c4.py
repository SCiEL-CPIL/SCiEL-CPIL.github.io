import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), offset=0.0):
        if n == 1:
            return np.array([center])
        angles = np.linspace(offset, 2 * np.pi + offset, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _bottleneck_repair(points, max_iter=50, tol=1e-10):
        """
        Directed local search: Identify the worst triangle and try to move its vertices
        to increase its area, while maintaining high scores for other triangles.
        This is a 'combinatorial' repair step distinct from gradient-based smoothing.
        """
        pts = _normalize_points(points.copy())
        best_score = compute_score(pts)
        
        for _ in range(max_iter):
            if time.time() > start_time + time_limit - 5:
                break
                
            # Find worst triangle
            x = pts[:, 0]
            y = pts[:, 1]
            v1x = x[tri_j] - x[tri_i]
            v1y = y[tri_j] - y[tri_i]
            v2x = x[tri_k] - x[tri_i]
            v2y = y[tri_k] - y[tri_i]
            cross = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * cross
            min_idx = np.argmin(areas)
            min_area = areas[min_idx]
            
            i, j, k = tri_indices[min_idx]
            pi, pj, pk = pts[i], pts[j], pts[k]
            
            # Try to expand this triangle
            # Strategy: Move each vertex slightly away from the centroid of the other two
            improved = False
            for scale in [0.05, 0.02, 0.01, 0.005]:
                # Move i away from jk
                centroid_jk = (pj + pk) / 2.0
                dir_i = pi - centroid_jk
                norm_i = np.linalg.norm(dir_i)
                if norm_i > 1e-9:
                    dir_i = dir_i / norm_i
                    trial = pts.copy()
                    trial[i] = pi + dir_i * scale
                    s = compute_score(_normalize_points(trial))
                    if s > best_score + tol:
                        pts = _normalize_points(trial)
                        best_score = s
                        improved = True
                        break
                
                # Move j away from ik
                centroid_ik = (pi + pk) / 2.0
                dir_j = pj - centroid_ik
                norm_j = np.linalg.norm(dir_j)
                if norm_j > 1e-9:
                    dir_j = dir_j / norm_j
                    trial = pts.copy()
                    trial[j] = pj + dir_j * scale
                    s = compute_score(_normalize_points(trial))
                    if s > best_score + tol:
                        pts = _normalize_points(trial)
                        best_score = s
                        improved = True
                        break
                        
                # Move k away from ij
                centroid_ij = (pi + pj) / 2.0
                dir_k = pk - centroid_ij
                norm_k = np.linalg.norm(dir_k)
                if norm_k > 1e-9:
                    dir_k = dir_k / norm_k
                    trial = pts.copy()
                    trial[k] = pk + dir_k * scale
                    s = compute_score(_normalize_points(trial))
                    if s > best_score + tol:
                        pts = _normalize_points(trial)
                        best_score = s
                        improved = True
                        break
            
            if not improved:
                # If simple expansion fails, try random perturbations of the worst triangle vertices
                rng = np.random.RandomState(int(time.time() * 1000) % 2**31)
                for _ in range(5):
                    trial = pts.copy()
                    idxs = [i, j, k]
                    idx = rng.choice(idxs)
                    trial[idx] += rng.randn(2) * 0.02
                    s = compute_score(_normalize_points(trial))
                    if s > best_score + tol:
                        pts = _normalize_points(trial)
                        best_score = s
                        improved = True
                        break
                if not improved:
                    break
                    
        return pts, best_score

    def _powell_polish(points):
        """Standard Powell polish for final refinement."""
        def obj(x):
            pts = x.reshape(-1, 2)
            return -compute_score(_normalize_points(pts))
        
        try:
            res = opt.minimize(
                obj,
                points.flatten(),
                method='Powell',
                options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            return candidate, score
        except Exception:
            return points, compute_score(points)

    # Generate diverse initial configurations for DE population
    rng = np.random.RandomState(42)
    pop_size = 30
    population = []
    
    # Seed with structured configs
    base_configs = []
    # 1. Regular 13-gon
    base_configs.append(_regular_polygon(13, 1.0))
    # 2. 12 outer + 1 center
    base_configs.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    # 3. 7 outer + 6 inner
    base_configs.append(np.vstack([_regular_polygon(7, 1.0), _regular_polygon(6, 0.5, offset=np.pi/6)]))
    # 4. 6 outer + 6 inner + 1 center
    base_configs.append(np.vstack([_regular_polygon(6, 1.0), _regular_polygon(6, 0.5, offset=np.pi/6), np.array([[0.0, 0.0]])]))
    
    for cfg in base_configs:
        population.append(_normalize_points(cfg.copy()))
        
    # Fill rest with random perturbations of bases
    while len(population) < pop_size:
        base = rng.choice(base_configs)
        perturbed = base + rng.randn(*base.shape) * 0.2
        population.append(_normalize_points(perturbed))
        
    # Evaluate initial population
    scores = np.array([compute_score(p) for p in population])
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_points = population[best_idx].copy()
    
    # Differential Evolution Parameters
    F = 0.8  # Mutation factor
    CR = 0.9  # Crossover rate
    max_generations = 50
    
    for gen in range(max_generations):
        if time.time() > start_time + time_limit - 10:
            break
            
        # Sort population by score (descending)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_pop = [population[i] for i in sorted_indices]
        sorted_scores = scores[sorted_indices]
        
        # Update best
        if sorted_scores[0] > best_score:
            best_score = sorted_scores[0]
            best_points = sorted_pop[0].copy()
            
        # Generate new population
        new_pop = [sorted_pop[0].copy()] # Elite preservation
        new_scores = [sorted_scores[0]]
        
        for i in range(1, pop_size):
            # Select 3 distinct random individuals from sorted pop
            a, b, c = rng.choice(pop_size, 3, replace=False)
            
            # Mutation
            mutant = sorted_pop[a] + F * (sorted_pop[b] - sorted_pop[c])
            
            # Crossover
            trial = np.zeros_like(mutant)
            j_rand = rng.randint(0, N * 2)
            for j in range(N * 2):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = sorted_pop[i][j]
            
            trial_pts = _normalize_points(trial.reshape(-1, 2))
            trial_score = compute_score(trial_pts)
            
            # Selection
            if trial_score > sorted_scores[i]:
                new_pop.append(trial_pts)
                new_scores.append(trial_score)
            else:
                new_pop.append(sorted_pop[i])
                new_scores.append(sorted_scores[i])
                
        population = new_pop
        scores = np.array(new_scores)
        
        # Periodic local repair on top candidates
        if gen % 5 == 0:
            top_k = 5
            top_indices = np.argsort(scores)[::-1][:top_k]
            for idx in top_indices:
                if time.time() > start_time + time_limit - 5:
                    break
                repaired, rep_score = _bottleneck_repair(population[idx], max_iter=20)
                if rep_score > scores[idx]:
                    population[idx] = repaired
                    scores[idx] = rep_score
                    if rep_score > best_score:
                        best_score = rep_score
                        best_points = repaired.copy()

    # Final polish on the best point found
    if time.time() < start_time + time_limit - 5:
        polished, pol_score = _powell_polish(best_points)
        if pol_score > best_score:
            best_points = polished
            best_score = pol_score
            
    # One last bottleneck repair on the very best
    if time.time() < start_time + time_limit - 2:
        final_pts, final_score = _bottleneck_repair(best_points, max_iter=50)
        if final_score > best_score:
            best_points = final_pts
            best_score = final_score

    return best_points
