import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 115.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, k):
            pass
# Correct triplet generation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _greedy_init(rng):
    """
    Greedy initialization: place points one by one to maximize the minimum distance to existing points.
    """
    pts = []
    p0 = _random_point_in_triangle(rng)
    pts.append(np.array(p0))
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        for _ in range(300):
            cand = np.array(_random_point_in_triangle(rng))
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts.append(best_pos)
    
    return np.array(pts)

def _structured_init():
    """Create a structured initialization based on known Heilbronn-like configurations."""
    # Vertices, edge midpoints, and interior points
    pts = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
        [0.5, S3H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.125, S3H/4], [0.875, S3H/4]
    ]
    return np.array(pts)

def _local_refine(pts, rng, time_budget=5.0):
    """
    Refine a single configuration using Nelder-Mead on the true min-area objective.
    """
    start_t = time.time()
    
    def nm_obj(pts_flat):
        pts_tmp = pts_flat.reshape(N, 2)
        pts_tmp = _project_to_triangle(pts_tmp)
        return -_min_triangle_area(pts_tmp)
    
    try:
        result = minimize(
            nm_obj,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if result.success:
            pts_opt = result.x.reshape(N, 2)
            pts_opt = _project_to_triangle(pts_opt)
            
            if _check_inside(pts_opt):
                return pts_opt
    except Exception:
        pass
    
    return pts

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Differential Evolution (DE) as the global search method, followed by local refinement.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # Initial Population
    pop_size = 20
    population = []
    
    # Add some diverse initializers
    population.append(_greedy_init(rng))
    population.append(_structured_init())
    
    for _ in range(pop_size - 2):
        p = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        p = _project_to_triangle(p)
        population.append(p)
    
    # Evaluate initial population
    scores = [_min_triangle_area(p) for p in population]
    best_idx = np.argmax(scores)
    best_pts = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # Differential Evolution Parameters
    F = 0.8  # Differential weight
    CR = 0.9  # Crossover rate
    
    gen = 0
    while time.time() - start_time < DEADLINE - 10:
        gen += 1
        
        for i in range(pop_size):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Select 3 distinct individuals
            idx1, idx2, idx3 = rng.sample([x for x in range(pop_size) if x != i], 3)
            
            # Create mutant vector: v = x[idx1] + F * (x[idx2] - x[idx3])
            mutant = population[idx1] + F * (population[idx2] - population[idx3])
            
            # Crossover
            trial = np.zeros_like(population[i])
            j_rand = rng.randint(0, 2*N - 1)
            for j in range(2 * N):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = population[i][j]
            
            # Repair (Project to triangle)
            trial_pts = trial.reshape(N, 2)
            trial_pts = _project_to_triangle(trial_pts)
            
            # Evaluate trial
            trial_score = _min_triangle_area(trial_pts)
            
            # Selection
            if trial_score > scores[i]:
                population[i] = trial_pts
                scores[i] = trial_score
                
                if trial_score > best_score:
                    best_score = trial_score
                    best_pts = trial_pts.copy()
        
        # Periodic local refinement of the best individual to escape local minima
        if gen % 5 == 0:
            refined_pts = _local_refine(best_pts, rng, time_budget=2.0)
            refined_score = _min_triangle_area(refined_pts)
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts.copy()
                # Inject refined solution into population
                population[0] = best_pts
                scores[0] = best_score
    
    # Final refinement with remaining time
    remaining_time = DEADLINE - 10 - (time.time() - start_time)
    if remaining_time > 2.0:
        best_pts = _local_refine(best_pts, rng, time_budget=remaining_time)
        final_score = _min_triangle_area(best_pts)
        if final_score > best_score:
            best_score = final_score
    
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
