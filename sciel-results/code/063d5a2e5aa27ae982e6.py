# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import time
import random

N = 100
CUBE_HALF = 0.5

def _score(c):
    """
    Compute the maximum radius for a given set of centers.
    r = min(min_pairwise_dist / 2, min_boundary_dist)
    """
    if N > 1:
        dists = pdist(c, metric='euclidean')
        min_pair_dist = np.min(dists)
    else:
        min_pair_dist = np.inf
        
    abs_c = np.abs(c)
    boundary_dists = CUBE_HALF - np.max(abs_c, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _get_bottleneck_pairs(c):
    """Return indices of pairs that are closest to the minimum distance."""
    dists = pdist(c, metric='euclidean')
    min_d = np.min(dists)
    # Find all pairs within 10% of minimum
    threshold = min_d * 1.1
    close_mask = dists < threshold
    # Convert condensed indices to pair indices
    n = N
    pairs = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if close_mask[idx]:
                pairs.append((i, j))
            idx += 1
    return pairs

def _initial_fcc_perturbed(scale=0.25, noise=0.02):
    """
    Create FCC-like lattice initial placement with perturbation.
    FCC has higher packing density than simple cubic.
    """
    a = scale
    points = []
    basis = np.array([[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]])
    
    # Generate grid of FCC unit cells covering the cube
    for ix in range(-1, 4):
        for iy in range(-1, 4):
            for iz in range(-1, 4):
                origin = np.array([ix*a, iy*a, iz*a])
                for b in basis:
                    p = origin + b * a
                    if np.all(np.abs(p) <= CUBE_HALF):
                        points.append(p)
    
    points = np.array(points)
    np.random.shuffle(points)
    
    if len(points) >= N:
        pts = points[:N].copy()
    else:
        # Pad with random points
        remaining = N - len(points)
        random_pts = np.random.uniform(-CUBE_HALF, CUBE_HALF, size=(remaining, 3))
        pts = np.vstack([points, random_pts])
    
    # Add small random perturbation to break symmetry
    pts += np.random.normal(0, noise, size=(N, 3))
    pts = np.clip(pts, -CUBE_HALF, CUBE_HALF)
    
    return pts

def _initial_random_clustered():
    """
    Create random initial placement with some clustering to encourage denser packing.
    """
    # Generate clusters
    n_clusters = 8
    cluster_centers = np.random.uniform(-0.4, 0.4, size=(n_clusters, 3))
    pts = []
    per_cluster = N // n_clusters
    remainder = N - per_cluster * n_clusters
    
    for i in range(n_clusters):
        count = per_cluster + (1 if i < remainder else 0)
        cluster_pts = cluster_centers[i] + np.random.normal(0, 0.08, size=(count, 3))
        pts.append(cluster_pts)
    
    pts = np.vstack(pts)
    pts = np.clip(pts, -CUBE_HALF, CUBE_HALF)
    np.random.shuffle(pts)
    return pts[:N]

def _stochastic_local_search(c, time_limit=40, T_start=0.02, T_end=1e-8, alpha=0.997):
    """
    Stochastic Local Search with simulated annealing.
    Uses multiple move strategies: single point, double point, swap, and gradient-based moves.
    """
    start_time = time.time()
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T_start
    step_size = 0.04
    
    # Move type probabilities (will be adaptive)
    p_single = 0.35
    p_double = 0.25
    p_swap = 0.25
    p_gradient = 0.15
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        r = random.random()
        c_new = c.copy()
        
        if r < p_single:
            # Move one point
            idx = random.randint(0, N - 1)
            delta = np.random.randn(3) * step_size
            c_new[idx] = c_new[idx] + delta
            c_new[idx] = np.clip(c_new[idx], -CUBE_HALF, CUBE_HALF)
        elif r < p_single + p_double:
            # Move two points
            i1, i2 = random.sample(range(N), 2)
            c_new[i1] = c_new[i1] + np.random.randn(3) * step_size
            c_new[i2] = c_new[i2] + np.random.randn(3) * step_size
            c_new[i1] = np.clip(c_new[i1], -CUBE_HALF, CUBE_HALF)
            c_new[i2] = np.clip(c_new[i2], -CUBE_HALF, CUBE_HALF)
        elif r < p_single + p_double + p_swap:
            # Swap two points (helps escape local minima)
            i1, i2 = random.sample(range(N), 2)
            c_new[i1], c_new[i2] = c[i2].copy(), c[i1].copy()
        else:
            # Gradient-based move: push apart the closest pair
            pairs = _get_bottleneck_pairs(c)
            if pairs:
                i, j = random.choice(pairs)
                vec = c[i] - c[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    direction = vec / norm
                    push = step_size * 0.5
                    c_new[i] = c_new[i] + direction * push
                    c_new[j] = c_new[j] - direction * push
                    c_new[i] = np.clip(c_new[i], -CUBE_HALF, CUBE_HALF)
                    c_new[j] = np.clip(c_new[j], -CUBE_HALF, CUBE_HALF)
                else:
                    # Coincident, random perturbation
                    c_new[i] = c_new[i] + np.random.randn(3) * step_size
                    c_new[j] = c_new[j] + np.random.randn(3) * step_size
                    c_new[i] = np.clip(c_new[i], -CUBE_HALF, CUBE_HALF)
                    c_new[j] = np.clip(c_new[j], -CUBE_HALF, CUBE_HALF)
            else:
                # Fallback to single point move
                idx = random.randint(0, N - 1)
                delta = np.random.randn(3) * step_size
                c_new[idx] = c_new[idx] + delta
                c_new[idx] = np.clip(c_new[idx], -CUBE_HALF, CUBE_HALF)
        
        new_score = _score(c_new)
        
        # Acceptance criterion
        if new_score >= current_score:
            c = c_new
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_c = c.copy()
        else:
            delta = new_score - current_score
            if delta > -100 * T:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    c = c_new
                    current_score = new_score
            no_improve_count += 1
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
        
        # Adaptive step size: decrease when stuck, increase when exploring
        if no_improve_count > max_no_improve:
            step_size = max(0.01, step_size * 1.5)  # Increase step to escape
            no_improve_count = 0
        else:
            step_size = max(0.005, 0.04 * (T / T_start) ** 0.4)
        
        # Adaptive move probabilities: increase swap/gradient when stuck
        if no_improve_count > max_no_improve // 2:
            p_single = 0.2
            p_double = 0.2
            p_swap = 0.35
            p_gradient = 0.25
        else:
            p_single = 0.35
            p_double = 0.25
            p_swap = 0.25
            p_gradient = 0.15
    
    return best_c, best_score

def _polish(c, time_limit=15):
    """
    Final polishing: coordinate descent with small steps.
    """
    start_time = time.time()
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for step_size in step_sizes:
        while time.time() - start_time < time_limit:
            improved = False
            for i in range(N):
                # Try 6 axis-aligned directions
                for dx, dy, dz in [
                    (step_size, 0, 0), (-step_size, 0, 0),
                    (0, step_size, 0), (0, -step_size, 0),
                    (0, 0, step_size), (0, 0, -step_size),
                ]:
                    c_test = c.copy()
                    c_test[i] = c[i] + np.array([dx, dy, dz])
                    c_test[i] = np.clip(c_test[i], -CUBE_HALF, CUBE_HALF)
                    s_test = _score(c_test)
                    if s_test > current_score + 1e-12:
                        c = c_test
                        current_score = s_test
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_c = c.copy()
                        break
                if improved:
                    break
            if not improved:
                break
    
    return best_c, best_score

def construct():
    """
    Construct the optimal arrangement of 100 equal spheres in a unit cube.
    """
    best_c = None
    best_score = -np.inf
    
    # Time budget: 120s total. Leave 5s for overhead.
    total_budget = 115
    
    # 1. FCC start with perturbation
    c_fcc = _initial_fcc_perturbed(scale=0.25, noise=0.02)
    c_best, s_best = _stochastic_local_search(c_fcc, time_limit=30)
    if s_best > best_score:
        best_score = s_best
        best_c = c_best
    
    # 2. Another FCC with different parameters
    c_fcc2 = _initial_fcc_perturbed(scale=0.28, noise=0.03)
    c_best, s_best = _stochastic_local_search(c_fcc2, time_limit=30)
    if s_best > best_score:
        best_score = s_best
        best_c = c_best
    
    # 3. Random clustered start
    c_clustered = _initial_random_clustered()
    c_best, s_best = _stochastic_local_search(c_clustered, time_limit=25)
    if s_best > best_score:
        best_score = s_best
        best_c = c_best
    
    # 4. Pure random start
    c_rand = np.random.uniform(-CUBE_HALF, CUBE_HALF, size=(N, 3))
    c_best, s_best = _stochastic_local_search(c_rand, time_limit=20)
    if s_best > best_score:
        best_score = s_best
        best_c = c_best
    
    # 5. Final refinement on the best solution
    if best_c is not None:
        c_polish, s_polish = _polish(best_c, time_limit=15)
        if s_polish > best_score:
            best_score = s_polish
            best_c = c_polish
    
    return best_c
# EVOLVE-BLOCK-END
