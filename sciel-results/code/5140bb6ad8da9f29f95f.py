import numpy as np
import scipy.optimize as opt
import scipy.spatial as spat
import time
import itertools

N = 14

def _hull_area(points):
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _score(points):
    """Exact score: min triangle area / hull area."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for degeneracy
    if np.any(np.diff(np.sort(points[:,0])) < 1e-9) and np.any(np.diff(np.sort(points[:,1])) < 1e-9):
         # This is a weak check, ConvexHull handles it better
         pass

    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    
    # Compute all triangle areas vectorized
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_points = None
    best_score = 0.0
    
    # 1. Generate initial population
    # Strategy: Mix of regular polygons, random points in unit disk, and ring structures
    population_size = 50
    pop = []
    
    # Add regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    poly = np.column_stack((np.cos(angles), np.sin(angles)))
    pop.append(poly.copy())
    
    # Add random points in unit disk
    for _ in range(population_size - 1):
        r = np.sqrt(np.random.rand(N))
        theta = np.random.rand(N) * 2 * np.pi
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        # Ensure non-degenerate by adding small noise if needed, but random is usually fine
        pop.append(pts)
        
    # Add a few structured ring configs to seed the search
    # 7 inner, 7 outer
    r1, r2 = 0.5, 1.0
    a1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    a2 = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    ring1 = np.vstack([r1*np.cos(a1), r1*np.sin(a1)]).T
    ring2 = np.vstack([r2*np.cos(a2), r2*np.sin(a2)]).T
    pop.append(np.vstack([ring1, ring2]))
    
    # Evaluate initial population
    scores = np.array([_score(p) for p in pop])
    best_idx = np.argmax(scores)
    best_points = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    # 2. Differential Evolution
    # Parameters
    F = 0.5  # Scale factor
    CR = 0.9  # Crossover rate
    max_iter = 1000
    
    # Since we have 50 points and 28 dimensions, DE is feasible.
    # We need to be careful with the time budget.
    # _score takes ~0.5ms. 50 evals per gen = 25ms.
    # 1000 gens = 25s. This fits well within 140s.
    
    # We will implement a simple DE/rand/1/bin loop
    for gen in range(max_iter):
        if time.time() > deadline - 10:
            break
            
        for i in range(population_size):
            # Select 3 distinct indices different from i
            a, b, c = np.random.choice([j for j in range(population_size) if j != i], 3, replace=False)
            
            # Mutation: v = pop[a] + F * (pop[b] - pop[c])
            v = pop[a] + F * (pop[b] - pop[c])
            
            # Crossover: binomial
            j_rand = np.random.randint(0, 2*N)
            trial = np.zeros(2*N)
            for j in range(2*N):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = pop[i][j]
            
            trial_pts = trial.reshape(N, 2)
            
            # Evaluate trial
            trial_score = _score(trial_pts)
            
            # Selection
            if trial_score > scores[i]:
                pop[i] = trial_pts
                scores[i] = trial_score
                if trial_score > best_score:
                    best_score = trial_score
                    best_points = trial_pts.copy()
                    
        # Elitism: Keep the best point in the population
        best_in_pop_idx = np.argmax(scores)
        pop[0] = pop[best_in_pop_idx]
        scores[0] = scores[best_in_pop_idx]
        
        # Optional: Local refinement on the best point every few generations
        if gen % 10 == 0 and time.time() < deadline - 20:
             # Quick coordinate descent polish on best_points
             current = best_points.copy()
             current_score = best_score
             step = 0.01
             for _ in range(5): # 5 rounds of CD
                 improved = False
                 for p_idx in range(N):
                     for c in range(2):
                         for d in [-1, 1]:
                             new_p = current.copy()
                             new_p[p_idx, c] += d * step
                             new_s = _score(new_p)
                             if new_s > current_score:
                                 current = new_p
                                 current_score = new_s
                                 improved = True
                 if not improved:
                     break
             if current_score > best_score:
                 best_score = current_score
                 best_points = current.copy()
                 # Update population with improved best
                 pop[0] = best_points.copy()
                 scores[0] = best_score

    return best_points

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    pts = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_score(pts)}")
