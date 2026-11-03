"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

# Triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2])
H = np.sqrt(3) / 2.0

def point_in_triangle(p, a=V1, b=V2, c=V3):
    """Check if point p is inside or on the triangle abc."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def project_to_triangle(p):
    """Project point p to the nearest point inside the triangle."""
    # Simple projection logic for equilateral triangle
    # If outside, move towards centroid until inside
    centroid = (V1 + V2 + V3) / 3.0
    if point_in_triangle(p):
        return p
    # Bisection on t in [0, 1] for centroid + t*(p - centroid)
    t_low, t_high = 0.0, 1.0
    for _ in range(20):
        t_mid = (t_low + t_high) / 2.0
        p_mid = centroid + t_mid * (p - centroid)
        if point_in_triangle(p_mid):
            t_low = t_mid
        else:
            t_high = t_mid
    return centroid + t_low * (p - centroid)

def triangle_area(p1, p2, p3):
    """Area of triangle formed by three points."""
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

def min_triangle_area(points):
    """Minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def valid_points(points):
    """Ensure all points are strictly inside (with small margin) or on boundary."""
    margin = 1e-9
    valid = np.zeros(len(points), dtype=bool)
    for i, p in enumerate(points):
        # Check barycentric coordinates or simple sign check
        # Using point_in_triangle is robust
        if point_in_triangle(p):
            valid[i] = True
        else:
            valid[i] = False
    return valid

def construct():
    np.random.seed(42)
    
    # Differential Evolution Parameters
    pop_size = 50
    max_generations = 100
    F = 0.8  # Mutation scale
    CR = 0.9  # Crossover rate
    
    # Initialize population
    # Use a mix of random and lattice-like starts to ensure diversity
    population = []
    for i in range(pop_size):
        if i == 0:
            # Start with a reasonable lattice
            pts = []
            # 3x3 grid + 2 extra? No, 11 points.
            # Let's just use random for DE, it's robust.
            pass
        
        pts = []
        for _ in range(N):
            # Random point in triangle
            # Use rejection sampling or barycentric
            r1 = np.random.rand()
            r2 = np.random.rand()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            p = V1 + r1 * (V2 - V1) + r2 * (V3 - V1)
            pts.append(p)
        population.append(np.array(pts))
    
    # Evaluate initial population
    scores = np.array([min_triangle_area(pop) for pop in population])
    
    for gen in range(max_generations):
        new_population = []
        new_scores = []
        
        for i in range(pop_size):
            # Select 3 distinct individuals
            idxs = np.random.choice(pop_size, size=4, replace=False)
            a = population[idxs[0]]
            b = population[idxs[1]]
            c = population[idxs[2]]
            target = population[i]
            
            # Mutation
            mutant = a + F * (b - c)
            
            # Crossover
            trial = target.copy()
            j_rand = np.random.randint(0, N)
            for j in range(N):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Constraint: Project points back if outside
            for j in range(N):
                if not point_in_triangle(trial[j]):
                    trial[j] = project_to_triangle(trial[j])
            
            # Evaluate
            trial_score = min_triangle_area(trial)
            
            # Selection
            if trial_score >= scores[i]:
                new_population.append(trial)
                new_scores.append(trial_score)
            else:
                new_population.append(target)
                new_scores.append(scores[i])
        
        population = new_population
        scores = np.array(new_scores)
        
        # Optional: Local polish on the best individual every few generations
        # To save time, only do it at the end or when improvement stalls
        if gen % 20 == 19:
            best_idx = np.argmax(scores)
            best_pop = population[best_idx].copy()
            
            # Simple local search: perturb each point slightly and accept if better
            for _ in range(50): # 50 local steps
                improved = False
                for j in range(N):
                    old_p = best_pop[j].copy()
                    # Try 4 small moves
                    for dx in [-0.01, 0.01]:
                        for dy in [-0.01, 0.01]:
                            new_p = old_p + np.array([dx, dy])
                            if point_in_triangle(new_p):
                                trial_pop = best_pop.copy()
                                trial_pop[j] = new_p
                                trial_score = min_triangle_area(trial_pop)
                                if trial_score > scores[best_idx]:
                                    best_pop[j] = new_p
                                    scores[best_idx] = trial_score
                                    improved = True
                if not improved:
                    break
            
            population[best_idx] = best_pop
            scores[best_idx] = min_triangle_area(best_pop)

    # Final Polish: Nelder-Mead on the best individual
    best_idx = np.argmax(scores)
    best_pop = population[best_idx].copy()
    
    # Flatten for optimizer
    def objective(flat_points):
        pts = flat_points.reshape(N, 2)
        # Check validity
        for p in pts:
            if not point_in_triangle(p):
                return -1e10
        return -min_triangle_area(pts) # Minimize negative area
    
    x0 = best_pop.flatten()
    # Nelder-Mead can be slow, use it carefully
    # We'll do a limited version
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if res.fun > -1e9: # Valid
        final_pop = res.x.reshape(N, 2)
        # Ensure strictly inside
        for i in range(N):
            if not point_in_triangle(final_pop[i]):
                final_pop[i] = project_to_triangle(final_pop[i])
        
        final_score = min_triangle_area(final_pop)
        if final_score > scores[best_idx]:
            best_pop = final_pop
            scores[best_idx] = final_score

    return best_pop
# EVOLVE-BLOCK-END
