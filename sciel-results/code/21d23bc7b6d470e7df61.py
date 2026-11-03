# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

N = 11

# Triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2])

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

def clip_to_triangle(p):
    """Clip point p to be inside the triangle if it's outside."""
    if point_in_triangle(p):
        return p
    # If outside, project towards centroid or find nearest point on boundary
    # Simple heuristic: move towards centroid until inside
    centroid = (V1 + V2 + V3) / 3
    t = 0.0
    # Binary search for t such that centroid + t*(p-centroid) is inside
    # p is outside, centroid is inside.
    # We want the point on the segment from p to centroid that is on the boundary.
    # Actually, just moving to centroid is safe but loses information.
    # Let's try a few steps towards centroid.
    for step in [0.5, 0.8, 0.95, 1.0]:
        p_new = p + step * (centroid - p)
        if point_in_triangle(p_new):
            return p_new
    return centroid

def triangle_area(p1, p2, p3):
    """Area of triangle formed by three points."""
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

def min_triangle_area(points):
    """Minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        area = triangle_area(points[i], points[j], points[k])
        if area < min_area:
            min_area = area
    return min_area

def evaluate_points(points):
    """Evaluate a set of points, clipping if necessary."""
    clipped = np.array([clip_to_triangle(p) for p in points])
    return min_triangle_area(clipped)

def construct():
    np.random.seed(42)
    
    # Differential Evolution Parameters
    pop_size = 60
    max_iter = 300
    F = 0.8  # Mutation factor
    CR = 0.9 # Crossover rate
    
    # Bounds for each coordinate
    # x in [0, 1], y in [0, sqrt(3)/2]
    lower_bounds = np.zeros(2*N)
    upper_bounds = np.zeros(2*N)
    for i in range(N):
        lower_bounds[2*i] = 0.0
        upper_bounds[2*i] = 1.0
        lower_bounds[2*i+1] = 0.0
        upper_bounds[2*i+1] = np.sqrt(3)/2
        
    # Initialize population
    pop = np.random.uniform(lower_bounds, upper_bounds, size=(pop_size, 2*N))
    # Ensure initial points are valid by clipping
    for i in range(pop_size):
        pts = pop[i].reshape(N, 2)
        clipped = np.array([clip_to_triangle(p) for p in pts])
        pop[i] = clipped.flatten()
        
    # Evaluate initial population
    scores = np.array([evaluate_points(pop[i].reshape(N, 2)) for i in range(pop_size)])
    
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_pop = pop[best_idx].copy()
    
    for gen in range(max_iter):
        for i in range(pop_size):
            # Choose three distinct random indices different from i
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            while r1 == i: r1 = np.random.choice(pop_size)
            while r2 == i or r2 == r1: r2 = np.random.choice(pop_size)
            while r3 == i or r3 == r1 or r3 == r2: r3 = np.random.choice(pop_size)
            
            # Mutation
            v = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Crossover
            trial = np.copy(pop[i])
            j_rand = np.random.randint(0, 2*N)
            for j in range(2*N):
                if np.random.rand() <= CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = pop[i][j]
                    
            # Clip trial points to triangle
            trial_pts = trial.reshape(N, 2)
            trial_clipped = np.array([clip_to_triangle(p) for p in trial_pts])
            trial_flat = trial_clipped.flatten()
            
            # Evaluate trial
            trial_score = evaluate_points(trial_clipped)
            
            # Selection
            if trial_score >= scores[i]:
                pop[i] = trial_flat
                scores[i] = trial_score
                if trial_score > best_score:
                    best_score = trial_score
                    best_pop = trial_flat.copy()
                    
    return best_pop.reshape(N, 2)
# EVOLVE-BLOCK-END
