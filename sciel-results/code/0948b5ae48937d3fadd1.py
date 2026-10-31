import numpy as np
import time

N = 11

# Vertices of the equilateral triangle
# (0,0), (1,0), (0.5, sqrt(3)/2)
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])

def bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian coordinates.
    Point = u*V0 + v*V1 + (1-u-v)*V2
    Note: Standard barycentric often uses weights (w0, w1, w2) summing to 1.
    Here we use (u, v) where u is weight for V1, v is weight for V2, and (1-u-v) for V0.
    Wait, let's be consistent.
    Let's define barycentric coords (alpha, beta) such that:
    P = alpha * V1 + beta * V2 + (1 - alpha - beta) * V0
    Since V0 is (0,0), P = alpha * V1 + beta * V2.
    Constraints: alpha >= 0, beta >= 0, alpha + beta <= 1.
    """
    alpha = bary[:, 0]
    beta = bary[:, 1]
    # P = alpha * V1 + beta * V2
    # V1 = (1, 0), V2 = (0.5, sqrt(3)/2)
    x = alpha * 1.0 + beta * 0.5
    y = alpha * 0.0 + beta * (np.sqrt(3) / 2.0)
    return np.column_stack((x, y))

def compute_min_area(cart_points):
    """
    Compute the minimum area of any triangle formed by 3 of the points.
    """
    n = len(cart_points)
    min_area = np.inf
    # Precompute coordinates for speed
    x = cart_points[:, 0]
    y = cart_points[:, 1]
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
                area = 0.5 * abs(x[i] * (y[j] - y[k]) + 
                                 x[j] * (y[k] - y[i]) + 
                                 x[k] * (y[i] - y[j]))
                if area < min_area:
                    min_area = area
    return min_area

def objective(bary_population):
    """
    Objective function for Differential Evolution.
    Returns an array of costs (negative of min area) for the population.
    """
    costs = np.zeros(len(bary_population))
    for i, bary in enumerate(bary_population):
        cart = bary_to_cart(bary)
        min_area = compute_min_area(cart)
        costs[i] = -min_area
    return costs

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for safety
    
    # Differential Evolution Parameters
    pop_size = 60
    dim = N * 2  # 11 points * 2 barycentric coords
    F = 0.7  # Mutation factor
    CR = 0.9  # Crossover rate
    max_generations = 1000  # Will be cut off by time limit
    
    # Initialize population
    # Random points inside the triangle
    # u, v such that u>=0, v>=0, u+v<=1
    # We can sample u, v from uniform [0,1] and reject if u+v>1, or use a transformation.
    # Transformation: u = r1 * s, v = r2 * (1-s) where r1, r2, s are uniform?
    # Simpler: Sample u, v uniform [0,1]. If u+v > 1, reflect or clip.
    # Reflection: if u+v > 1, swap to keep in bounds? 
    # Let's just use rejection sampling for initialization to ensure validity.
    
    def random_bary_point():
        while True:
            u = np.random.random()
            v = np.random.random()
            if u + v <= 1.0:
                return np.array([u, v])
                
    population = np.array([np.concatenate([random_bary_point() for _ in range(N)]) for _ in range(pop_size)])
    
    # Evaluate initial population
    current_costs = objective(population)
    best_idx = np.argmin(current_costs)
    best_bary = population[best_idx].copy()
    best_cost = current_costs[best_idx]
    
    # Differential Evolution Loop
    generation = 0
    while generation < max_generations:
        if time.time() > deadline:
            break
        
        for i in range(pop_size):
            if time.time() > deadline:
                break
            
            # Select 3 distinct random individuals different from i
            while True:
                a, b, c = np.random.choice(pop_size, 3, replace=False)
                if a != i and b != i and c != i:
                    break
            
            # Mutation: v = x_a + F * (x_b - x_c)
            mutant = population[a] + F * (population[b] - population[c])
            
            # Boundary Handling: Clip to [0,1] for u,v, then ensure u+v<=1
            # Clipping u,v to [0,1] might result in u+v > 1.
            # If u+v > 1, we need to project back into the triangle.
            # Simple projection: scale u,v such that u+v=1? That puts it on the edge.
            # Better: if u+v > 1, reduce both proportionally or reflect.
            # Let's use a simple rejection/reflection strategy for the mutant.
            # If invalid, generate a new random valid point? Or clip and fix?
            # Clipping u,v to [0,1] is safe for individual coords.
            # Then if u+v > 1, we can set v = 1-u (if u<1) or u=1-v.
            # This pushes to the boundary, which is acceptable but might reduce exploration.
            # Alternative: If u+v > 1, reflect across the line u+v=1?
            # Reflection of (u,v) across u+v=1 is (1-v, 1-u).
            # Check: (1-v) + (1-u) = 2 - (u+v). If u+v>1, is 2-(u+v) < 1?
            # If u+v = 1.5, new sum = 0.5. Valid.
            # If u+v = 0.5, we don't reflect.
            
            mutant_u = mutant[:, 0]
            mutant_v = mutant[:, 1]
            
            # Clip to [0,1]
            mutant_u = np.clip(mutant_u, 0.0, 1.0)
            mutant_v = np.clip(mutant_v, 0.0, 1.0)
            
            # Check triangle constraint u + v <= 1
            sum_uv = mutant_u + mutant_v
            invalid_mask = sum_uv > 1.0
            
            if np.any(invalid_mask):
                # Reflect invalid points
                # For points where u+v > 1, reflect to (1-v, 1-u)
                # Note: 1-v and 1-u will be >= 0 because v,u <= 1.
                # And (1-v) + (1-u) = 2 - (u+v) < 1.
                new_u = 1.0 - mutant_v[invalid_mask]
                new_v = 1.0 - mutant_u[invalid_mask]
                mutant_u[invalid_mask] = new_u
                mutant_v[invalid_mask] = new_v
            
            mutant = np.column_stack((mutant_u, mutant_v)).flatten()
            
            # Crossover
            trial = population[i].copy()
            # Determine crossover indices
            j_rand = np.random.randint(0, dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Evaluate trial
            trial_cost = objective(trial.reshape(N, 2))[0]
            
            # Selection
            if trial_cost < current_costs[i]:
                population[i] = trial
                current_costs[i] = trial_cost
                if trial_cost < best_cost:
                    best_cost = trial_cost
                    best_bary = trial.copy()
        
        generation += 1
        
    # Convert best barycentric coordinates to Cartesian
    best_cart = bary_to_cart(best_bary.reshape(N, 2))
    
    return best_cart

# The construct function is the entry point
# We don't need to call it here, the evaluator will do it.
# But we need to make sure it's defined.
