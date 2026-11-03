"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _validate(x, y, r, w, h, tol=1e-5):
    if w + h > 2.0 + 1e-4:
        return False
    if w < 1e-8 or h < 1e-8:
        return False
    for i in range(len(r)):
        if r[i] < 1e-8:
            return False
        if x[i] - r[i] < -tol or x[i] + r[i] > w + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > h + tol:
            return False
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            rad_sum = r[i] + r[j]
            if dist_sq < rad_sum * rad_sum - tol:
                return False
    return True

def construct():
    deadline = time.time() + 145  # Safety margin
    
    # Variables: x (N), y (N), r (N), w (1)
    # h is derived: h = 2 - w
    # Total variables: 3*N + 1 = 64
    
    n_vars = 3 * N + 1
    
    # Bounds
    # w: [0.5, 1.5] (ensures h in [0.5, 1.5])
    # x: [0, 1.5] (max w)
    # y: [0, 1.5] (max h)
    # r: [0, 0.75] (max radius in 1.5 box)
    
    bounds = []
    for _ in range(N): bounds.append((0, 1.5)) # x
    for _ in range(N): bounds.append((0, 1.5)) # y
    for _ in range(N): bounds.append((1e-5, 0.75)) # r
    bounds.append((0.5, 1.5)) # w
    
    def fitness(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        w = v[3*N]
        h = 2.0 - w
        
        # Check perimeter constraint
        if w + h > 2.0 + 1e-6:
            return -1e10
        
        penalty = 0.0
        lam = 1000.0
        
        # Boundary violations
        for i in range(N):
            # x
            if x[i] - r[i] < 0:
                penalty += lam * (x[i] - r[i]) ** 2
            if x[i] + r[i] > w:
                penalty += lam * (x[i] + r[i] - w) ** 2
            # y
            if y[i] - r[i] < 0:
                penalty += lam * (y[i] - r[i]) ** 2
            if y[i] + r[i] > h:
                penalty += lam * (y[i] + r[i] - h) ** 2
            # r
            if r[i] < 1e-5:
                penalty += lam * (r[i] - 1e-5) ** 2
        
        # Overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                rad_sum = r[i] + r[j]
                if dist_sq < rad_sum * rad_sum:
                    gap = rad_sum - np.sqrt(dist_sq)
                    penalty += lam * gap ** 2
        
        return -np.sum(r) + penalty

    def repair(v):
        """Clamp variables to bounds and ensure w+h=2"""
        x = np.clip(v[:N], 0, 1.5)
        y = np.clip(v[N:2*N], 0, 1.5)
        r = np.clip(v[2*N:3*N], 1e-5, 0.75)
        w = np.clip(v[3*N], 0.5, 1.5)
        return np.concatenate([x, y, r, [w]])

    # Initialize population
    pop_size = 40
    pop = []
    for _ in range(pop_size):
        v = np.array([random.uniform(lo, hi) for lo, hi in bounds])
        # Bias initial w towards 1.0 (square)
        if random.random() < 0.5:
            v[3*N] = random.uniform(0.8, 1.2)
        pop.append(repair(v))
    
    # Evaluate initial population
    scores = [fitness(v) for v in pop]
    
    best_val = -np.inf
    best_v = None
    
    def update_best(v, score):
        nonlocal best_val, best_v
        if score > best_val:
            # Validate to ensure we don't keep invalid high-scoring penalty solutions
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            w = v[3*N]
            h = 2.0 - w
            if _validate(x, y, r, w, h):
                best_val = -score  # score is negative sum r + penalty
                best_v = v.copy()
    
    for v, s in zip(pop, scores):
        update_best(v, s)
        
    # Differential Evolution Parameters
    F = 0.8  # Scale factor
    CR = 0.9  # Crossover rate
    
    # DE Loop
    while time.time() < deadline - 5:
        new_pop = []
        for i in range(pop_size):
            # Select 3 distinct parents
            a, b, c = random.sample([j for j in range(pop_size) if j != i], 3)
            
            # Mutation
            mutant = pop[a] + F * (pop[b] - pop[c])
            
            # Crossover
            trial = np.empty(n_vars)
            j_rand = random.randint(0, n_vars - 1)
            for j in range(n_vars):
                if random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = pop[i][j]
            
            trial = repair(trial)
            trial_score = fitness(trial)
            
            # Selection
            if trial_score <= scores[i]:
                new_pop.append(trial)
                new_scores = trial_score
            else:
                new_pop.append(pop[i])
                new_scores = scores[i]
            
            # Keep track of best in new pop
            if new_scores < scores[i]: # Lower is better (negative sum r)
                scores[i] = new_scores
                update_best(new_pop[-1], new_scores)
            
            # If trial was better, update pop and scores
            if trial_score <= scores[i]:
                 pop[i] = new_pop[-1]
            else:
                 pop[i] = pop[i] # unchanged
            
            # Wait, standard DE replaces pop[i] if trial is better.
            # Let's simplify the loop to be standard DE.

    # Re-structuring DE loop for clarity and correctness
    pop = []
    for _ in range(pop_size):
        v = np.array([random.uniform(lo, hi) for lo, hi in bounds])
        if random.random() < 0.5:
            v[3*N] = random.uniform(0.8, 1.2)
        pop.append(repair(v))
    scores = [fitness(v) for v in pop]
    
    best_val = -np.inf
    best_v = None
    
    for v, s in zip(pop, scores):
        update_best(v, s)
        
    F = 0.8
    CR = 0.9
    
    while time.time() < deadline - 5:
        for i in range(pop_size):
            if time.time() > deadline - 2:
                break
            
            # Select 3 distinct parents from current population
            candidates = [j for j in range(pop_size) if j != i]
            a, b, c = random.sample(candidates, 3)
            
            # Mutation
            mutant = pop[a] + F * (pop[b] - pop[c])
            
            # Crossover
            trial = np.empty(n_vars)
            j_rand = random.randint(0, n_vars - 1)
            for j in range(n_vars):
                if random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = pop[i][j]
            
            trial = repair(trial)
            trial_score = fitness(trial)
            
            # Selection
            if trial_score <= scores[i]:
                pop[i] = trial
                scores[i] = trial_score
                update_best(trial, trial_score)
            
            # Local refinement: if a solution is valid and good, try small perturbations
            # This helps escape local optima in the penalty landscape
            if scores[i] < 0 and random.random() < 0.1:
                 # Small random walk
                 perturbed = pop[i] + np.random.normal(0, 0.01, n_vars)
                 perturbed = repair(perturbed)
                 p_score = fitness(perturbed)
                 if p_score < scores[i]:
                     pop[i] = perturbed
                     scores[i] = p_score
                     update_best(perturbed, p_score)
                     
    # Final validation and extraction
    if best_v is None:
        # Fallback: Grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        fallback = np.array(cells[:N])
        if _validate(fallback[:, 0], fallback[:, 1], fallback[:, 2], 1.0, 1.0):
            return fallback
        return np.zeros((N, 3))

    x = best_v[:N]
    y = best_v[N:2*N]
    r = best_v[2*N:3*N]
    w = best_v[3*N]
    h = 2.0 - w
    
    # Scale coordinates to fit exactly in w x h? 
    # No, the optimizer found positions in the w x h box.
    # We just need to return (x, y, r).
    
    result = np.column_stack([x, y, r])
    
    # One last check
    if _validate(x, y, r, w, h):
        return result
    
    # If invalid, try to shrink radii slightly to ensure validity
    r_shrunk = r * 0.95
    if _validate(x, y_shrunk, r_shrunk, w, h):
         return np.column_stack([x, y, r_shrunk])
         
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    fallback = np.array(cells[:N])
    if _validate(fallback[:, 0], fallback[:, 1], fallback[:, 2], 1.0, 1.0):
        return fallback
        
    return np.zeros((N, 3))

# EVOLVE-BLOCK-END
