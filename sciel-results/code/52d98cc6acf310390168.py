"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import random
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    # Set random seeds for reproducibility in development, but we want diversity in evolution
    # np.random.seed(42)
    
    def get_penalty_and_sum(x, y, r):
        """
        Calculate the sum of radii and the total penalty for violations.
        Returns (sum_radii, penalty)
        """
        sum_r = np.sum(r)
        penalty = 0.0
        
        # Boundary violations
        # x must be in [r, 1-r]
        for i in range(N):
            if x[i] < r[i]:
                penalty += (r[i] - x[i]) ** 2
            if x[i] > 1.0 - r[i]:
                penalty += (x[i] - (1.0 - r[i])) ** 2
            if y[i] < r[i]:
                penalty += (r[i] - y[i]) ** 2
            if y[i] > 1.0 - r[i]:
                penalty += (y[i] - (1.0 - r[i])) ** 2
        
        # Overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
                min_dist = r[i] + r[j]
                if dist_sq < min_dist ** 2:
                    # Penalty is proportional to the square of the overlap depth
                    overlap = min_dist - np.sqrt(dist_sq)
                    penalty += overlap ** 2
        
        return sum_r, penalty

    def is_feasible(x, y, r, tol=1e-5):
        """Check if the configuration is feasible within tolerance."""
        # Boundaries
        if np.any(x < r - tol) or np.any(x > 1.0 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1.0 - r + tol):
            return False
        
        # Overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - tol) ** 2:
                    return False
        return True

    def repair(x, y, r):
        """
        Heuristically repair an infeasible solution to a feasible one.
        This helps provide good starting points for local optimization.
        """
        # 1. Clamp radii to max possible based on boundaries
        for i in range(N):
            max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            if r[i] > max_r:
                r[i] = max_r
            # Ensure radius is positive
            if r[i] < 0.01:
                r[i] = 0.01
        
        # 2. Resolve overlaps iteratively
        # We do this by pushing circles apart.
        # To make it deterministic and fast, we do a few passes.
        for _ in range(20):
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[i] + r[j]
                    if dist < min_dist and dist > 1e-9:
                        # Push apart
                        overlap = min_dist - dist
                        # Move each circle by half the overlap (or proportional to mass/radius)
                        # Here we just move them apart along the vector
                        push_x = (dx / dist) * (overlap / 2)
                        push_y = (dy / dist) * (overlap / 2)
                        
                        x[i] += push_x
                        y[i] += push_y
                        x[j] -= push_x
                        y[j] -= push_y
                        moved = True
                    elif dist <= 1e-9:
                        # If centers are identical, push randomly
                        x[i] += 0.01
                        x[j] -= 0.01
                        moved = True
            
            # Clamp to boundaries after moving
            for i in range(N):
                x[i] = np.clip(x[i], r[i], 1.0 - r[i])
                y[i] = np.clip(y[i], r[i], 1.0 - r[i])
            
            if not moved:
                break
        
        # Final check: if still overlapping, reduce radii
        # Simple approach: shrink all radii by a factor if still infeasible
        if not is_feasible(x, y, r):
             scale = 0.9
             r = r * scale
             # Re-clamp
             for i in range(N):
                max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                if r[i] > max_r:
                    r[i] = max_r
        
        return x, y, r

    def local_optimize(x, y, r):
        """
        Use scipy to maximize sum of radii starting from a feasible point.
        """
        # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
        vars_init = np.concatenate([x, y, r])
        
        def objective(vars):
            x = vars[0:N]
            y = vars[N:2*N]
            r = vars[2*N:3*N]
            
            # Hard constraints handled by bounds and constraint functions
            # We minimize negative sum of radii
            return -np.sum(r)
        
        # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.01, 0.5)) # r
        
        # Constraints:
        # 1. x >= r
        # 2. x + r <= 1
        # 3. y >= r
        # 4. y + r <= 1
        # 5. (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
        
        constraints = []
        
        # Boundary constraints
        for i in range(N):
            # x_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N + i]})
            # 1 - x_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i] - v[2*N + i]})
            # y_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N + i] - v[2*N + i]})
            # 1 - y_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[N + i] - v[2*N + i]})
            
        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
                def overlap_fun(v, i=i, j=j):
                    xi = v[i]
                    xj = v[j]
                    yi = v[N + i]
                    yj = v[N + j]
                    ri = v[2*N + i]
                    rj = v[2*N + j]
                    return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                constraints.append({'type': 'ineq', 'fun': overlap_fun})
                
        # SLSQP can be slow with many constraints. 
        # Given 26 circles, there are 26*4 + 26*25/2 = 104 + 325 = 429 constraints.
        # This might be too slow for 120s if we run it multiple times.
        # Let's limit the number of local optimization runs.
        
        try:
            # Use SLSQP
            res = minimize(objective, vars_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-6})
            if res.success:
                x_opt = res.x[0:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:3*N]
                # Final feasibility check
                if is_feasible(x_opt, y_opt, r_opt):
                    return x_opt, y_opt, r_opt
        except Exception:
            pass
            
        # If SLSQP fails or is too slow, return the initial repaired solution
        return x, y, r

    # --- Main Evolutionary Process ---
    
    # 1. Initialize Population
    pop_size = 30
    population = []
    
    # Strategy 1: Random uniform
    for _ in range(pop_size // 2):
        x = np.random.rand(N)
        y = np.random.rand(N)
        r = np.random.rand(N) * 0.1 + 0.02 # Small initial radii
        x, y, r = repair(x, y, r)
        population.append((x, y, r))
        
    # Strategy 2: Grid-based start
    side = int(np.ceil(np.sqrt(N)))
    for _ in range(pop_size // 2):
        x = []
        y = []
        r = []
        for i in range(side):
            for j in range(side):
                if len(x) < N:
                    # Jitter the grid points
                    cx = (i + 0.5) / side + np.random.uniform(-0.05, 0.05)
                    cy = (j + 0.5) / side + np.random.uniform(-0.05, 0.05)
                    cx = np.clip(cx, 0, 1)
                    cy = np.clip(cy, 0, 1)
                    # Initial radius based on grid spacing
                    max_r = min(cx, 1-cx, cy, 1-cy, 0.5/side)
                    r_val = max_r * np.random.uniform(0.8, 1.0)
                    x.append(cx)
                    y.append(cy)
                    r.append(r_val)
        x = np.array(x)
        y = np.array(y)
        r = np.array(r)
        x, y, r = repair(x, y, r)
        population.append((x, y, r))
        
    # Evaluate initial population
    scores = []
    for x, y, r in population:
        sum_r, pen = get_penalty_and_sum(x, y, r)
        # Fitness: High sum, low penalty
        # If penalty is 0, fitness is sum_r
        # If penalty > 0, fitness is sum_r - 10 * penalty
        fitness = sum_r - 10.0 * pen
        scores.append(fitness)
        
    best_x, best_y, best_r = None, None, None
    best_score = -np.inf
    
    # Simple Genetic Algorithm Loop
    generations = 20
    for gen in range(generations):
        # Select best so far
        best_idx = np.argmax(scores)
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_x, best_y, best_r = population[best_idx]
            
        # Elitism: Keep top 2
        sorted_indices = np.argsort(scores)[::-1]
        top_indices = sorted_indices[:2]
        
        # Create new population
        new_population = []
        new_scores = []
        
        # Add elites
        for idx in top_indices:
            new_population.append(population[idx])
            new_scores.append(scores[idx])
            
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection: Roulette wheel or Tournament
            # Simple: Pick two random parents
            p1_idx = np.random.randint(0, pop_size)
            p2_idx = np.random.randint(0, pop_size)
            
            x1, y1, r1 = population[p1_idx]
            x2, y2, r2 = population[p2_idx]
            
            # Crossover: Uniform
            mask = np.random.rand(N) > 0.5
            x_child = np.where(mask, x1, x2)
            y_child = np.where(mask, y1, y2)
            r_child = np.where(mask, r1, r2)
            
            # Mutation
            if np.random.rand() < 0.8: # Mutation rate
                # Gaussian noise
                x_child += np.random.normal(0, 0.05, N)
                y_child += np.random.normal(0, 0.05, N)
                r_child += np.random.normal(0, 0.02, N)
                
                # Clip values
                x_child = np.clip(x_child, 0, 1)
                y_child = np.clip(y_child, 0, 1)
                r_child = np.clip(r_child, 0.01, 0.5)
            
            # Repair
            x_child, y_child, r_child = repair(x_child, y_child, r_child)
            
            new_population.append((x_child, y_child, r_child))
            
            # Evaluate
            sum_r, pen = get_penalty_and_sum(x_child, y_child, r_child)
            fitness = sum_r - 10.0 * pen
            new_scores.append(fitness)
            
        population = new_population
        scores = new_scores

    # 2. Local Optimization on Top Candidates
    # Get top 3 unique candidates from the final population
    final_sorted_indices = np.argsort(scores)[::-1]
    top_candidates = []
    for idx in final_sorted_indices:
        # Check uniqueness roughly
        unique = True
        for tx, ty, tr in top_candidates:
            if np.allclose(population[idx][0], tx, atol=0.1) and \
               np.allclose(population[idx][1], ty, atol=0.1) and \
               np.allclose(population[idx][2], tr, atol=0.1):
                unique = False
                break
        if unique:
            top_candidates.append(population[idx])
        if len(top_candidates) >= 3:
            break
            
    best_final_score = -np.inf
    best_final_sol = None
    
    # Run local optimization on top candidates
    # Note: SLSQP might be slow. We time-box it.
    start_time = time.time()
    time_limit = 100 # Leave 20s for overhead
    
    for x, y, r in top_candidates:
        if time.time() - start_time > time_limit:
            break
            
        # Ensure it's a good start
        if not is_feasible(x, y, r):
             x, y, r = repair(x, y, r)
             
        x_opt, y_opt, r_opt = local_optimize(x, y, r)
        
        # Evaluate result
        sum_r, pen = get_penalty_and_sum(x_opt, y_opt, r_opt)
        if pen < 1e-5: # Feasible
            if sum_r > best_final_score:
                best_final_score = sum_r
                best_final_sol = (x_opt, y_opt, r_opt)
                
    # Fallback: If local opt failed or wasn't run, use best GA solution
    if best_final_sol is None:
        # Find best feasible in GA population
        for i in range(len(population)):
            x, y, r = population[i]
            if is_feasible(x, y, r):
                sum_r, _ = get_penalty_and_sum(x, y, r)
                if sum_r > best_final_score:
                    best_final_score = sum_r
                    best_final_sol = (x, y, r)
                    
    if best_final_sol is None:
        # Last resort: Grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_final_sol = (np.array([c[0] for c in cells[:N]]), np.array([c[1] for c in cells[:N]]), np.array([c[2] for c in cells[:N]]))
        
    x_res, y_res, r_res = best_final_sol
    
    # Format output
    result = np.zeros((N, 3))
    result[:, 0] = x_res
    result[:, 1] = y_res
    result[:, 2] = r_res
    
    return result
# EVOLVE-BLOCK-END
