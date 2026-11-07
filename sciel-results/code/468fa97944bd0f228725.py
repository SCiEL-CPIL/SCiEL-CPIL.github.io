import numpy as np
import time

def _greedy_tour(prizes, distance_matrix, max_len, heuristic):
    """Deterministic greedy tour: at each step pick the node maximizing heuristic[i,j] * prize[j]
    among feasible unvisited nodes. Returns total prize collected."""
    n = len(prizes)
    d = distance_matrix
    d_j0 = d[0, :]
    cost = d + d_j0[None, :]  # cost of going i->j and then j->0
    
    eta = heuristic.copy()
    eta[eta <= 0] = 1e-9
    
    cur = 0
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    length = 0.0
    prize = 0.0
    
    while True:
        remaining = max_len - length
        valid_mask = (~visited) & (cost[cur, :] <= remaining)
        valid_nodes = np.where(valid_mask)[0]
        
        if len(valid_nodes) == 0:
            break
        
        # Greedy: pick node with highest heuristic[i,j] * prize[j]
        scores = eta[cur, valid_nodes] * prizes[valid_nodes]
        best_idx = np.argmax(scores)
        next_node = valid_nodes[best_idx]
        
        visited[next_node] = True
        length += d[cur, next_node]
        prize += prizes[next_node]
        cur = next_node
    
    return prize

def _run_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=0):
    """Standard ACO run as specified."""
    n = len(prizes)
    rng = np.random.default_rng(seed)
    
    eta = heuristic.copy()
    eta[eta <= 0] = 1e-9
    
    tau = np.ones((n, n))
    
    d = distance_matrix
    d_j0 = d[0, :]
    cost = d + d_j0[None, :]
    
    best_global = 0.0
    
    for _ in range(num_iters):
        best_iter = 0.0
        
        for _ in range(num_ants):
            cur = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            length = 0.0
            prize = 0.0
            
            while True:
                remaining = max_len - length
                valid_mask = (~visited) & (cost[cur, :] <= remaining)
                valid_nodes = np.where(valid_mask)[0]
                
                if len(valid_nodes) == 0:
                    break
                
                w = tau[cur, valid_nodes] * eta[cur, valid_nodes]
                sum_w = w.sum()
                if sum_w < 1e-10:
                    break
                p = w / sum_w
                
                next_node = rng.choice(valid_nodes, p=p)
                
                visited[next_node] = True
                length += d[cur, next_node]
                prize += prizes[next_node]
                cur = next_node
            
            if prize > best_iter:
                best_iter = prize
        
        if best_iter > best_global:
            best_global = best_iter
        
        tau = 0.9 * tau + (best_iter / 10.0)
    
    return best_global

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill
    
    n = len(prizes)
    d = distance_matrix
    p = prizes
    eps = 1e-9
    
    d_j0 = d[0, :]
    
    # Initialize with a reasonable base heuristic: prize / (distance + return_distance)
    base_heuristic = p[None, :] / (d + d_j0[None, :] + eps)
    base_heuristic[base_heuristic <= 0] = 1e-9
    
    # Work in log-space for the heuristic to ensure positivity
    # We'll evolve a log-heuristic matrix, but only for off-diagonal, non-zero entries
    # Actually, let's evolve a perturbation in log-space around the base
    
    # Strategy: CMA-ES-like search over a parameterization of the heuristic
    # Parameterize: log(heuristic[i,j]) = log(base[i,j]) + x[i,j]
    # where x is what we optimize. We'll use a simple evolutionary strategy.
    
    # For efficiency, we'll optimize a low-dimensional parameterization:
    # heuristic[i,j] = p[j]^a / (d[i,j] + d[j,0])^b * exp(c * p[j] * d[i,j])
    # This gives us 3 parameters to optimize: a, b, c
    # Plus we can add a per-node bias term
    
    # Actually, let's do a more direct approach: evolve the full log-heuristic matrix
    # but use a fast greedy fitness function
    
    # Population-based search over log-heuristic matrices
    # Dimension: n*n = 2500, which is high. Let's use a lower-dim parameterization.
    
    # Better approach: evolve a small set of parameters that define the heuristic formula
    # h[i,j] = (p[j] / (d[i,j] + d[j,0] + eps))^alpha * exp(beta * p[j]) * (1 + gamma * d[i,j])
    # 3 parameters: alpha, beta, gamma
    
    # Even better: let's do a two-stage approach
    # Stage 1: Evolve 3-5 parameters of a parametric heuristic family using greedy fitness
    # Stage 2: Fine-tune with small perturbations of the resulting matrix using ACO verification
    
    def parametric_heuristic(params):
        """Generate heuristic from parameters [alpha, beta, gamma, delta]"""
        alpha, beta, gamma, delta = params
        # h[i,j] = (p[j] / (d[i,j] + d[j,0] + eps))^alpha * exp(beta * p[j]) * (1 + gamma * d[i,j])^delta
        num = np.maximum(p, eps) ** alpha
        den = (d + d_j0[None, :] + eps) ** alpha
        base = num[None, :] / den
        bias = np.exp(beta * p[None, :])
        dist_factor = (1.0 + gamma * d) ** delta
        h = base * bias * dist_factor
        h[h <= 0] = 1e-9
        return h
    
    # Evaluate a candidate heuristic using greedy tour (fast)
    def evaluate_heuristic(h):
        return _greedy_tour(p, d, max_len, h)
    
    # Initial population of parameter vectors
    # Search space: alpha in [0.2, 3.0], beta in [-2.0, 2.0], gamma in [-0.5, 0.5], delta in [0.5, 2.0]
    n_params = 4
    pop_size = 30
    
    # Initial population: centered around known good values
    # Good baseline: alpha=1, beta=0, gamma=0, delta=1
    center = np.array([1.0, 0.0, 0.0, 1.0])
    sigma = np.array([0.5, 0.5, 0.3, 0.3])
    
    rng = np.random.default_rng(123)
    
    # Generate initial population
    population = center[None, :] + rng.normal(0, 1, size=(pop_size, n_params)) * sigma[None, :]
    
    # Evaluate initial population
    fitnesses = np.zeros(pop_size)
    for i in range(pop_size):
        h = parametric_heuristic(population[i])
        fitnesses[i] = evaluate_heuristic(h)
    
    best_idx = np.argmax(fitnesses)
    best_params = population[best_idx].copy()
    best_fitness = fitnesses[best_idx]
    
    # Evolutionary loop
    max_generations = 50
    for gen in range(max_generations):
        if time.time() > deadline - 60.0:
            break
        
        # Sort by fitness
        sorted_idx = np.argsort(fitnesses)[::-1]
        population = population[sorted_idx]
        fitnesses = fitnesses[sorted_idx]
        
        # Keep best
        best_params = population[0].copy()
        best_fitness = fitnesses[0]
        
        # Generate offspring
        new_population = []
        new_fitnesses = []
        
        # Elitism: keep top 5
        elite_count = 5
        for i in range(min(elite_count, pop_size)):
            new_population.append(population[i].copy())
            new_fitnesses.append(fitnesses[i])
        
        # Generate remaining via mutation from top parents
        for i in range(elite_count, pop_size):
            # Pick a parent from top half
            parent_idx = rng.integers(0, max(1, pop_size // 2))
            # CMA-ES-like step size (simplified)
            step = sigma * (0.5 + 0.5 * rng.random())
            child = population[parent_idx] + rng.normal(0, 1, size=n_params) * step
            new_population.append(child)
            
            if time.time() > deadline - 60.0:
                break
        
        population = np.array(new_population[:pop_size])
        # Re-evaluate non-elite members
        for i in range(elite_count, len(population)):
            h = parametric_heuristic(population[i])
            fitnesses[i] = evaluate_heuristic(h)
        
        # Update sigma based on improvement (simple adaptation)
        if gen > 0 and best_fitness > fitnesses[0] + 1e-6:
            sigma *= 0.95  # Converging
        elif gen > 0 and best_fitness < fitnesses[0] - 1e-6:
            sigma *= 1.05  # Diverging
    
    # Get best parametric heuristic
    best_h = parametric_heuristic(best_params)
    best_h[best_h <= 0] = 1e-9
    
    # Stage 2: Fine-tune the matrix directly using small perturbations
    # Evaluate top candidates with actual ACO to verify
    candidates = [best_h]
    
    # Add a few variations
    for scale in [0.7, 0.85, 1.0, 1.15, 1.3]:
        candidates.append(best_h * scale)
    
    # Also try the original base heuristic
    candidates.append(base_heuristic.copy())
    
    # Evaluate with ACO
    best_aco_score = -1
    best_final_h = best_h
    
    for cand in candidates:
        if time.time() > deadline - 30.0:
            break
        # Quick ACO check
        score = _run_aco(p, d, max_len, cand, num_ants=10, num_iters=15, seed=0)
        if score > best_aco_score:
            best_aco_score = score
            best_final_h = cand
    
    # Full ACO on best
    if time.time() < deadline - 15.0:
        score = _run_aco(p, d, max_len, best_final_h, num_ants=20, num_iters=30, seed=0)
        best_aco_score = max(best_aco_score, score)
    
    # Final validation
    best_final_h = best_final_h.copy()
    best_final_h[best_final_h <= 0] = 1e-9
    if best_final_h.shape != (n, n) or not np.all(np.isfinite(best_final_h)):
        return np.ones((n, n))
    
    return best_final_h
