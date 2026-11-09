"""K4-free graphs without independent 6-sets (Ramsey number R(4,6))."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 145.0  # seconds; evaluator kills at 150s, keep margin

def _count_k4(adj_np):
    """Count K4 subgraphs in adjacency matrix (numpy int array, 0/1, symmetric, 0 diag)."""
    n = adj_np.shape[0]
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj_np[i,j] == 0:
                continue
            common = np.bitwise_and(adj_np[i,:], adj_np[j,:])
            common[i] = 0
            common[j] = 0
            # Count edges among common neighbors
            for a in range(n):
                if common[a] == 0:
                    continue
                for b in range(a+1, n):
                    if common[b] != 0 and adj_np[a,b] != 0:
                        count += 1
    return count

def _count_indep6(adj_np):
    """Count independent 6-sets in graph (cliques of size 6 in complement)."""
    n = adj_np.shape[0]
    comp = 1 - adj_np
    np.fill_diagonal(comp, 0)
    count = 0
    # Use itertools to enumerate 6-subsets
    from itertools import combinations
    for subset in combinations(range(n), 6):
        ok = True
        for idx in range(6):
            for jdx in range(idx+1, 6):
                if comp[subset[idx], subset[jdx]] != 0:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            count += 1
    return count

def _total_violations(adj_np):
    """Total number of K4s + independent 6-sets."""
    return _count_k4(adj_np) + _count_indep6(adj_np)

def _is_valid(adj_np):
    """Check if graph is valid: no K4 and no independent 6-set."""
    return _total_violations(adj_np) == 0

def _edges_from_adj(adj_np):
    """Extract edge list from numpy adjacency matrix."""
    n = adj_np.shape[0]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj_np[i,j] != 0:
                edges.append([i, j])
    return edges

def _paley_17():
    """Paley graph on 17 vertices (known valid)."""
    squares = set((x*x) % 17 for x in range(1, 17))
    n = 17
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i+1, n):
            if (j - i) % 17 in squares:
                adj[i,j] = 1
                adj[j,i] = 1
    return adj

def _random_graph(n, p, rng):
    """Generate a random graph with edge probability p."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < p:
                adj[i,j] = 1
                adj[j,i] = 1
    return adj

def _mutate_adj(adj, n, rng, num_flips):
    """Flip a random number of edges."""
    new_adj = adj.copy()
    for _ in range(num_flips):
        i = rng.randint(0, n-1)
        j = rng.randint(0, n-1)
        if i != j:
            new_adj[i,j] ^= 1
            new_adj[j,i] = new_adj[i,j]
    return new_adj

def _crossover(a, b, rng):
    """Uniform crossover of two adjacency matrices."""
    n = a.shape[0]
    child = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < 0.5:
                child[i,j] = a[i,j]
                child[j,i] = a[i,j]
            else:
                child[i,j] = b[i,j]
                child[j,i] = b[i,j]
    return child

def _genetic_search(n, rng, time_budget):
    """Genetic algorithm to find valid graph on n vertices."""
    pop_size = 50
    elite_size = 5
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        p = 0.2 + rng.random() * 0.3
        adj = _random_graph(n, p, rng)
        population.append(adj)
    
    # Evaluate initial population
    scores = [_total_violations(adj) for adj in population]
    
    best_adj = None
    best_score = min(scores)
    if best_score == 0:
        return population[scores.index(0)]
    
    start_time = time.time()
    generation = 0
    
    while time.time() - start_time < time_budget:
        generation += 1
        
        # Sort by fitness (lower is better)
        indices = sorted(range(pop_size), key=lambda i: scores[i])
        
        # Update best
        if scores[indices[0]] < best_score:
            best_score = scores[indices[0]]
            best_adj = population[indices[0]]
            if best_score == 0:
                return best_adj
        
        # Elitism: keep top individuals
        new_population = [population[indices[i]].copy() for i in range(elite_size)]
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def select():
                candidates = rng.sample(range(pop_size), 3)
                best = min(candidates, key=lambda i: scores[i])
                return population[best]
            
            parent1 = select()
            parent2 = select()
            
            # Crossover
            if rng.random() < 0.7:
                child = _crossover(parent1, parent2, rng)
            else:
                child = parent1.copy()
            
            # Mutation
            num_flips = rng.randint(1, max(1, n // 5))
            child = _mutate_adj(child, n, rng, num_flips)
            
            new_population.append(child)
        
        population = new_population
        scores = [_total_violations(adj) for adj in population]
    
    # Return best found
    if best_adj is not None:
        return best_adj
    return population[0]

def _targeted_repair(adj, n, rng, max_iter=500):
    """Try to repair violations by targeting specific problematic subgraphs."""
    best_adj = adj.copy()
    best_score = _total_violations(best_adj)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        if best_score == 0:
            return best_adj
        
        # Try flipping random edges
        improved = False
        for _ in range(20):
            i = rng.randint(0, n-1)
            j = rng.randint(0, n-1)
            if i == j:
                continue
            
            trial = best_adj.copy()
            trial[i,j] ^= 1
            trial[j,i] = trial[i,j]
            
            new_score = _total_violations(trial)
            if new_score < best_score:
                best_adj = trial
                best_score = new_score
                improved = True
                if best_score == 0:
                    return best_adj
        
        if not improved:
            # Try more aggressive mutation
            trial = _mutate_adj(best_adj, n, rng, rng.randint(2, 5))
            new_score = _total_violations(trial)
            if new_score < best_score:
                best_adj = trial
                best_score = new_score
            elif rng.random() < 0.1:
                # Accept worse solution with small probability (simulated annealing-like)
                best_adj = trial
                best_score = new_score
    
    return best_adj

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_n = 17
    best_adj = _paley_17()
    
    # Try to find valid graphs for increasing n
    for target_n in [20, 22, 24, 26, 28, 30, 32, 34, 35, 36]:
        if time.time() > _DEADLINE - 15:
            break
        
        remaining_time = _DEADLINE - 15 - time.time()
        if remaining_time < 2:
            break
        
        # Try genetic search with time budget
        ga_time = min(remaining_time * 0.6, 30.0)
        
        adj_found = _genetic_search(target_n, rng, ga_time)
        
        if adj_found is not None and _is_valid(adj_found):
            best_n = target_n
            best_adj = adj_found
            continue
        
        # Try targeted repair on the best found
        if adj_found is not None:
            repaired = _targeted_repair(adj_found, target_n, rng, max_iter=200)
            if _is_valid(repaired):
                best_n = target_n
                best_adj = repaired
                continue
        
        # Try a few more random restarts
        for attempt in range(3):
            if time.time() > _DEADLINE - 10:
                break
            
            p = 0.2 + rng.random() * 0.3
            adj = _random_graph(target_n, p, rng)
            score = _total_violations(adj)
            
            if score == 0:
                best_n = target_n
                best_adj = adj
                break
            
            # Try to repair
            repaired = _targeted_repair(adj, target_n, rng, max_iter=100)
            if _is_valid(repaired):
                best_n = target_n
                best_adj = repaired
                break
    
    # Convert to edge list
    edges = _edges_from_adj(best_adj)
    
    return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-END
