"""K4-free graphs without independent 6-sets (Ramsey number R(4,6))."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 145.0  # seconds; evaluator kills at 150s, keep margin

def _build_masks(n, adj):
    """Build bitmask representation from adjacency matrix."""
    masks = [0] * n
    for u in range(n):
        m = 0
        for v in range(n):
            if adj[u, v]:
                m |= (1 << v)
        masks[u] = m
    return masks

def _has_k4(n, masks):
    """Check if graph contains a K4 using bitmask operations."""
    for u in range(n):
        for v in range(u + 1, n):
            if not (masks[u] & (1 << v)):
                continue
            common = masks[u] & masks[v]
            common &= ~((1 << (v + 1)) - 1)
            w_mask = common
            while w_mask:
                lsb_w = w_mask & -w_mask
                wi = lsb_w.bit_length() - 1
                w_mask ^= lsb_w
                x_mask = common & ~((1 << (wi + 1)) - 1)
                while x_mask:
                    lsb_x = x_mask & -x_mask
                    xi = lsb_x.bit_length() - 1
                    x_mask ^= lsb_x
                    if masks[wi] & (1 << xi):
                        return True
    return False

def _has_ind6(n, masks):
    """Check if graph has an independent 6-set (clique of size 6 in complement)."""
    full_mask = (1 << n) - 1
    comp_masks = [0] * n
    for u in range(n):
        comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
    
    for u in range(n):
        for v in range(u + 1, n):
            if not (comp_masks[u] & (1 << v)):
                continue
            common = comp_masks[u] & comp_masks[v]
            common &= ~((1 << (v + 1)) - 1)
            w_mask = common
            while w_mask:
                lsb_w = w_mask & -w_mask
                wi = lsb_w.bit_length() - 1
                w_mask ^= lsb_w
                x_mask = common & ~((1 << (wi + 1)) - 1)
                while x_mask:
                    lsb_x = x_mask & -x_mask
                    xi = lsb_x.bit_length() - 1
                    x_mask ^= lsb_x
                    if comp_masks[wi] & (1 << xi):
                        common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                        common2 &= ~((1 << (xi + 1)) - 1)
                        y_mask = common2
                        while y_mask:
                            lsb_y = y_mask & -y_mask
                            yi = lsb_y.bit_length() - 1
                            y_mask ^= lsb_y
                            z_mask = common2 & ~((1 << (yi + 1)) - 1)
                            while z_mask:
                                lsb_z = z_mask & -z_mask
                                zi = lsb_z.bit_length() - 1
                                z_mask ^= lsb_z
                                if comp_masks[yi] & (1 << zi):
                                    return True
    return False

def _count_k4(n, masks):
    """Count K4 subgraphs using bitmask operations."""
    k4_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if not (masks[u] & (1 << v)):
                continue
            common = masks[u] & masks[v]
            common &= ~((1 << (v + 1)) - 1)
            w_mask = common
            while w_mask:
                lsb_w = w_mask & -w_mask
                wi = lsb_w.bit_length() - 1
                w_mask ^= lsb_w
                x_mask = common & ~((1 << (wi + 1)) - 1)
                while x_mask:
                    lsb_x = x_mask & -x_mask
                    xi = lsb_x.bit_length() - 1
                    x_mask ^= lsb_x
                    if masks[wi] & (1 << xi):
                        k4_count += 1
    return k4_count

def _count_ind6(n, masks):
    """Count independent 6-sets using bitmask operations."""
    full_mask = (1 << n) - 1
    comp_masks = [0] * n
    for u in range(n):
        comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
    
    ind6_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if not (comp_masks[u] & (1 << v)):
                continue
            common = comp_masks[u] & comp_masks[v]
            common &= ~((1 << (v + 1)) - 1)
            w_mask = common
            while w_mask:
                lsb_w = w_mask & -w_mask
                wi = lsb_w.bit_length() - 1
                w_mask ^= lsb_w
                x_mask = common & ~((1 << (wi + 1)) - 1)
                while x_mask:
                    lsb_x = x_mask & -x_mask
                    xi = lsb_x.bit_length() - 1
                    x_mask ^= lsb_x
                    if comp_masks[wi] & (1 << xi):
                        common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                        common2 &= ~((1 << (xi + 1)) - 1)
                        y_mask = common2
                        while y_mask:
                            lsb_y = y_mask & -y_mask
                            yi = lsb_y.bit_length() - 1
                            y_mask ^= lsb_y
                            z_mask = common2 & ~((1 << (yi + 1)) - 1)
                            while z_mask:
                                lsb_z = z_mask & -z_mask
                                zi = lsb_z.bit_length() - 1
                                z_mask ^= lsb_z
                                if comp_masks[yi] & (1 << zi):
                                    ind6_count += 1
    return ind6_count

def _total_violations(n, adj):
    """Total number of K4s + independent 6-sets."""
    masks = _build_masks(n, adj)
    return _count_k4(n, masks) + _count_ind6(n, masks)

def _is_valid(n, adj):
    """Check if graph is valid: no K4 and no independent 6-set."""
    masks = _build_masks(n, adj)
    if _has_k4(n, masks):
        return False
    if _has_ind6(n, masks):
        return False
    return True

def _edges_from_adj(n, adj):
    """Extract edge list from adjacency matrix."""
    return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]

def _clebsch_16():
    """Clebsch graph on 16 vertices (K4-free, alpha=5)."""
    clebsch_edges = [
        (0, 6), (0, 7), (0, 10), (0, 11),
        (1, 5), (1, 8), (1, 10), (1, 13),
        (2, 4), (2, 9), (2, 11), (2, 13),
        (3, 4), (3, 7), (3, 9), (3, 12),
        (5, 6), (5, 8), (5, 12), (5, 15),
        (6, 7), (6, 9), (6, 14), (6, 15),
        (7, 8), (7, 11), (7, 14), (7, 15),
        (8, 9), (8, 12), (8, 15),
        (9, 10), (9, 14),
        (10, 11), (10, 13), (10, 14),
        (11, 12), (11, 14), (11, 15),
        (12, 13), (12, 15),
        (13, 14), (13, 15)
    ]
    n = 16
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in clebsch_edges:
        adj[u, v] = 1
        adj[v, u] = 1
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
    scores = [_total_violations(n, adj) for adj in population]
    
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
        scores = [_total_violations(n, adj) for adj in population]
    
    # Return best found
    if best_adj is not None:
        return best_adj
    return population[0]

def _targeted_repair(adj, n, rng, max_iter=500):
    """Try to repair violations by targeting specific problematic subgraphs."""
    best_adj = adj.copy()
    best_score = _total_violations(n, best_adj)
    
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
            
            new_score = _total_violations(n, trial)
            if new_score < best_score:
                best_adj = trial
                best_score = new_score
                improved = True
                if best_score == 0:
                    return best_adj
        
        if not improved:
            # Try more aggressive mutation
            trial = _mutate_adj(best_adj, n, rng, rng.randint(2, 5))
            new_score = _total_violations(n, trial)
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
    
    best_n = 16
    best_adj = _clebsch_16()
    
    # Verify baseline
    if not _is_valid(best_n, best_adj):
        # Fallback: empty graph on 5 vertices (valid: no K4, no ind 6)
        best_n = 5
        best_adj = np.zeros((5, 5), dtype=np.int8)
    
    # Try to find valid graphs for increasing n
    for target_n in [17, 18, 19, 20, 22, 24, 26, 28, 30, 32, 34, 35, 36]:
        if time.time() > _DEADLINE - 15:
            break
        
        remaining_time = _DEADLINE - 15 - time.time()
        if remaining_time < 2:
            break
        
        # Try genetic search with time budget
        ga_time = min(remaining_time * 0.5, 25.0)
        
        adj_found = _genetic_search(target_n, rng, ga_time)
        
        if adj_found is not None and _is_valid(target_n, adj_found):
            best_n = target_n
            best_adj = adj_found
            continue
        
        # Try targeted repair on the best found
        if adj_found is not None:
            repaired = _targeted_repair(adj_found, target_n, rng, max_iter=200)
            if _is_valid(target_n, repaired):
                best_n = target_n
                best_adj = repaired
                continue
        
        # Try a few more random restarts
        for attempt in range(3):
            if time.time() > _DEADLINE - 10:
                break
            
            p = 0.2 + rng.random() * 0.3
            adj = _random_graph(target_n, p, rng)
            score = _total_violations(target_n, adj)
            
            if score == 0:
                best_n = target_n
                best_adj = adj
                break
            
            # Try to repair
            repaired = _targeted_repair(adj, target_n, rng, max_iter=100)
            if _is_valid(target_n, repaired):
                best_n = target_n
                best_adj = repaired
                break
    
    # Convert to edge list
    edges = _edges_from_adj(best_n, best_adj)
    
    return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-END
