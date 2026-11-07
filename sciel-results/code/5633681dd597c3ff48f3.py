"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

def _compute_score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _generate_all_edges(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def construct():
    n = 14
    m = 35
    all_edges = _generate_all_edges(n)
    
    # Initialize with a heuristic: small values
    # Using a set of small integers is a good start for minimizing sums/products
    initial_values = list(range(1, n + 1))
    
    # Randomize edges slightly to avoid local minima
    initial_edges = random.sample(all_edges, m)
    
    best_values = initial_values
    best_edges = initial_edges
    best_score = _compute_score(best_values, best_edges)
    
    # Local Search
    T = 5.0
    cooling = 0.9995
    min_T = 0.01
    current_values = best_values
    current_edges = best_edges
    current_score = best_score
    
    # Precompute edge set for fast lookup
    current_edge_set = set(tuple(e) for e in current_edges)
    
    for _ in range(20000): # Fixed iterations for determinism and speed control
        if T < min_T:
            break
            
        # Choose move
        if random.random() < 0.4:
            # Value Perturbation
            idx = random.randint(0, n - 1)
            # Change to a random small value
            new_val = random.randint(1, 30)
            # Ensure distinctness
            while new_val in current_values and new_val != current_values[idx]:
                new_val = random.randint(1, 30)
            if new_val == current_values[idx]:
                continue
            new_values = current_values.copy()
            new_values[idx] = new_val
            new_edges = current_edges
        else:
            # Edge Swap
            # Pick a random edge to remove
            rem_idx = random.randint(0, m - 1)
            rem_edge = current_edges[rem_idx]
            
            # Pick a random edge to add (not currently in set)
            # To avoid O(N) check every time, we can sample from all_edges
            # and check if it's in current_edge_set
            new_edge = None
            for _ in range(10): # Try a few times
                cand = random.choice(all_edges)
                if cand not in current_edge_set:
                    new_edge = cand
                    break
            if new_edge is None:
                # Fallback: pick any non-edge
                non_edges = [e for e in all_edges if e not in current_edge_set]
                if not non_edges:
                    continue
                new_edge = random.choice(non_edges)
            
            new_edges = current_edges.copy()
            new_edges[rem_idx] = new_edge
            new_values = current_values
            
        new_score = _compute_score(new_values, new_edges)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_values = new_values
            current_edges = new_edges
            current_score = new_score
            current_edge_set = set(tuple(e) for e in current_edges)
            
            if current_score < best_score:
                best_score = current_score
                best_values = current_values
                best_edges = current_edges
                
        T *= cooling
        
    # Final cleanup to ensure types
    return {'values': list(best_values), 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
