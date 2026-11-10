import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin
    
    n = 14
    m = 35
    maximum = 10000
    
    # Use the most compact set of values to maximize collisions
    values = list(range(1, n + 1))
    
    # Precompute all possible edges and their sums/products
    all_edges = list(combinations(range(n), 2))
    E = len(all_edges)
    
    edge_sums = [0] * E
    edge_prods = [0] * E
    for idx, (u, v) in enumerate(all_edges):
        edge_sums[idx] = values[u] + values[v]
        edge_prods[idx] = values[u] * values[v]
    
    def get_score(edge_indices):
        """Calculate the raw value for a set of edge indices."""
        sums = set()
        prods = set()
        for idx in edge_indices:
            sums.add(edge_sums[idx])
            prods.add(edge_prods[idx])
        return max(len(sums), len(prods))
    
    best_score = float('inf')
    best_edges = None
    
    # Local search with random restarts
    # We will run multiple restarts, each doing a number of swaps
    
    num_restarts = 1000
    swaps_per_restart = 1000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with a random set of edges
        current_edges = set(random.sample(range(E), m))
        current_score = get_score(current_edges)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = list(current_edges)
            
        # Simulated Annealing / Hill Climbing
        # Temperature decays
        T = 5.0
        for s in range(swaps_per_restart):
            if time.time() > deadline:
                break
                
            # Pick a random edge to remove
            e_out = random.choice(list(current_edges))
            
            # Pick a random edge to add (not in current set)
            # To make this faster, we can just pick a random index and check
            # Or maintain a list of available edges. With E=91, it's small.
            e_in = random.randint(0, E - 1)
            
            if e_in in current_edges:
                continue
                
            # Calculate new score
            new_edges = (current_edges - {e_out}) | {e_in}
            new_score = get_score(new_edges)
            
            # Acceptance criteria
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(current_edges)
            else:
                # Accept worse moves with probability exp(-(new_score - current_score)/T)
                # Since scores are integers, difference is at least 1
                # T should be small enough to prefer improvements but allow exploration
                # Let's use a simple probability that decreases with s
                p = 0.1 if s < swaps_per_restart // 2 else 0.01
                if random.random() < p:
                    current_edges = new_edges
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = list(current_edges)
        
        # Decay temperature for next restart? Or just keep it simple.
        # Let's try a more aggressive local search: 
        # Try all possible swaps from the current best found so far?
        # No, that's too slow if we do it from scratch each time.
        
        # Let's add a second phase: refine the best_edges found so far
        if best_edges is not None:
            current_edges = set(best_edges)
            current_score = get_score(current_edges)
            
            # Try all single swaps
            improved = True
            while improved:
                improved = False
                for e_out in list(current_edges):
                    if time.time() > deadline:
                        break
                    for e_in in range(E):
                        if e_in in current_edges:
                            continue
                        new_edges = (current_edges - {e_out}) | {e_in}
                        new_score = get_score(new_edges)
                        if new_score < current_score:
                            current_edges = new_edges
                            current_score = new_score
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_edges = list(current_edges)
                            break # restart inner loop with new current_edges
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break

    # Fallback if something went wrong (should not happen)
    if best_edges is None:
        best_edges = list(range(m))
    
    # Format output
    edges_list = [[all_edges[idx][0], all_edges[idx][1]] for idx in best_edges]
    
    return {
        'values': values,
        'edges': edges_list
    }
