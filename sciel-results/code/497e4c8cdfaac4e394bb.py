"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import Counter

P = {'n': 14, 'm': 35, 'maximum': 10000}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit, evaluator kills at 90s
    
    n = P['n']
    m = P['m']
    max_val = P['maximum']
    
    # Pre-compute all pairs of indices
    all_pairs = list(itertools.combinations(range(n), 2))
    
    def get_score(vals, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        return max(len(sums), len(prods))

    def find_best_edges(vals):
        # Precompute sums and products for all pairs
        sums = [vals[i] + vals[j] for i, j in all_pairs]
        prods = [vals[i] * vals[j] for i, j in all_pairs]
        
        sum_counts = Counter(sums)
        prod_counts = Counter(prods)
        
        # Score each pair
        scores = []
        for k, (i, j) in enumerate(all_pairs):
            s_score = sum_counts[sums[k]]
            p_score = prod_counts[prods[k]]
            # Higher score means more collisions in the complete graph
            scores.append((s_score + p_score, i, j))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Select top m
        best_edges = [(i, j) for _, i, j in scores[:m]]
        return best_edges

    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Candidate pool: Small numbers are crucial for product collisions
    # We focus on 1..100 to allow for diverse small factor sets
    
    num_restarts = 30
    max_val_iters = 100
    max_edge_iters = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize values
        # Strategy: Mix of very small numbers (1-10) and some slightly larger (11-50)
        # Small numbers have more divisors, leading to product collisions
        pool = list(range(1, 51))
        random.shuffle(pool)
        current_values = pool[:n]
        current_values.sort()
        
        current_edges = find_best_edges(current_values)
        current_score = get_score(current_values, current_edges)
        
        if current_score < best_score:
            best_score = current_score
            best_values = current_values[:]
            best_edges = [e[:] for e in current_edges]
            
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Value Swap
            # Try to swap a random index with a random candidate
            for _ in range(10):
                idx = random.randint(0, n - 1)
                old_val = current_values[idx]
                
                # Pick a new value from 1..100
                new_val = random.randint(1, 100)
                if new_val == old_val or new_val in current_values:
                    continue
                    
                temp_vals = current_values[:]
                temp_vals[idx] = new_val
                temp_vals.sort()
                
                temp_edges = find_best_edges(temp_vals)
                temp_score = get_score(temp_vals, temp_edges)
                
                if temp_score < current_score:
                    current_values = temp_vals
                    current_edges = temp_edges
                    current_score = temp_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_values = current_values[:]
                        best_edges = [e[:] for e in current_edges]
                    break # Restart value swap loop
            
            # 2. Edge Swap
            # If no value improvement, try edge swap
            if not improved:
                # Find non-edges
                edge_set = set(frozenset(e) for e in current_edges)
                non_edges = [p for p in all_pairs if frozenset(p) not in edge_set]
                
                if non_edges:
                    for _ in range(5):
                        e_idx = random.randint(0, m - 1)
                        new_edge = random.choice(non_edges)
                        
                        temp_edges = current_edges[:]
                        temp_edges[e_idx] = list(new_edge)
                        
                        temp_score = get_score(current_values, temp_edges)
                        
                        if temp_score < current_score:
                            current_edges = temp_edges
                            current_score = temp_score
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_values = current_values[:]
                                best_edges = [e[:] for e in current_edges]
                            break

    # Final verification and type conversion
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = [list(e) for e in list(itertools.combinations(range(n), 2))[:m]]
    else:
        # Re-optimize edges for the best values found
        best_edges = find_best_edges(best_values)
        
    # Ensure types are Python int and list
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
