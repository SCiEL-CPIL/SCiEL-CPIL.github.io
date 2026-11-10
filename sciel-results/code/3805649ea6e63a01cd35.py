import random
import time
from itertools import combinations
from collections import Counter

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    # Try a few value sets
    # Set 1: 1..14
    # Set 2: 1..13, 15
    # Set 3: Random small
    
    value_sets = [
        list(range(1, 15)),
        list(range(1, 14)) + [15],
        list(range(1, 14)) + [16],
    ]
    
    best_global_score = float('inf')
    best_global_values = None
    best_global_edges = None
    
    for values in value_sets:
        if time.time() - start_time > timeout - 5:
            break
            
        n = len(values)
        m = 35
        
        all_pairs = list(combinations(range(n), 2))
        pair_sum = [values[i] + values[j] for i, j in all_pairs]
        pair_prod = [values[i] * values[j] for i, j in all_pairs]
        pair_to_idx = {p: i for i, p in enumerate(all_pairs)}
        
        # Greedy Initialization
        current_edges = set()
        sums = Counter()
        prods = Counter()
        
        # Candidates: all pairs
        candidates = set(all_pairs)
        
        for _ in range(m):
            best_edge = None
            best_delta = float('inf')
            # To save time, don't check all candidates every step if too many
            # But 91 is small.
            for e in candidates:
                idx = pair_to_idx[e]
                s = pair_sum[idx]
                p = pair_prod[idx]
                
                # Cost: 1 if new sum, 0 if existing. 1 if new prod, 0 if existing.
                cost = (0 if sums[s] > 0 else 1) + (0 if prods[p] > 0 else 1)
                
                if cost < best_delta:
                    best_delta = cost
                    best_edge = e
            # If multiple edges have same cost, pick one randomly?
            # For simplicity, pick the first one found with min cost.
            # Better: collect all min cost edges and pick random.
            min_cost_edges = []
            for e in candidates:
                idx = pair_to_idx[e]
                s = pair_sum[idx]
                p = pair_prod[idx]
                cost = (0 if sums[s] > 0 else 1) + (0 if prods[p] > 0 else 1)
                if cost == best_delta:
                    min_cost_edges.append(e)
            
            chosen_edge = random.choice(min_cost_edges)
            current_edges.add(chosen_edge)
            candidates.remove(chosen_edge)
            
            idx = pair_to_idx[chosen_edge]
            sums[pair_sum[idx]] += 1
            prods[pair_prod[idx]] += 1
        
        current_score = max(len(sums), len(prods))
        best_score = current_score
        best_edges = set(current_edges)
        
        # Local Search
        # Directed swaps
        while time.time() - start_time < timeout - 5:
            # Find bad edges (unique sum or prod)
            bad_edges = []
            for e in current_edges:
                idx = pair_to_idx[e]
                s = pair_sum[idx]
                p = pair_prod[idx]
                if sums[s] == 1 or prods[p] == 1:
                    bad_edges.append(e)
            
            # Find good non-edges (sum or prod already present)
            good_non_edges = []
            for e in all_pairs:
                if e not in current_edges:
                    idx = pair_to_idx[e]
                    s = pair_sum[idx]
                    p = pair_prod[idx]
                    if sums[s] > 0 or prods[p] > 0:
                        good_non_edges.append(e)
            
            if bad_edges and good_non_edges:
                e1 = random.choice(bad_edges)
                e2 = random.choice(good_non_edges)
                
                # Swap
                idx1 = pair_to_idx[e1]
                s1 = pair_sum[idx1]
                p1 = pair_prod[idx1]
                
                idx2 = pair_to_idx[e2]
                s2 = pair_sum[idx2]
                p2 = pair_prod[idx2]
                
                # Update counters
                sums[s1] -= 1
                prods[p1] -= 1
                sums[s2] += 1
                prods[p2] += 1
                
                new_score = max(len(sums), len(prods))
                
                if new_score < current_score:
                    current_edges.remove(e1)
                    current_edges.add(e2)
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = set(current_edges)
                else:
                    # Revert
                    sums[s1] += 1
                    prods[p1] += 1
                    sums[s2] -= 1
                    prods[p2] -= 1
            else:
                # Stuck, do random swap
                e1 = random.choice(list(current_edges))
                e2 = random.choice(all_pairs)
                while e2 in current_edges:
                    e2 = random.choice(all_pairs)
                
                idx1 = pair_to_idx[e1]
                s1 = pair_sum[idx1]
                p1 = pair_prod[idx1]
                
                idx2 = pair_to_idx[e2]
                s2 = pair_sum[idx2]
                p2 = pair_prod[idx2]
                
                sums[s1] -= 1
                prods[p1] -= 1
                sums[s2] += 1
                prods[p2] += 1
                
                new_score = max(len(sums), len(prods))
                
                if new_score < current_score:
                    current_edges.remove(e1)
                    current_edges.add(e2)
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = set(current_edges)
                else:
                    sums[s1] += 1
                    prods[p1] += 1
                    sums[s2] -= 1
                    prods[p2] -= 1
        
        if best_score < best_global_score:
            best_global_score = best_score
            best_global_values = values
            best_global_edges = best_edges
    
    return {'values': best_global_values, 'edges': [list(e) for e in best_global_edges]}

# EVOLVE-BLOCK-END
