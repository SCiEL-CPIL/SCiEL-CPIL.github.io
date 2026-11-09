import time
import random
from itertools import combinations

def _raw_score(values, edges):
    """Compute the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        su = values[u]
        sv = values[v]
        sums.add(su + sv)
        prods.add(su * sv)
    return max(len(sums), len(prods))

def _greedy_edge_selection(values, time_limit):
    """
    Greedily select 35 edges that minimize the growth of distinct sums and products.
    At each step, choose the edge (u, v) not yet selected that minimizes the new
    distinct counts. Ties are broken randomly to explore different structures.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sums and products
    edge_sums = {}
    edge_prods = {}
    for u, v in all_edges:
        edge_sums[(u, v)] = values[u] + values[v]
        edge_prods[(u, v)] = values[u] * values[v]
        
    # State
    selected_edges = []
    selected_set = set()
    current_sums = set()
    current_prods = set()
    
    # For efficiency, we can just iterate. 
    # Since N=14, total edges = 91. We pick 35.
    # A simple O(35 * 91) loop is very fast.
    
    # To add some diversity, we can shuffle the candidate order if scores tie.
    # Or we can do a few restarts with different random seeds for tie-breaking.
    
    best_score = float('inf')
    best_edges = None
    
    start_time = time.time()
    
    # We will run a few iterations of greedy with different tie-breaking randomness
    # to find a robust local minimum.
    
    for restart in range(20):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(restart) # Fixed seed for reproducibility within restart, but different per restart
        sel_edges = []
        sel_set = set()
        c_sums = set()
        c_prods = set()
        
        for _ in range(35):
            best_edge = None
            best_score_candidate = float('inf')
            
            # Collect all valid candidates
            candidates = []
            for e in all_edges:
                if e in sel_set:
                    continue
                s = edge_sums[e]
                p = edge_prods[e]
                
                # Calculate score if we add this edge
                # New distinct sums = len(c_sums) + (0 if s in c_sums else 1)
                # New distinct prods = len(c_prods) + (0 if p in c_prods else 1)
                # We want to minimize max(new_sums, new_prods)
                
                new_sums_count = len(c_sums) + (0 if s in c_sums else 1)
                new_prods_count = len(c_prods) + (0 if p in c_prods else 1)
                
                score = max(new_sums_count, new_prods_count)
                
                candidates.append((score, e))
            
            # Sort by score, then randomly pick from the best ones to allow diversity
            # Actually, just picking the absolute best is the standard greedy.
            # Let's pick the best, but if there are many ties, pick randomly among them.
            candidates.sort(key=lambda x: x[0])
            
            if not candidates:
                break
                
            min_score = candidates[0][0]
            best_candidates = [c[1] for c in candidates if c[0] == min_score]
            
            # Pick one randomly from the best candidates
            chosen_edge = random.choice(best_candidates)
            
            sel_edges.append(chosen_edge)
            sel_set.add(chosen_edge)
            
            s = edge_sums[chosen_edge]
            p = edge_prods[chosen_edge]
            c_sums.add(s)
            c_prods.add(p)
            
        if len(sel_edges) == 35:
            final_score = max(len(c_sums), len(c_prods))
            if final_score < best_score:
                best_score = final_score
                best_edges = [list(e) for e in sel_edges]
                
    if best_edges is None:
        # Fallback: just take first 35
        best_edges = [list(e) for e in all_edges[:35]]
        best_score = _raw_score(values, best_edges)
        
    return best_edges, best_score

def _generate_value_candidates():
    """
    Generate value sets specifically designed to maximize collisions in sums and products.
    """
    candidates = []
    
    # 1. Small integers 1-14
    candidates.append(list(range(1, 15)))
    
    # 2. Multiples of 2
    candidates.append([i * 2 for i in range(1, 15)])
    
    # 3. Multiples of 3
    candidates.append([i * 3 for i in range(1, 15)])
    
    # 4. Highly composite small numbers
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 20, 24, 30, 36])
    
    # 5. Arithmetic progression with diff 1 (1-14) is already there.
    # Try diff 2
    candidates.append([1 + i*2 for i in range(14)])
    
    # 6. Values that share many factors (multiples of 6)
    candidates.append([6*i for i in range(1, 15)])
    
    # 7. Random small numbers with seed
    random.seed(123)
    base = list(range(1, 50))
    random.shuffle(base)
    candidates.append(base[:14])
    
    # 8. Another random set
    random.seed(456)
    base = list(range(1, 100))
    random.shuffle(base)
    candidates.append(base[:14])
    
    # 9. Mixed small primes and composites
    candidates.append([2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20])
    
    # 10. Values centered around 10
    candidates.append([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    
    # 11. Values with many divisors
    candidates.append([6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
    
    # 12. Random 1-20
    random.seed(789)
    base = list(range(1, 21))
    random.shuffle(base)
    candidates.append(base[:14])
    
    # 13. Powers of 2 (bad for products, good for sums? No, products are unique)
    # Skip powers of 2 for products.
    
    # 14. Small numbers 1-14 but shuffled? No, values are a set.
    
    # 15. Try numbers that are close to each other
    candidates.append([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    
    # 16. Try numbers that are multiples of a common factor but small
    candidates.append([4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56])
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_global_score = float('inf')
    best_global_result = None
    
    value_sets = _generate_value_candidates()
    
    # Shuffle value sets to avoid bias
    random.seed(100)
    random.shuffle(value_sets)
    
    for values in value_sets:
        # Ensure types are correct
        values = [int(v) for v in values]
        
        # Check constraints
        if len(set(values)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
            
        # Allocate time for edge optimization
        time_left = deadline - time.time()
        if time_left < 1.0:
            break
            
        # Give each value set a chunk of time
        edge_time = min(2.0, time_left / 2.0)
        
        edges, score = _greedy_edge_selection(values, edge_time)
        
        if score < best_global_score:
            best_global_score = score
            best_global_result = {
                'values': values,
                'edges': edges
            }
            
        # Early exit if we hit a very low score
        if best_global_score <= 10:
            break

    if best_global_result is None:
        # Fallback
        values = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        return {
            'values': values,
            'edges': edges
        }
        
    return best_global_result
