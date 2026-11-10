import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def get_score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def score_edges_given_values(values):
    """Compute all edge data and return a function to score a given edge set."""
    n = len(values)
    edge_data = {}
    for i in range(n):
        for j in range(i+1, n):
            edge_data[(i,j)] = (values[i] + values[j], values[i] * values[j])
    return edge_data

def fast_edge_select(values, time_budget=2.0):
    """Select 35 edges minimizing max(distinct_sums, distinct_prods) using iterative refinement."""
    n = len(values)
    edge_data = score_edges_given_values(values)
    all_edges = list(combinations(range(n), 2))
    m = P['m']
    
    # Start with greedy
    chosen = set()
    sum_counts = {}
    prod_counts = {}
    
    # Score each edge by how "popular" its sum and product are
    all_sums = [edge_data[e][0] for e in all_edges]
    all_prods = [edge_data[e][1] for e in all_edges]
    
    sum_freq = {}
    prod_freq = {}
    for s in all_sums:
        sum_freq[s] = sum_freq.get(s, 0) + 1
    for p in all_prods:
        prod_freq[p] = prod_freq.get(p, 0) + 1
    
    # Greedy: pick edges with highest combined frequency
    scored_edges = sorted(all_edges, key=lambda e: -(sum_freq[edge_data[e][0]] + prod_freq[edge_data[e][1]]))
    
    for e in scored_edges:
        if len(chosen) >= m:
            break
        chosen.add(e)
        s, p = edge_data[e]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    distinct_sums = set(sum_counts.keys())
    distinct_prods = set(prod_counts.keys())
    current_score = max(len(distinct_sums), len(distinct_prods))
    
    # Local search: try to reduce the larger set
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        best_move = None
        best_score = current_score
        target_set = 'sums' if len(distinct_sums) >= len(distinct_prods) else 'prods'
        
        for e_out in list(chosen):
            s_out, p_out = edge_data[e_out]
            
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                del sum_counts[s_out]
                distinct_sums.discard(s_out)
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                del prod_counts[p_out]
                distinct_prods.discard(p_out)
            
            for e_in in all_edges:
                if e_in in chosen:
                    continue
                s_in, p_in = edge_data[e_in]
                new_sums = len(distinct_sums) + (0 if s_in in sum_counts else 1)
                new_prods = len(distinct_prods) + (0 if p_in in prod_counts else 1)
                new_score = max(new_sums, new_prods)
                
                if new_score < best_score:
                    best_score = new_score
                    best_move = (e_out, e_in)
            
            # Restore
            sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
            if s_out not in distinct_sums:
                distinct_sums.add(s_out)
            prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
            if p_out not in distinct_prods:
                distinct_prods.add(p_out)
        
        if best_move:
            e_out, e_in = best_move
            s_out, p_out = edge_data[e_out]
            s_in, p_in = edge_data[e_in]
            
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                del sum_counts[s_out]
                distinct_sums.discard(s_out)
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                del prod_counts[p_out]
                distinct_prods.discard(p_out)
            chosen.remove(e_out)
            
            sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
            if s_in not in distinct_sums:
                distinct_sums.add(s_in)
            prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
            if p_in not in distinct_prods:
                distinct_prods.add(p_in)
            chosen.add(e_in)
            
            current_score = best_score
            improved = True
    
    return [list(e) for e in chosen], current_score

def optimize_values_with_edges(initial_values, time_budget=5.0):
    """Optimize values via simulated annealing, re-optimizing edges at each step."""
    n = len(initial_values)
    current = list(initial_values)
    current_edges, current_score = fast_edge_select(current, time_budget=1.0)
    
    best = list(current)
    best_edges = current_edges
    best_score = current_score
    
    rng = random.Random(12345)
    start = time.time()
    temp = 10.0
    cooling = 0.995
    
    no_improve = 0
    while (time.time() - start) < time_budget:
        # Random perturbation: swap two values or add random offset
        move_type = rng.random()
        if move_type < 0.5:
            i, j = rng.sample(range(n), 2)
            current[i], current[j] = current[j], current[i]
            revert = lambda: current.__setitem__(i, current[j]) or current.__setitem__(j, current[i])
        elif move_type < 0.8:
            i = rng.randrange(n)
            delta = rng.randint(-5, 5)
            old_val = current[i]
            new_val = old_val + delta
            if new_val < 1 or new_val > 10000 or new_val in current:
                continue
            current[i] = new_val
            revert = lambda: current.__setitem__(i, old_val)
        else:
            i = rng.randrange(n)
            old_val = current[i]
            new_val = rng.randint(1, 10000)
            if new_val in current:
                continue
            current[i] = new_val
            revert = lambda: current.__setitem__(i, old_val)
        
        # Ensure all values are distinct positive integers <= 10000
        if len(set(current)) != n or any(v < 1 or v > 10000 for v in current):
            revert()
            continue
        
        new_edges, new_score = fast_edge_select(current, time_budget=0.5)
        
        delta_score = new_score - current_score
        if delta_score < 0 or rng.random() < pow(2.718281828, -delta_score / max(temp, 0.1)):
            current_score = new_score
            current_edges = new_edges
            if new_score < best_score:
                best_score = new_score
                best = list(current)
                best_edges = new_edges
                no_improve = 0
            else:
                no_improve += 1
        else:
            revert()
            no_improve += 1
        
        temp *= cooling
        if no_improve > 200:
            break
    
    return best, best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = P['n']
    
    # Diverse initial value sets
    initial_sets = [
        list(range(1, n+1)),
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],
        sorted(random.sample(range(1, 100), n)),
        sorted(random.sample(range(1, 500), n)),
        [2, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        # More structured sets
        [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],  # Will be deduplicated
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        [6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136],
    ]
    
    # Deduplicate and validate
    valid_initial_sets = []
    for vals in initial_sets:
        if len(set(vals)) == n and all(1 <= v <= 10000 for v in vals):
            valid_initial_sets.append(sorted(vals))
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Phase 1: Evaluate initial sets with edge optimization
    for vals in valid_initial_sets:
        if time.time() > deadline - 30:
            break
        edges, score = fast_edge_select(vals, time_budget=3.0)
        if score < best_score:
            best_score = score
            best_values = list(vals)
            best_edges = edges
    
    if best_values is None:
        best_values = list(range(1, n+1))
        best_edges, best_score = fast_edge_select(best_values, time_budget=5.0)
    
    # Phase 2: Simulated annealing on values
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        best_values, best_edges, best_score = optimize_values_with_edges(
            best_values, time_budget=min(remaining_time - 5, 40.0)
        )
    
    # Phase 3: Final edge optimization with more time
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        final_edges, final_score = fast_edge_select(best_values, time_budget=min(remaining_time - 2, 10.0))
        if final_score < best_score:
            best_score = final_score
            best_edges = final_edges
    
    # Final validation
    if best_values is None:
        best_values = list(range(1, n+1))
        best_edges, best_score = fast_edge_select(best_values, time_budget=5.0)
    
    return {'values': [int(v) for v in best_values], 'edges': [[int(a), int(b)] for a, b in best_edges]}
