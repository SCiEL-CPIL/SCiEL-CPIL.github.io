import time
import random
from itertools import combinations
from collections import Counter
import math

DEADLINE = 85.0

def _evaluate_edge_set(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def _optimize_edges(values, m, deadline):
    """
    Optimize the set of m edges for the given values.
    Uses Simulated Annealing with a final polish phase.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_edges = len(all_edges)
    
    # Precompute sums and products for all edges
    edge_sums = [values[i] + values[j] for i, j in all_edges]
    edge_prods = [values[i] * values[j] for i, j in all_edges]
    
    def _score_from_counts(sum_counts, prod_counts):
        distinct_sums = sum(1 for v in sum_counts.values() if v > 0)
        distinct_prods = sum(1 for v in prod_counts.values() if v > 0)
        return max(distinct_sums, distinct_prods)

    best_score = float('inf')
    best_edges_set = None
    
    rng = random.Random(42)
    
    for restart in range(8):
        if time.time() > deadline:
            break
            
        if restart == 0:
            # Greedy initialization: add edges that minimize cost increase
            current_edges_set = set()
            sum_counts = Counter()
            prod_counts = Counter()
            remaining = list(range(num_edges))
            rng.shuffle(remaining)
            
            for _ in range(m):
                best_idx = -1
                best_cost = float('inf')
                for idx in remaining:
                    s = edge_sums[idx]
                    p = edge_prods[idx]
                    
                    old_s = sum_counts[s]
                    old_p = prod_counts[p]
                    
                    sum_counts[s] = old_s + 1
                    prod_counts[p] = old_p + 1
                    
                    cost = _score_from_counts(sum_counts, prod_counts)
                    
                    # Revert
                    if old_s == 0:
                        del sum_counts[s]
                    else:
                        sum_counts[s] = old_s
                    if old_p == 0:
                        del prod_counts[p]
                    else:
                        prod_counts[p] = old_p
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = idx
                        
                if best_idx != -1:
                    s = edge_sums[best_idx]
                    p = edge_prods[best_idx]
                    sum_counts[s] += 1
                    prod_counts[p] += 1
                    current_edges_set.add(best_idx)
                    remaining.remove(best_idx)
                    
        elif restart == 1:
            # Random initialization
            current_edges_set = set(rng.sample(range(num_edges), m))
            sum_counts = Counter()
            prod_counts = Counter()
            for idx in current_edges_set:
                sum_counts[edge_sums[idx]] += 1
                prod_counts[edge_prods[idx]] += 1
                
        else:
            # Perturbation of best found so far
            if best_edges_set is None:
                continue
            current_edges_set = set(best_edges_set)
            sum_counts = Counter()
            prod_counts = Counter()
            for idx in current_edges_set:
                sum_counts[edge_sums[idx]] += 1
                prod_counts[edge_prods[idx]] += 1
            
            # Randomly swap k edges
            k = 2 + restart
            for _ in range(k):
                if not current_edges_set:
                    continue
                in_list = list(current_edges_set)
                out_list = [i for i in range(num_edges) if i not in current_edges_set]
                if not out_list:
                    continue
                idx_in = rng.choice(in_list)
                idx_out = rng.choice(out_list)
                
                s_in = edge_sums[idx_in]
                p_in = edge_prods[idx_in]
                s_out = edge_sums[idx_out]
                p_out = edge_prods[idx_out]
                
                # Remove idx_in
                sum_counts[s_in] -= 1
                if sum_counts[s_in] == 0: del sum_counts[s_in]
                prod_counts[p_in] -= 1
                if prod_counts[p_in] == 0: del prod_counts[p_in]
                current_edges_set.remove(idx_in)
                
                # Add idx_out
                sum_counts[s_out] += 1
                prod_counts[p_out] += 1
                current_edges_set.add(idx_out)

        current_score = _score_from_counts(sum_counts, prod_counts)
        
        if current_score < best_score:
            best_score = current_score
            best_edges_set = set(current_edges_set)
            
        # Simulated Annealing
        T = 5.0
        T_min = 0.01
        alpha = 0.995
        iters = 0
        max_iters = 3000
        no_improve = 0
        patience = 500
        
        while iters < max_iters:
            if time.time() > deadline:
                break
            
            in_list = list(current_edges_set)
            out_list = [i for i in range(num_edges) if i not in current_edges_set]
            if not in_list or not out_list:
                break
                
            idx_in = rng.choice(in_list)
            idx_out = rng.choice(out_list)
            
            s_in = edge_sums[idx_in]
            p_in = edge_prods[idx_in]
            s_out = edge_sums[idx_out]
            p_out = edge_prods[idx_out]
            
            # Update counters
            old_s_in = sum_counts[s_in]
            old_p_in = prod_counts[p_in]
            old_s_out = sum_counts[s_out]
            old_p_out = prod_counts[p_out]
            
            # Remove in
            if old_s_in == 1:
                del sum_counts[s_in]
            else:
                sum_counts[s_in] = old_s_in - 1
            if old_p_in == 1:
                del prod_counts[p_in]
            else:
                prod_counts[p_in] = old_p_in - 1
            
            # Add out
            sum_counts[s_out] = old_s_out + 1
            prod_counts[p_out] = old_p_out + 1
            
            new_score = _score_from_counts(sum_counts, prod_counts)
            
            accept = False
            if new_score < current_score:
                accept = True
            elif new_score == current_score:
                accept = True
            else:
                delta = new_score - current_score
                accept = rng.random() < math.exp(-delta / max(T, 1e-10))
            
            if accept:
                current_score = new_score
                current_edges_set.remove(idx_in)
                current_edges_set.add(idx_out)
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_edges_set = set(current_edges_set)
            else:
                # Revert
                if old_s_out == 0:
                    del sum_counts[s_out]
                else:
                    sum_counts[s_out] = old_s_out
                if old_p_out == 0:
                    del prod_counts[p_out]
                else:
                    prod_counts[p_out] = old_p_out
                sum_counts[s_in] = old_s_in
                prod_counts[p_in] = old_p_in
                no_improve += 1
            
            T *= alpha
            if T < T_min:
                T = T_min
            iters += 1
            
            if no_improve > patience:
                break
    
    if best_edges_set is None:
        best_edges_set = set(range(m))
        best_score = _evaluate_edge_set(values, [all_edges[i] for i in best_edges_set])
    
    return [list(all_edges[idx]) for idx in best_edges_set], best_score

def construct():
    start_time = time.time()
    n = 14
    m = 35
    
    # Generate candidate value sets with emphasis on smooth numbers and structured sets
    candidates = []
    
    # Set 1: Small integers
    candidates.append(list(range(1, 15)))
    
    # Set 2: Start from 2
    candidates.append(list(range(2, 16)))
    
    # Set 3: Smooth numbers (highly composite-like)
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36])
    
    # Set 4: Powers of 2 and 3 mixed
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 48])
    
    # Set 5: More dense smooth
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    
    # Set 6: Arithmetic progression with step 2
    candidates.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    
    # Set 7: Mixed smooth
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 27])
    
    # Set 8: Another smooth set
    candidates.append([1, 2, 3, 4, 6, 9, 12, 16, 18, 24, 27, 32, 36, 48])
    
    # Set 9: Dense small
    candidates.append([1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 18, 20, 24, 27])
    
    # Set 10: Another variant
    candidates.append([1, 2, 3, 4, 6, 9, 12, 15, 18, 20, 24, 27, 30, 36])
    
    # Set 11: Geometric-ish
    candidates.append([1, 2, 3, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30])
    
    # Set 12: Very smooth
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 27, 36, 48, 54])
    
    # Set 13: Dense arithmetic
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18])
    
    # Set 14: Another smooth
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 36, 48])
    
    # Set 15: Small with some gaps
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16])
    
    # Add some random dense sets
    rng = random.Random(123)
    for _ in range(15):
        # Generate random set biased towards small numbers
        vals = set()
        while len(vals) < n:
            if rng.random() < 0.6:
                vals.add(rng.randint(1, 20))
            elif rng.random() < 0.8:
                vals.add(rng.randint(1, 40))
            else:
                vals.add(rng.randint(1, 60))
        if len(vals) == n and max(vals) <= 10000:
            candidates.append(sorted(vals))
    
    # Deduplicate candidates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    candidates = unique_candidates
    
    best_score = float('inf')
    best_result = None
    
    for values in candidates:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        if len(values) != n or len(set(values)) != n:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
        if not all(isinstance(v, int) for v in values):
            continue
            
        deadline_for_this = min(start_time + DEADLINE - 2, time.time() + 6.0)
        edges, score = _optimize_edges(values, m, deadline_for_this)
        
        if score is not None and score < best_score:
            best_score = score
            best_result = {
                'values': [int(v) for v in values],
                'edges': [[int(i), int(j)] for i, j in edges]
            }
            
    if best_result is None:
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_result = {'values': values, 'edges': edges}
        
    return best_result
