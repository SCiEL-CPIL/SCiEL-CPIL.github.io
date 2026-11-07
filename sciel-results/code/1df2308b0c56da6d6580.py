import time
import random
from itertools import combinations
from collections import Counter
import math

DEADLINE = 88.0  # Safety margin for 90s limit

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
    
    # Number of restarts depends on time available
    # We will run a loop of restarts until deadline
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        if restart_count == 1:
            # Greedy initialization: add edges that minimize cost increase
            current_edges_set = set()
            sum_counts = Counter()
            prod_counts = Counter()
            remaining = list(range(num_edges))
            rng.shuffle(remaining)
            
            for _ in range(m):
                best_idx = -1
                best_cost = float('inf')
                # Check a subset of remaining to speed up? No, m is small, remaining is ~90.
                # Just check all remaining for simplicity as n is small.
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
                    
        elif restart_count == 2:
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
            k = 2 + (restart_count % 3)
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
        max_iters = 5000  # Increased iterations for deeper search per restart
        no_improve = 0
        patience = 800
        
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
            
            if no_improve > patience:
                break
                
            T *= alpha
            if T < T_min:
                T = T_min
            iters += 1
        
        # Polish phase: Greedy local search for the current best
        if best_edges_set and time.time() < deadline:
            current_edges_set = set(best_edges_set)
            sum_counts = Counter()
            prod_counts = Counter()
            for idx in current_edges_set:
                sum_counts[edge_sums[idx]] += 1
                prod_counts[edge_prods[idx]] += 1
            current_score = _score_from_counts(sum_counts, prod_counts)
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                in_list = list(current_edges_set)
                out_list = [i for i in range(num_edges) if i not in current_edges_set]
                for idx_in in in_list:
                    for idx_out in out_list:
                        s_in = edge_sums[idx_in]
                        p_in = edge_prods[idx_in]
                        s_out = edge_sums[idx_out]
                        p_out = edge_prods[idx_out]
                        
                        old_s_in = sum_counts[s_in]
                        old_p_in = prod_counts[p_in]
                        old_s_out = sum_counts[s_out]
                        old_p_out = prod_counts[p_out]
                        
                        if old_s_in == 1:
                            del sum_counts[s_in]
                        else:
                            sum_counts[s_in] = old_s_in - 1
                        if old_p_in == 1:
                            del prod_counts[p_in]
                        else:
                            prod_counts[p_in] = old_p_in - 1
                        
                        sum_counts[s_out] = old_s_out + 1
                        prod_counts[p_out] = old_p_out + 1
                        
                        new_score = _score_from_counts(sum_counts, prod_counts)
                        
                        if new_score < current_score:
                            current_score = new_score
                            current_edges_set.remove(idx_in)
                            current_edges_set.add(idx_out)
                            best_score = new_score
                            best_edges_set = set(current_edges_set)
                            improved = True
                            break
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
                    if improved:
                        break

    if best_edges_set is None:
        # Fallback
        best_edges_set = set(range(m))
        
    return [all_edges[idx] for idx in best_edges_set], best_score

def construct():
    start_time = time.time()
    n = 14
    m = 35
    
    # Candidate value sets
    # Focus on sets with many common factors and small values to encourage overlap in sums and products
    candidates = [
        list(range(1, 15)),
        list(range(2, 16)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 24],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 27],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        [1, 2, 3, 4, 6, 9, 12, 16, 18, 24, 27, 32, 36, 48],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 36, 48],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 18, 20, 24, 27],
        [1, 2, 3, 4, 6, 9, 12, 15, 18, 20, 24, 27, 30, 36],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 24, 25],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 40],
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 18, 20, 24, 25, 30],
    ]
    
    # Generate a few random dense sets with common factors
    rng = random.Random(123)
    for _ in range(20):
        base = rng.randint(1, 5)
        step = rng.randint(1, 3)
        vals = [base + i*step for i in range(n)]
        if len(set(vals)) == n and max(vals) <= 10000:
            candidates.append(vals)
            
    # Generate sets based on multiples of a small number
    for base in [1, 2, 3, 4, 5]:
        vals = [base * (i + 1) for i in range(n)]
        if max(vals) <= 10000:
            candidates.append(vals)
            
    # Generate sets with many small factors
    for _ in range(20):
        vals = []
        for _ in range(n):
            v = rng.randint(1, 50)
            while v in vals:
                v = rng.randint(1, 50)
            vals.append(v)
        if len(set(vals)) == n and max(vals) <= 10000:
            candidates.append(vals)

    best_score = float('inf')
    best_result = None
    
    # Sort candidates by a heuristic: prefer smaller values and more common factors
    # Simple heuristic: sum of values
    candidates.sort(key=lambda x: sum(x))
    
    for values in candidates:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        if len(values) != n or len(set(values)) != n:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
            
        # Allocate time based on how many candidates are left
        # Simple approach: fixed time per candidate, but stop if deadline near
        deadline_for_this = min(start_time + DEADLINE - 2, time.time() + 5.0)
        
        try:
            edges, score = _optimize_edges(values, m, deadline_for_this)
        except Exception:
            continue
        
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
