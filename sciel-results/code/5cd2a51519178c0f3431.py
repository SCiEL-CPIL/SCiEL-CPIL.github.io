"""Direct constructive approach for small sums and products along selected graph edges."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin for 90s limit
    
    n = 14
    m = 35
    max_val = 10000
    all_edges = list(combinations(range(n), 2))
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        ns = len(sums)
        np_ = len(prods)
        return ns, np_, max(ns, np_)
    
    def find_edges_greedy_pruning(values, rng, time_limit):
        """
        Greedy edge selection that prioritizes edges sharing existing sums/products.
        This helps keep the distinct counts low.
        """
        sums_seen = set()
        prods_seen = set()
        selected = []
        remaining = list(all_edges)
        rng.shuffle(remaining)
        
        # We don't strictly enforce a max_distinct limit that might prevent finding 35 edges,
        # but we strongly prefer edges that don't add new distinct values.
        
        for _ in range(m):
            if time.time() > time_limit:
                break
                
            best_edge = None
            best_score = float('inf')
            
            for edge in remaining:
                u, v = edge
                s = values[u] + values[v]
                p = values[u] * values[v]
                
                sum_new = s not in sums_seen
                prod_new = p not in prods_seen
                
                # Score: lower is better
                # Heavy penalty for adding new distinct sums/prods
                score = 0
                if sum_new:
                    score += 100
                if prod_new:
                    score += 100
                
                # Tie-breaker: prefer smaller sums/prods (might help with future collisions? 
                # Actually, smaller values are generally "denser" so maybe prefer larger? 
                # No, let's just use a small epsilon based on value to stabilize)
                score += (s + p) / 100000.0
                
                if score < best_score:
                    best_score = score
                    best_edge = edge
            
            if best_edge is None:
                # Fallback: pick any remaining edge
                best_edge = remaining[0]
            
            selected.append(best_edge)
            remaining.remove(best_edge)
            u, v = best_edge
            sums_seen.add(values[u] + values[v])
            prods_seen.add(values[u] * values[v])
        
        # Fill if needed (should not happen if m <= len(all_edges))
        if len(selected) < m:
            used = set(tuple(e) for e in selected)
            for e in all_edges:
                if len(selected) >= m:
                    break
                if tuple(e) not in used:
                    selected.append(e)
                    used.add(tuple(e))
        
        return selected

    def local_search(values, edges, rng, time_limit):
        """
        Local search: random swaps of edges and values to reduce max(distinct_sums, distinct_prods).
        """
        edges_set = set(tuple(e) for e in edges)
        _, _, current_score = evaluate(values, edges)
        
        improved = True
        # Limit iterations to ensure we don't waste too much time on one strategy
        max_iterations = 500 
        iterations = 0
        
        while improved and (time.time() - start_time) < time_limit and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try a few random moves
            for _ in range(20):
                if (time.time() - start_time) > time_limit:
                    break
                
                if rng.random() < 0.5:
                    # Edge swap
                    i = rng.randint(0, len(edges)-1)
                    new_edge = rng.choice(all_edges)
                    if tuple(new_edge) in edges_set:
                        continue
                    
                    old_edge = edges[i]
                    edges[i] = new_edge
                    edges_set.discard(tuple(old_edge))
                    edges_set.add(tuple(new_edge))
                    
                    ns, np_, new_score = evaluate(values, edges)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        edges[i] = old_edge
                        edges_set.discard(tuple(new_edge))
                        edges_set.add(tuple(old_edge))
                else:
                    # Value change
                    idx = rng.randint(0, n-1)
                    old_val = values[idx]
                    
                    # Generate candidates
                    candidates = set()
                    # Neighbors
                    for delta in range(-5, 6):
                        v = old_val + delta
                        if 1 <= v <= max_val:
                            candidates.add(v)
                    # Multiples of small bases
                    for base in [2, 3, 4, 5, 6, 8, 10, 12]:
                        # Try values close to old_val that are multiples of base
                        mult = max(1, old_val // base)
                        for m_offset in range(-2, 3):
                            v = base * (mult + m_offset)
                            if 1 <= v <= max_val:
                                candidates.add(v)
                    # Random
                    for _ in range(5):
                        candidates.add(rng.randint(1, max_val))
                    
                    best_new_val = None
                    best_new_score = current_score
                    for new_val in sorted(candidates):
                        if new_val < 1 or new_val > max_val:
                            continue
                        if new_val in values:
                            continue
                        
                        values[idx] = new_val
                        ns, np_, new_score = evaluate(values, edges)
                        if new_score < best_new_score:
                            best_new_score = new_score
                            best_new_val = new_val
                        else:
                            values[idx] = old_val
                            break
                        values[idx] = old_val
                    
                    if best_new_val is not None:
                        values[idx] = best_new_val
                        current_score = best_new_score
                        improved = True
        
        return values, edges, current_score

    best_overall_score = float('inf')
    best_result = None
    rng = random.Random(42)
    
    # Value strategies:
    # Using highly composite numbers or multiples of common factors helps create collisions in sums and products.
    value_strategies = [
        # Multiples of 6
        [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84],
        # Multiples of 12
        [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168],
        # Multiples of 4
        [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56],
        # Multiples of 2
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        # Small integers
        list(range(1, 15)),
        # Powers of 2 and 3 mixed
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        # Arithmetic progression step 3
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
        # Arithmetic progression step 5
        [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    ]
    
    for init_values in value_strategies:
        if (time.time() - start_time) > deadline - 5.0:
            break
        
        values = list(init_values)
        # Ensure distinct
        if len(set(values)) != n:
            # Perturb if not distinct
            for i in range(n):
                while values[i] in values[:i] or values[i] in values[i+1:]:
                    values[i] += 1
            if max(values) > max_val:
                continue
        
        edges = find_edges_greedy_pruning(values, rng, deadline - 5.0)
        
        # Local search
        values, edges, score = local_search(values, edges, rng, deadline - 2.0)
        
        if score < best_overall_score:
            best_overall_score = score
            best_result = {
                'values': list(values),
                'edges': [list(e) for e in edges]
            }
            
        # If we have a very good score, we can stop early or do more polish
        if best_overall_score <= 10:
             # Do a final polish pass with more iterations
             if (time.time() - start_time) < deadline - 2.0:
                 values, edges, score = local_search(values, edges, rng, deadline - 1.0)
                 if score < best_overall_score:
                     best_overall_score = score
                     best_result = {
                         'values': list(values),
                         'edges': [list(e) for e in edges]
                     }

    if best_result is None:
        # Fallback
        best_result = {
            'values': list(range(1, n+1)),
            'edges': [list(e) for e in all_edges[:m]]
        }
    
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return {'values': values, 'edges': edges}

# EVOLVE-BLOCK-END
