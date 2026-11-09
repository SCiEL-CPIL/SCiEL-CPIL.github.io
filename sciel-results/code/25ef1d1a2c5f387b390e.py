"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

def _evaluate_state(values, edges_set):
    """
    Fast evaluation using pre-computed lookup or direct calculation.
    """
    sums = set()
    prods = set()
    for u, v in edges_set:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return max(len(sums), len(prods))

def _generate_value_candidates():
    """
    Generate diverse sets of 14 distinct positive integers <= 10000.
    Focus on numbers that create additive and multiplicative collisions.
    """
    candidates = []
    
    # 1. Small integers 1-14 (high additive collision potential)
    candidates.append(list(range(1, 15)))
    
    # 2. Multiples of a common base (e.g., 1..14 * 2)
    candidates.append([i * 2 for i in range(1, 15)])
    
    # 3. Mix of small primes and composites
    candidates.append([2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20])
    
    # 4. Numbers with high factor overlap (multiples of 6)
    candidates.append([6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
    
    # 5. Random small numbers
    random.seed(123)
    base = list(range(1, 50))
    random.shuffle(base)
    candidates.append(base[:14])
    
    # 6. Another random set
    random.seed(456)
    base = list(range(1, 100))
    random.shuffle(base)
    candidates.append(base[:14])
    
    # 7. Geometric progression (multiplicative collisions)
    candidates.append([2**i for i in range(1, 15)])
    
    # 8. Arithmetic progression with small difference
    candidates.append([1 + i*3 for i in range(14)])
    
    # 9. Highly composite numbers
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 20, 24, 30, 36])
    
    # 10. Small primes
    candidates.append([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43])
    
    return candidates

def _greedy_edge_selection(values, n=14, m=35):
    """
    Greedily select edges to minimize the number of distinct sums and products.
    Start with no edges, add the edge that causes the least increase in distinct sums/products.
    """
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    
    current_edges = set()
    sums = set()
    prods = set()
    
    # We want exactly 35 edges.
    # Simple greedy: sort all edges by some heuristic? 
    # Better: iterate through random orders and pick the best candidate at each step.
    # To make it fast, we'll just try a few random orderings and pick the best resulting set.
    
    best_score = float('inf')
    best_edges = None
    
    for _ in range(10): # 10 restarts of greedy
        current_edges = set()
        sums = set()
        prods = set()
        
        remaining_edges = list(all_edges)
        random.shuffle(remaining_edges)
        
        for _ in range(m):
            best_edge = None
            best_delta = float('inf')
            best_new_sums = None
            best_new_prods = None
            
            # Check a subset of remaining edges to save time, or all if fast enough
            # 91 edges total, checking 91 * 35 steps is ~3000 evals. Very fast.
            for e in remaining_edges:
                u, v = e
                s = values[u] + values[v]
                p = values[u] * values[v]
                
                # Count how many NEW distinct sums and prods this edge adds
                new_sums = 0 if s in sums else 1
                new_prods = 0 if p in prods else 1
                
                # Heuristic: minimize sum of new distincts. 
                # Or maximize overlap.
                # Let's just minimize the increase in max(len(sums), len(prods))
                # Approximation: minimize new_sums + new_prods
                cost = new_sums + new_prods
                
                if cost < best_delta:
                    best_delta = cost
                    best_edge = e
                    best_new_sums = s
                    best_new_prods = p
            
            if best_edge:
                current_edges.add(best_edge)
                remaining_edges.remove(best_edge)
                sums.add(best_new_sums)
                prods.add(best_new_prods)
        
        score = max(len(sums), len(prods))
        if score < best_score:
            best_score = score
            best_edges = set(current_edges)
            
    return [list(e) for e in best_edges], best_score

def _local_search_refinement(values, edges_list, time_limit):
    """
    Refine the edge set using local search (swap moves).
    Focus on reducing the bottleneck (max of distinct sums/prods).
    """
    n = len(values)
    all_edges = set(combinations(range(n), 2))
    current_edges = set(tuple(sorted(e)) for e in edges_list)
    
    # Precompute sums and prods for current state
    sums = set()
    prods = set()
    for u, v in current_edges:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    
    current_score = max(len(sums), len(prods))
    best_score = current_score
    best_edges = set(current_edges)
    
    start_time = time.time()
    temp = 5.0
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to remove
        if not current_edges:
            break
        remove_edge = random.choice(list(current_edges))
        
        # Pick a random edge to add
        add_edge = random.choice(list(all_edges - current_edges)) if len(all_edges - current_edges) > 0 else None
        if not add_edge:
            break
            
        # Try the swap
        current_edges.remove(remove_edge)
        current_edges.add(add_edge)
        
        # Recalculate score efficiently?
        # For n=14, full recalc is fast.
        new_sums = set()
        new_prods = set()
        for u, v in current_edges:
            new_sums.add(values[u] + values[v])
            new_prods.add(values[u] * values[v])
        
        new_score = max(len(new_sums), len(new_prods))
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < pow(2.71828, -delta / temp):
            current_score = new_score
            sums = new_sums
            prods = new_prods
            if new_score < best_score:
                best_score = new_score
                best_edges = set(current_edges)
        else:
            # Revert
            current_edges.remove(add_edge)
            current_edges.add(remove_edge)
            
        temp *= 0.999
        if temp < 0.01:
            break
            
    return [list(e) for e in best_edges], best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_global_score = float('inf')
    best_global_result = None
    
    value_sets = _generate_value_candidates()
    
    for values in value_sets:
        # Ensure types are correct
        values = [int(v) for v in values]
        
        # Check constraints
        if len(set(values)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
            
        # 1. Greedy Initialization
        greedy_edges, greedy_score = _greedy_edge_selection(values)
        
        if greedy_score < best_global_score:
            best_global_score = greedy_score
            best_global_result = {
                'values': values,
                'edges': greedy_edges
            }
            
        # 2. Local Search Refinement
        if time.time() > deadline:
            break
            
        # Allocate time for refinement
        time_left = deadline - time.time()
        refine_time = min(5.0, time_left / 2.0) # Give up to 5s per value set
        
        refined_edges, refined_score = _local_search_refinement(values, greedy_edges, refine_time)
        
        if refined_score < best_global_score:
            best_global_score = refined_score
            best_global_result = {
                'values': values,
                'edges': refined_edges
            }
            
        # Early exit if we hit a very low score
        if best_global_score <= 10:
            break

    if best_global_result is None:
        # Fallback
        return {
            'values': list(range(1, 15)),
            'edges': [list(e) for e in list(combinations(range(14), 2))[:35]]
        }
        
    return best_global_result
# EVOLVE-BLOCK-END
