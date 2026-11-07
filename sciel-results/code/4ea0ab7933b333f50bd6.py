import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = 14
    num_edges = 35
    
    # Helper to check time
    def time_check():
        return time.time() > deadline

    # Helper to compute score
    def compute_score(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    # Helper to validate
    def is_valid(values):
        if len(values) != n:
            return False
        if len(set(values)) != n:
            return False
        if any(not isinstance(v, int) for v in values):
            return False
        if any(v < 1 or v > 10000 for v in values):
            return False
        return True

    # Generate initial diverse candidate value sets
    # We want sets that naturally have collisions.
    # 1. Arithmetic Progression: v_i = a + i*d
    # 2. Geometric-like but small: v_i = base^i (bad for sums, good for prods? No, Sidon)
    # 3. Multiplicative Grid: v_i = x * y where x,y from small sets
    # 4. Random with structure: Pick from a small pool of "factor-rich" numbers
    
    def generate_initial_values():
        candidates = []
        
        # Strategy A: Arithmetic Progression
        # v_i = start + i * step
        for start in [1, 2, 3, 5, 10, 100]:
            for step in [1, 2, 3, 5, 10, 20, 50]:
                vals = [start + i * step for i in range(n)]
                if is_valid(vals):
                    candidates.append(vals)
        
        # Strategy B: Multiplicative Structure
        # v_i = a * b_i where b_i are from a small set of multipliers
        # To ensure distinctness, we need care.
        # Let's try: v_i = (i+1) * k
        for k in [1, 2, 3, 4, 5, 6, 7, 10, 12, 15, 20]:
            vals = [(i+1) * k for i in range(n)]
            if is_valid(vals):
                candidates.append(vals)
        
        # Strategy C: Factor-rich numbers
        # Numbers with many divisors often have product collisions?
        # Let's pick 14 distinct numbers from 1..100 that are highly composite or multiples
        pool = [i for i in range(1, 101) if i % 2 == 0 or i % 3 == 0 or i % 5 == 0]
        random.shuffle(pool)
        if len(pool) >= n:
            candidates.append(pool[:n])
            
        # Strategy D: Pure random (with restarts)
        for _ in range(20):
            vals = random.sample(range(1, 10001), n)
            candidates.append(vals)
            
        return candidates

    def generate_initial_edges(values):
        # Greedy: Pick edges that minimize distinct sums first, then products
        # Or just pick a random 35 edges
        all_edges = list(combinations(range(n), 2))
        random.shuffle(all_edges)
        return all_edges[:num_edges]
    
    def improve_edges(values, edges, max_iter=1000):
        """Local search on edges only for fixed values."""
        current_edges = set(edges)
        current_score = compute_score(values, edges)
        
        # Precompute sum/prod for all possible edges
        edge_props = {}
        for u, v in combinations(range(n), 2):
            edge_props[(u,v)] = (values[u]+values[v], values[u]*values[v])
            
        # Maintain counts for fast update
        sum_counts = {}
        prod_counts = {}
        for u, v in current_edges:
            s, p = edge_props[(u,v)]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
            
        distinct_sums = len(sum_counts)
        distinct_prods = len(prod_counts)
        
        def try_swap(remove_edge, add_edge):
            nonlocal sum_counts, prod_counts, distinct_sums, distinct_prods
            if add_edge in current_edges:
                return False
            
            u1, v1 = remove_edge
            s1, p1 = edge_props[remove_edge]
            u2, v2 = add_edge
            s2, p2 = edge_props[add_edge]
            
            # Remove
            sum_counts[s1] -= 1
            if sum_counts[s1] == 0:
                distinct_sums -= 1
            prod_counts[p1] -= 1
            if prod_counts[p1] == 0:
                distinct_prods -= 1
                
            # Add
            if s2 not in sum_counts:
                sum_counts[s2] = 1
                distinct_sums += 1
            else:
                sum_counts[s2] += 1
            if p2 not in prod_counts:
                prod_counts[p2] = 1
                distinct_prods += 1
            else:
                prod_counts[p2] += 1
                
            new_score = max(distinct_sums, distinct_prods)
            
            if new_score < current_score:
                current_edges.discard(remove_edge)
                current_edges.add(add_edge)
                return True
            else:
                # Revert
                sum_counts[s2] -= 1
                if sum_counts[s2] == 0:
                    distinct_sums -= 1
                prod_counts[p2] -= 1
                if prod_counts[p2] == 0:
                    distinct_prods -= 1
                    
                sum_counts[s1] += 1
                if sum_counts[s1] == 1:
                    distinct_sums += 1
                prod_counts[p1] += 1
                if prod_counts[p1] == 1:
                    distinct_prods += 1
                return False

        # Iterate
        for _ in range(max_iter):
            if time_check():
                break
            # Pick random edge to remove
            remove_edge = random.choice(list(current_edges))
            # Pick random edge to add
            add_edge = random.choice(list(edge_props.keys()))
            if add_edge == remove_edge:
                continue
            try_swap(remove_edge, add_edge)
            
        return list(current_edges), compute_score(values, list(current_edges))

    def improve_values(values, edges, max_iter=1000):
        """Local search on values only for fixed edges."""
        current_values = list(values)
        current_score = compute_score(current_values, edges)
        
        for _ in range(max_iter):
            if time_check():
                break
            idx = random.randint(0, n-1)
            # Perturb value
            # Try small changes to maintain "structure"
            delta = random.choice([-10, -5, -2, -1, 1, 2, 5, 10, 20, 50])
            new_val = current_values[idx] + delta
            
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in current_values:
                # Check if it's already there (and not at idx)
                if current_values.index(new_val) != idx:
                    continue
            
            old_val = current_values[idx]
            current_values[idx] = new_val
            new_score = compute_score(current_values, edges)
            
            if new_score < current_score:
                current_score = new_score
            else:
                current_values[idx] = old_val
                
        return current_values, current_score

    # Main Search Loop
    best_overall_score = float('inf')
    best_overall_result = None
    
    initial_values_pool = generate_initial_values()
    
    # We will run a few "annealing" runs from different starts
    num_runs = 5
    for run_idx in range(num_runs):
        if time_check():
            break
            
        # Pick a starting point
        start_vals = random.choice(initial_values_pool)
        start_edges = generate_initial_edges(start_vals)
        
        # Phase 1: Optimize Edges
        edges, score = improve_edges(start_vals, start_edges, max_iter=2000)
        
        # Phase 2: Optimize Values
        values, score = improve_values(start_vals, edges, max_iter=2000)
        
        # Phase 3: Alternate Optimization (Co-optimization)
        current_values = values
        current_edges = edges
        current_score = score
        
        for step in range(100):
            if time_check():
                break
                
            # Optimize Edges
            current_edges, score_e = improve_edges(current_values, current_edges, max_iter=500)
            if score_e < current_score:
                current_score = score_e
                if current_score < best_overall_score:
                    best_overall_score = current_score
                    best_overall_result = {
                        'values': list(current_values),
                        'edges': [list(e) for e in current_edges]
                    }
            else:
                # If no improvement, maybe try value optimization
                pass

            # Optimize Values
            current_values, score_v = improve_values(current_values, current_edges, max_iter=500)
            if score_v < current_score:
                current_score = score_v
                if current_score < best_overall_score:
                    best_overall_score = current_score
                    best_overall_result = {
                        'values': list(current_values),
                        'edges': [list(e) for e in current_edges]
                    }
                    
            # Random restart of values if stuck?
            if random.random() < 0.05:
                # Perturb all values slightly
                for i in range(n):
                    if random.random() < 0.5:
                        delta = random.randint(-5, 5)
                        nv = current_values[i] + delta
                        if 1 <= nv <= 10000 and nv not in current_values[:i] + current_values[i+1:]:
                            current_values[i] = nv
                # Re-evaluate
                current_score = compute_score(current_values, current_edges)
                if current_score < best_overall_score:
                    best_overall_score = current_score
                    best_overall_result = {
                        'values': list(current_values),
                        'edges': [list(e) for e in current_edges]
                    }

    # Fallback if no result found (should not happen)
    if best_overall_result is None:
        vals = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        best_overall_result = {'values': vals, 'edges': edges}

    # Final Verification
    values = best_overall_result['values']
    edges = best_overall_result['edges']
    
    # Ensure types are strict Python ints and lists
    best_overall_result['values'] = [int(v) for v in values]
    best_overall_result['edges'] = [[int(u), int(v)] for u, v in edges]
    
    assert len(best_overall_result['values']) == 14
    assert len(set(best_overall_result['values'])) == 14
    assert all(isinstance(v, int) for v in best_overall_result['values'])
    assert all(1 <= v <= 10000 for v in best_overall_result['values'])
    assert len(best_overall_result['edges']) == 35
    edge_set = set(tuple(sorted(e)) for e in best_overall_result['edges'])
    assert len(edge_set) == 35
    assert all(isinstance(i, int) for e in best_overall_result['edges'] for i in e)
    assert all(0 <= i < 14 for e in best_overall_result['edges'] for i in e)

    return best_overall_result
