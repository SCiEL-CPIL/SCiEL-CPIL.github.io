"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def _calc_metric(values, edges):
    """Calculate the raw value: max(distinct_sums, distinct_products)."""
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def _optimize_edges(values, edges, time_limit=1.0):
    """
    Local search on the set of edges.
    Tries to swap an edge in and an edge out to improve the metric.
    Uses incremental updates for speed.
    """
    n = len(values)
    m = 35
    
    # Pre-calculate sums and products for all possible pairs
    all_pairs = list(itertools.combinations(range(n), 2))
    pair_data = {}
    for i, j in all_pairs:
        s = values[i] + values[j]
        p = values[i] * values[j]
        pair_data[(i, j)] = (s, p)
        
    # Current state
    current_edges = set(tuple(e) for e in edges)
    sum_counts = {}
    prod_counts = {}
    
    for e in current_edges:
        s, p = pair_data[e]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        
    distinct_sums = len(sum_counts)
    distinct_prods = len(prod_counts)
    current_metric = max(distinct_sums, distinct_prods)
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 500 # Stop if stuck
    
    while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
        # Pick a random edge to remove
        if not current_edges:
            break
        e_out = random.choice(list(current_edges))
        s_out, p_out = pair_data[e_out]
        
        # Pick a random edge to add (not currently in set)
        # To be efficient, we can pick from a pool of candidates
        # For simplicity, pick random from all pairs not in current_edges
        # 91 total pairs. 35 in set. 56 available.
        available_edges = [p for p in all_pairs if p not in current_edges]
        if not available_edges:
            break
        e_in = random.choice(available_edges)
        s_in, p_in = pair_data[e_in]
        
        # Calculate new metric
        # Remove e_out
        sum_counts[s_out] -= 1
        if sum_counts[s_out] == 0:
            del sum_counts[s_out]
            distinct_sums -= 1
            
        prod_counts[p_out] -= 1
        if prod_counts[p_out] == 0:
            del prod_counts[p_out]
            distinct_prods -= 1
            
        # Add e_in
        if s_in not in sum_counts:
            distinct_sums += 1
            sum_counts[s_in] = 1
        else:
            sum_counts[s_in] += 1
            
        if p_in not in prod_counts:
            distinct_prods += 1
            prod_counts[p_in] = 1
        else:
            prod_counts[p_in] += 1
            
        new_metric = max(distinct_sums, distinct_prods)
        
        if new_metric < current_metric:
            # Accept move
            current_edges.remove(e_out)
            current_edges.add(e_in)
            current_metric = new_metric
            no_improve_count = 0
        else:
            # Revert
            # Undo Add e_in
            sum_counts[s_in] -= 1
            if sum_counts[s_in] == 0:
                del sum_counts[s_in]
                distinct_sums -= 1
                
            prod_counts[p_in] -= 1
            if prod_counts[p_in] == 0:
                del prod_counts[p_in]
                distinct_prods -= 1
                
            # Undo Remove e_out
            if s_out not in sum_counts:
                distinct_sums += 1
                sum_counts[s_out] = 1
            else:
                sum_counts[s_out] += 1
                
            if p_out not in prod_counts:
                distinct_prods += 1
                prod_counts[p_out] = 1
            else:
                prod_counts[p_out] += 1
                
            no_improve_count += 1
            
    return [list(e) for e in current_edges]

def _optimize_values(values, edges, time_limit=1.0):
    """
    Local search on the values.
    Tries to change one value at a time.
    """
    n = len(values)
    current_values = list(values)
    current_metric = _calc_metric(current_values, edges)
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
        idx = random.randint(0, n - 1)
        old_val = current_values[idx]
        
        # Try a few random new values
        # Bias towards small numbers
        for _ in range(5):
            new_val = random.randint(1, 50)
            if new_val == old_val:
                continue
            # Check distinctness
            if any(v == new_val for i, v in enumerate(current_values) if i != idx):
                continue
                
            current_values[idx] = new_val
            new_metric = _calc_metric(current_values, edges)
            
            if new_metric < current_metric:
                current_metric = new_metric
                no_improve_count = 0
                break
            else:
                current_values[idx] = old_val
                no_improve_count += 1
                
    return current_values

def construct():
    n = 14
    m = 35
    start_time = time.time()
    time_limit = 55.0
    
    random.seed(42)
    
    # Initialize
    best_values = list(range(1, n + 1))
    
    # Initialize edges randomly
    all_pairs = list(itertools.combinations(range(n), 2))
    best_edges = [list(e) for e in random.sample(all_pairs, m)]
    
    best_metric = _calc_metric(best_values, best_edges)
    
    # Main Loop: Alternate between optimizing edges and values
    while time.time() - start_time < time_limit:
        # Optimize Edges (1 second)
        current_edges = _optimize_edges(best_values, best_edges, time_limit=1.0)
        current_metric = _calc_metric(best_values, current_edges)
        
        if current_metric < best_metric:
            best_edges = current_edges
            best_metric = current_metric
            
        # Optimize Values (1 second)
        current_values = _optimize_values(best_values, best_edges, time_limit=1.0)
        current_metric = _calc_metric(current_values, best_edges)
        
        if current_metric < best_metric:
            best_values = current_values
            best_metric = current_metric
            
    # Ensure types are correct
    return {
        'values': [int(v) for v in best_values],
        'edges': [[int(i), int(j)] for i, j in best_edges]
    }
# EVOLVE-BLOCK-END
