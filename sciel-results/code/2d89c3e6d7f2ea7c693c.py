"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def _evaluate_configuration(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _greedy_select_edges(values, m=35):
    n = len(values)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute sums and products for all edges
    edge_data = []
    for i, j in all_edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_data.append((i, j, s, p))
        
    selected_edges = []
    current_sums = set()
    current_prods = set()
    
    for _ in range(m):
        best_edge = None
        best_score = float('inf')
        best_sum_count = float('inf')
        best_prod_count = float('inf')
        
        for i, j, s, p in edge_data:
            if (i, j) in selected_edges:
                continue
            
            new_sum_count = len(current_sums) + (1 if s not in current_sums else 0)
            new_prod_count = len(current_prods) + (1 if p not in current_prods else 0)
            
            score = max(new_sum_count, new_prod_count)
            
            # Tie-breaking: prefer lower total counts, then lower individual counts
            total_count = new_sum_count + new_prod_count
            
            if score < best_score or (score == best_score and total_count < best_sum_count + best_prod_count) or (score == best_score and total_count == best_sum_count + best_prod_count and (new_sum_count < best_sum_count or (new_sum_count == best_sum_count and new_prod_count < best_prod_count))):
                best_score = score
                best_edge = (i, j)
                best_sum_count = new_sum_count
                best_prod_count = new_prod_count
                
        if best_edge is None:
            break
            
        selected_edges.append(best_edge)
        s, p = edge_data[all_edges.index(best_edge)][2], edge_data[all_edges.index(best_edge)][3]
        # Wait, all_edges.index is O(N). Better to store index or recompute.
        # Let's recompute s and p from best_edge to be safe and simple.
        i, j = best_edge
        current_sums.add(values[i] + values[j])
        current_prods.add(values[i] * values[j])
        
    return selected_edges

def construct():
    # Candidate value sets
    candidates = [
        list(range(1, 15)),
        list(range(2, 16)),
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 20, 24, 30, 36],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
    ]
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    for vals in candidates:
        # Ensure distinct positive integers <= 10000
        if len(set(vals)) != 14:
            continue
        if any(v <= 0 or v > 10000 or not isinstance(v, int) for v in vals):
            continue
            
        edges = _greedy_select_edges(vals, 35)
        
        # Verify edges are valid
        if len(edges) != 35:
            continue
        if len(set(tuple(e) for e in edges)) != 35:
            continue
            
        score = _evaluate_configuration(vals, edges)
        
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = edges
            
    # Fallback if something went wrong
    if best_values is None:
        best_values = list(range(1, 15))
        best_edges = _greedy_select_edges(best_values, 35)
        
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
