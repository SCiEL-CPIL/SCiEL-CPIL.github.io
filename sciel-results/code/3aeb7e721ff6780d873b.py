import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    n = 14
    m = 35
    values = list(range(1, n + 1))
    
    # Precompute sums and products for all edges
    all_edges = list(combinations(range(n), 2))
    edge_sums = {}
    edge_prods = {}
    for u, v in all_edges:
        s = values[u] + values[v]
        p = values[u] * values[v]
        edge_sums[(u, v)] = s
        edge_prods[(u, v)] = p

    # 1. Greedy Initialization
    current_edges = []
    sum_counts = {}
    prod_counts = {}
    
    def add_edge(e):
        u, v = e
        s = edge_sums[e]
        p = edge_prods[e]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        current_edges.append(e)

    def remove_edge(e):
        u, v = e
        s = edge_sums[e]
        p = edge_prods[e]
        sum_counts[s] -= 1
        if sum_counts[s] == 0:
            del sum_counts[s]
        prod_counts[p] -= 1
        if prod_counts[p] == 0:
            del prod_counts[p]
        current_edges.remove(e)

    # Greedy selection
    available_edges = list(all_edges)
    for _ in range(m):
        best_edge = None
        best_score = float('inf')
        for e in available_edges:
            s = edge_sums[e]
            p = edge_prods[e]
            # Simulate adding
            new_sum_count = len(sum_counts) + (0 if s in sum_counts else 1)
            new_prod_count = len(prod_counts) + (0 if p in prod_counts else 1)
            score = max(new_sum_count, new_prod_count)
            if score < best_score:
                best_score = score
                best_edge = e
        
        if best_edge is not None:
            add_edge(best_edge)
            available_edges.remove(best_edge)

    current_score = max(len(sum_counts), len(prod_counts))
    best_score = current_score
    best_edges = list(current_edges)
    best_sum_counts = dict(sum_counts)
    best_prod_counts = dict(prod_counts)

    # 2. Local Search with Simulated Annealing
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute list of non-edges for faster sampling? 
    # Or just sample from all_edges and check.
    # With 91 edges, 35 in set, 56 out. 
    # Checking `if e in current_edges` is O(M) if list, O(1) if set.
    # Let's use a set for current edges for fast lookup.
    current_edge_set = set(current_edges)
    
    while time.time() < deadline and T > min_T:
        # Pick a random edge to remove
        idx_remove = random.randint(0, m - 1)
        e_remove = current_edges[idx_remove]
        
        # Pick a random edge to add
        while True:
            e_add = random.choice(all_edges)
            if e_add not in current_edge_set:
                break
                
        # Perform swap
        # Remove
        s_rm = edge_sums[e_remove]
        p_rm = edge_prods[e_remove]
        sum_counts[s_rm] -= 1
        if sum_counts[s_rm] == 0: del sum_counts[s_rm]
        prod_counts[p_rm] -= 1
        if prod_counts[p_rm] == 0: del prod_counts[p_rm]
        
        # Add
        s_add = edge_sums[e_add]
        p_add = edge_prods[e_add]
        sum_counts[s_add] = sum_counts.get(s_add, 0) + 1
        prod_counts[p_add] = prod_counts.get(p_add, 0) + 1
        
        new_score = max(len(sum_counts), len(prod_counts))
        
        # Acceptance criterion
        if new_score < current_score:
            accept = True
        elif new_score == current_score:
            accept = random.random() < 0.5 # Plateau wandering
        else:
            # Simulated Annealing
            delta = new_score - current_score
            accept = random.random() < pow(2.71828, -delta / T)
            
        if accept:
            current_edges[idx_remove] = e_add
            current_edge_set.remove(e_remove)
            current_edge_set.add(e_add)
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_edges = list(current_edges)
                best_sum_counts = dict(sum_counts)
                best_prod_counts = dict(prod_counts)
        else:
            # Revert
            # Add back removed
            sum_counts[s_rm] = sum_counts.get(s_rm, 0) + 1
            prod_counts[p_rm] = prod_counts.get(p_rm, 0) + 1
            # Remove added
            sum_counts[s_add] -= 1
            if sum_counts[s_add] == 0: del sum_counts[s_add]
            prod_counts[p_add] -= 1
            if prod_counts[p_add] == 0: del prod_counts[p_add]
            
            current_edges[idx_remove] = e_remove
            current_edge_set.remove(e_add)
            current_edge_set.add(e_remove)
            
        T *= cooling

    return {
        'values': values,
        'edges': [list(e) for e in best_edges]
    }
