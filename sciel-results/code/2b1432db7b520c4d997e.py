"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    
    n = 14
    m = 35
    
    # Strategy: Use small consecutive integers to maximize sum/product overlap
    # With values 1..14, sums range 3..27 (25 possible), products 1..182
    # But we only pick 35 edges, so distinct counts will be much lower
    
    values = list(range(1, n + 1))
    
    # All possible edges (91 total)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sum and product for all possible edges
    edge_sums = {}
    edge_prods = {}
    for u, v in all_edges:
        edge_sums[(u, v)] = values[u] + values[v]
        edge_prods[(u, v)] = values[u] * values[v]
    
    # Greedy initial: pick edges that reuse existing sum/product values
    # Sort all edges by a heuristic: prefer edges with smaller values
    sorted_edges = sorted(all_edges, key=lambda e: (values[e[0]] + values[e[1]], values[e[0]] * values[e[1]]))
    
    current_edges = []
    current_edge_set = set()
    current_sum_counts = {}
    current_prod_counts = {}
    
    for e in sorted_edges:
        if len(current_edges) >= m:
            break
        s = edge_sums[e]
        p = edge_prods[e]
        current_edges.append(e)
        current_edge_set.add(e)
        current_sum_counts[s] = current_sum_counts.get(s, 0) + 1
        current_prod_counts[p] = current_prod_counts.get(p, 0) + 1
    
    def get_score(edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(edge_sums[(u, v)])
            prods.add(edge_prods[(u, v)])
        return max(len(sums), len(prods))
    
    current_score = get_score(current_edges)
    best_score = current_score
    best_edges = list(current_edges)
    
    # Local search with swap moves
    # Track which edges are in current set for O(1) membership
    in_set = set(current_edges)
    
    steps = 0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        steps += 1
        
        # Pick a random edge to remove from current set
        idx_remove = random.randint(0, m - 1)
        e_remove = current_edges[idx_remove]
        
        # Pick a random non-edge to add
        # Sample from all_edges until we find one not in current set
        while True:
            e_add = random.choice(all_edges)
            if e_add not in in_set:
                break
        
        # Perform swap
        new_edges = list(current_edges)
        new_edges[idx_remove] = e_add
        
        # Evaluate new state
        new_score = get_score(new_edges)
        
        # Accept if better or equal (plateau wandering)
        if new_score <= current_score:
            # Update state
            current_edges = new_edges
            in_set.discard(e_remove)
            in_set.add(e_add)
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_edges = list(current_edges)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Occasional restart to escape local optima
        if no_improve > 2000 and steps % 500 == 0:
            # Perturb: swap 3 random edges
            for _ in range(3):
                idx_r = random.randint(0, m - 1)
                e_r = current_edges[idx_r]
                while True:
                    e_a = random.choice(all_edges)
                    if e_a not in in_set:
                        break
                current_edges[idx_r] = e_a
                in_set.discard(e_r)
                in_set.add(e_a)
            current_score = get_score(current_edges)
            no_improve = 0
    
    # Final polish: try to improve best_edges with targeted swaps
    if time.time() < deadline - 5:
        polish_edges = list(best_edges)
        polish_set = set(polish_edges)
        polish_score = get_score(polish_edges)
        polish_no_improve = 0
        
        while time.time() < deadline - 5 and polish_no_improve < 1000:
            idx_r = random.randint(0, m - 1)
            e_r = polish_edges[idx_r]
            while True:
                e_a = random.choice(all_edges)
                if e_a not in polish_set:
                    break
            
            new_pe = list(polish_edges)
            new_pe[idx_r] = e_a
            new_ps = get_score(new_pe)
            
            if new_ps <= polish_score:
                polish_edges = new_pe
                polish_set.discard(e_r)
                polish_set.add(e_a)
                polish_score = new_ps
                polish_no_improve = 0
                if polish_score < best_score:
                    best_score = polish_score
                    best_edges = list(polish_edges)
            else:
                polish_no_improve += 1
    
    # Ensure types are correct: list of ints, list of lists of ints
    return {
        'values': [int(v) for v in values],
        'edges': [[int(u), int(v)] for u, v in best_edges]
    }
# EVOLVE-BLOCK-END
