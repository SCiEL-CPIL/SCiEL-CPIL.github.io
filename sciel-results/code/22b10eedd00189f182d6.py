import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Candidate value sets designed to maximize collisions in sums and products
    value_sets = [
        list(range(1, 15)),
        list(range(2, 29, 2)),
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    ]
    
    # Remove duplicates to save time
    value_sets = list(set(tuple(v) for v in value_sets))
    value_sets = [list(v) for v in value_sets]
    
    best_score = float('inf')
    best_result = None
    
    for values in value_sets:
        if time.time() > deadline - 10:
            break
        
        n = len(values)
        all_pairs = list(combinations(range(n), 2))
        pair_sums = {pair: values[pair[0]] + values[pair[1]] for pair in all_pairs}
        pair_prods = {pair: values[pair[0]] * values[pair[1]] for pair in all_pairs}
        
        # Greedy Initialization
        edges = []
        selected_sums = set()
        selected_prods = set()
        available_pairs = set(all_pairs)
        
        for _ in range(35):
            best_pair = None
            best_cost = float('inf')
            current_max = max(len(selected_sums), len(selected_prods))
            
            for p in available_pairs:
                s = pair_sums[p]
                pr = pair_prods[p]
                
                # Check if adding this pair increases the distinct counts
                add_s = s not in selected_sums
                add_p = pr not in selected_prods
                
                new_len_s = len(selected_sums) + (1 if add_s else 0)
                new_len_p = len(selected_prods) + (1 if add_p else 0)
                cost = max(new_len_s, new_len_p)
                
                if cost < best_cost:
                    best_cost = cost
                    best_pair = p
                    if cost == current_max:
                        # If we can maintain the current max, prefer collisions
                        # This is a heuristic to break ties towards collisions
                        if not add_s and not add_p:
                            break
                            
            if best_pair is None:
                break
                
            edges.append(list(best_pair))
            selected_sums.add(pair_sums[best_pair])
            selected_prods.add(pair_prods[best_pair])
            available_pairs.remove(best_pair)
            
        # Simulated Annealing Refinement
        current_edges = set(tuple(e) for e in edges)
        current_sums = {pair_sums[e] for e in current_edges}
        current_prods = {pair_prods[e] for e in current_edges}
        current_score = max(len(current_sums), len(current_prods))
        
        best_edges_sa = set(current_edges)
        best_score_sa = current_score
        best_sums_sa = set(current_sums)
        best_prods_sa = set(current_prods)
        
        T = 1.0
        cooling = 0.995
        min_T = 0.01
        
        while time.time() < deadline - 5 and T > min_T:
            e_in = random.choice(list(current_edges))
            e_out = random.choice(list(available_pairs - current_edges))
            
            new_edges = set(current_edges)
            new_edges.remove(e_in)
            new_edges.add(e_out)
            
            new_sums = {pair_sums[e] for e in new_edges}
            new_prods = {pair_prods[e] for e in new_edges}
            new_score = max(len(new_sums), len(new_prods))
            
            delta = new_score - current_score
            if delta <= 0 or random.random() < pow(2.718, -delta / T):
                current_edges = new_edges
                current_sums = new_sums
                current_prods = new_prods
                current_score = new_score
                
                if current_score < best_score_sa:
                    best_score_sa = current_score
                    best_edges_sa = set(current_edges)
                    best_sums_sa = set(current_sums)
                    best_prods_sa = set(current_prods)
                    
            T *= cooling
        
        if best_score_sa < best_score:
            best_score = best_score_sa
            best_result = {
                'values': values,
                'edges': [list(e) for e in best_edges_sa]
            }
            
    if best_result is None:
        best_result = {
            'values': list(range(1, 15)),
            'edges': [list(e) for e in list(combinations(range(14), 2))[:35]]
        }
        
    return best_result
