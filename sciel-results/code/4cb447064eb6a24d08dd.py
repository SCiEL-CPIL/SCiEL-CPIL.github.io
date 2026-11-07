import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds safety margin
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    
    # Candidate value sets
    value_sets = [
        list(range(1, 15)),
        list(range(2, 16)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
    ]
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    for vals in value_sets:
        if time.time() > deadline:
            break
        
        # Precompute sums and products for all pairs
        pair_sums = {}
        pair_prods = {}
        for i in range(n):
            for j in range(i + 1, n):
                pair_sums[(i, j)] = vals[i] + vals[j]
                pair_prods[(i, j)] = vals[i] * vals[j]
        
        # Local Search on Edges
        # We will run multiple restarts
        
        def evaluate_state(edge_set):
            s_counts = {}
            p_counts = {}
            for e in edge_set:
                s = pair_sums[e]
                p = pair_prods[e]
                s_counts[s] = s_counts.get(s, 0) + 1
                p_counts[p] = p_counts.get(p, 0) + 1
            return max(len(s_counts), len(p_counts))
        
        def local_search(initial_edges, max_iters=50000):
            current_edges = set(initial_edges)
            s_counts = {}
            p_counts = {}
            for e in current_edges:
                s = pair_sums[e]
                p = pair_prods[e]
                s_counts[s] = s_counts.get(s, 0) + 1
                p_counts[p] = p_counts.get(p, 0) + 1
            
            cur_score = max(len(s_counts), len(p_counts))
            best_local_score = cur_score
            best_local_edges = set(current_edges)
            
            T = 5.0
            cooling = 0.9999
            min_T = 0.1
            
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Pick a random edge to remove
                old_edge = random.choice(list(current_edges))
                # Pick a random edge to add (not in current)
                # Optimization: Pre-shuffle all_edges and pick from those not in current?
                # For n=14, 91 edges. Checking membership is fast.
                new_edge = random.choice(all_edges)
                if new_edge in current_edges:
                    continue
                
                # Calculate delta
                old_s = pair_sums[old_edge]
                old_p = pair_prods[old_edge]
                new_s = pair_sums[new_edge]
                new_p = pair_prods[new_edge]
                
                # Temporarily remove old
                s_counts[old_s] -= 1
                if s_counts[old_s] == 0:
                    del s_counts[old_s]
                p_counts[old_p] -= 1
                if p_counts[old_p] == 0:
                    del p_counts[old_p]
                
                # Temporarily add new
                s_counts[new_s] = s_counts.get(new_s, 0) + 1
                p_counts[new_p] = p_counts.get(new_p, 0) + 1
                
                new_score = max(len(s_counts), len(p_counts))
                
                # Accept or Reject
                if new_score < cur_score or random.random() < min(1.0, ((cur_score - new_score) / T) if new_score > cur_score else 1.0):
                    # Accept
                    current_edges.remove(old_edge)
                    current_edges.add(new_edge)
                    cur_score = new_score
                    
                    if cur_score < best_local_score:
                        best_local_score = cur_score
                        best_local_edges = set(current_edges)
                else:
                    # Reject: Undo changes
                    s_counts[new_s] -= 1
                    if s_counts[new_s] == 0:
                        del s_counts[new_s]
                    p_counts[new_p] -= 1
                    if p_counts[new_p] == 0:
                        del p_counts[new_p]
                    
                    s_counts[old_s] = s_counts.get(old_s, 0) + 1
                    p_counts[old_p] = p_counts.get(old_p, 0) + 1
                
                T *= cooling
                if T < min_T:
                    break
            
            return best_local_score, list(best_local_edges)
        
        # Run multiple restarts for this value set
        for restart in range(20):
            if time.time() > deadline:
                break
            
            # Initial edges: random 35
            initial_edges = random.sample(all_edges, m)
            score, edges = local_search(initial_edges)
            
            if score < best_score:
                best_score = score
                best_values = vals
                best_edges = edges
                
            # If we hit a very low score, stop early?
            # Theoretical minimum? 
            # With 35 edges, if all sums are same (impossible for distinct positive ints in a graph without loops? No, 1+4=2+3=5).
            # If we can get distinct sums < 10 and distinct prods < 10, score is < 10.
            if best_score <= 8:
                break

    # Final check and formatting
    if best_edges is None:
        # Fallback
        best_values = list(range(1, 15))
        best_edges = [list(e) for e in random.sample(all_edges, m)]
        
    return {
        'values': [int(v) for v in best_values],
        'edges': [[int(i), int(j)] for i, j in best_edges]
    }
