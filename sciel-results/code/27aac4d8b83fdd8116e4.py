import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Fallback
    best_result = {
        'n': 9,
        'edges': [list(e) for e in combinations(range(9), 5)]
    }
    best_edges_count = len(best_result['edges'])
    
    def is_2_colorable(n, edges):
        def backtrack(colors_tuple):
            colors = list(colors_tuple)
            
            changed = True
            while changed:
                changed = False
                for e in edges:
                    c0, c1, c2, c3, c4 = colors[e[0]], colors[e[1]], colors[e[2]], colors[e[3]], colors[e[4]]
                    vals = [c0, c1, c2, c3, c4]
                    assigned_vals = [v for v in vals if v != -1]
                    unassigned_indices = [i for i, v in enumerate(vals) if v == -1]
                    
                    if len(assigned_vals) == 5:
                        if len(set(assigned_vals)) == 1:
                            return None
                    elif len(assigned_vals) == 4:
                        if len(set(assigned_vals)) == 1:
                            forced_color = 1 - assigned_vals[0]
                            u_idx = unassigned_indices[0]
                            v = e[u_idx]
                            if colors[v] == -1:
                                colors[v] = forced_color
                                changed = True
                            elif colors[v] != forced_color:
                                return None
            
            if all(c != -1 for c in colors):
                return True
            
            var = -1
            for i in range(len(colors)):
                if colors[i] == -1:
                    var = i
                    break
            
            if var == -1:
                return True

            colors[var] = 0
            res = backtrack(tuple(colors))
            if res is not None:
                return res
            
            colors[var] = 1
            res = backtrack(tuple(colors))
            if res is not None:
                return res
            
            return None

        initial_colors = tuple([-1] * n)
        return backtrack(initial_colors) is not None

    def try_random_construction(n, max_edges_limit, seed):
        random.seed(seed)
        edges = []
        edge_set = set()
        
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        for e in all_edges:
            if len(edges) >= max_edges_limit:
                break
            if e in edge_set:
                continue
            edge_set.add(e)
            edges.append(e)
            
            if not is_2_colorable(n, edges):
                return edges
        
        return None

    trials = 0
    # We want to find something with < 126 edges.
    # Likely < 60.
    
    # Strategy:
    # Try n=10, 11, 12...
    # For each n, run multiple random trials.
    # Stop if we find a solution with edges < best_edges_count.
    
    while time.time() < deadline:
        trials += 1
        n = 10 + (trials % 6) # 10 to 15
        
        # Limit the number of edges we are willing to accept in this trial
        # If we exceed best_edges_count, we discard this path.
        # But we don't know when it becomes non-2-colorable.
        # So we just run the trial, and if it returns a result, check its size.
        
        res = try_random_construction(n, best_edges_count, 42 + trials)
        
        if res is not None:
            if len(res) < best_edges_count:
                best_edges_count = len(res)
                best_result = {
                    'n': n,
                    'edges': [list(e) for e in res]
                }
                # If we get very low, we can tighten the limit for future trials
                # to avoid wasting time on worse solutions, though the loop condition handles that.

    return best_result
