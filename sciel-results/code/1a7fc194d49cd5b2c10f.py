import numpy as np
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    n = 16
    num_edges = 120
    num_k4 = 1820

    # Precompute edge indices
    edge_idx = {}
    idx = 0
    for v1 in range(n):
        for v2 in range(v1 + 1, n):
            edge_idx[(v1, v2)] = idx
            idx += 1

    # Generate all K4s and their edge indices
    vertices = list(range(n))
    k4_list = list(itertools.combinations(vertices, 4))

    k4_edges = []
    for k4 in k4_list:
        v1, v2, v3, v4 = k4
        e1 = edge_idx[(v1, v2)]
        e2 = edge_idx[(v1, v3)]
        e3 = edge_idx[(v1, v4)]
        e4 = edge_idx[(v2, v3)]
        e5 = edge_idx[(v2, v4)]
        e6 = edge_idx[(v3, v4)]
        k4_edges.append([e1, e2, e3, e4, e5, e6])

    k4_edges = np.array(k4_edges, dtype=np.int32)  # shape (1820, 6)

    # Precompute affected K4s for each edge
    edge_to_k4s = [[] for _ in range(num_edges)]
    for i in range(num_k4):
        for e in k4_edges[i]:
            edge_to_k4s[e].append(i)
    # Convert to numpy arrays for faster access
    edge_to_k4s_arr = [np.array(lst, dtype=np.int32) for lst in edge_to_k4s]

    def count_bad(coloring):
        """Count number of K4s with fewer than 5 distinct colors."""
        bad_count = 0
        for i in range(num_k4):
            c = coloring[k4_edges[i]]
            if len(set(c)) < 5:
                bad_count += 1
        return bad_count

    def run_sa(k, time_limit, initial_coloring=None, restarts=1):
        """Simulated annealing with delta updates."""
        best_coloring = None
        best_bad = float('inf')
        
        for restart in range(restarts):
            if time.time() > start_time + time_limit:
                break
                
            if initial_coloring is not None and restart == 0:
                coloring = initial_coloring.copy()
            else:
                coloring = np.random.randint(0, k, size=num_edges)
            
            # Initial bad mask
            bad_mask = np.zeros(num_k4, dtype=bool)
            for i in range(num_k4):
                if len(set(coloring[k4_edges[i]])) < 5:
                    bad_mask[i] = True
            current_bad = int(np.sum(bad_mask))
            
            if current_bad < best_bad:
                best_bad = current_bad
                best_coloring = coloring.copy()
                if best_bad == 0:
                    return best_coloring, 0
            
            temp = 1.0
            cooling = 0.9995
            end_time = time.time() + (time_limit / restarts) if restarts > 1 else time_limit
            
            while time.time() < end_time:
                # Pick random edge
                e_idx = np.random.randint(num_edges)
                old_color = coloring[e_idx]
                
                # Pick new color
                new_color = np.random.randint(k)
                if new_color == old_color:
                    continue
                
                # Identify affected K4s
                affected = edge_to_k4s_arr[e_idx]
                if len(affected) == 0:
                    continue
                
                # Calculate delta by checking affected K4s
                old_bad_in_affected = np.sum(bad_mask[affected])
                
                # Apply new color temporarily
                coloring[e_idx] = new_color
                
                new_bad_in_affected = 0
                for i in affected:
                    if len(set(coloring[k4_edges[i]])) < 5:
                        new_bad_in_affected += 1
                        bad_mask[i] = True
                    else:
                        bad_mask[i] = False
                
                delta = new_bad_in_affected - old_bad_in_affected
                
                # Accept or reject
                if delta < 0 or (temp > 0 and random.random() < np.exp(-delta / temp)):
                    current_bad += delta
                    if current_bad < best_bad:
                        best_bad = current_bad
                        best_coloring = coloring.copy()
                        if best_bad == 0:
                            return best_coloring, 0
                else:
                    # Reject: revert coloring and bad_mask
                    coloring[e_idx] = old_color
                    for i in affected:
                        if len(set(coloring[k4_edges[i]])) < 5:
                            bad_mask[i] = True
                        else:
                            bad_mask[i] = False
                
                temp *= cooling
                if temp < 0.001:
                    temp = 1.0  # Restart
        
        return best_coloring, best_bad

    def run_local_search(coloring, k, time_limit):
        """Greedy local search to polish a coloring."""
        current_bad = count_bad(coloring)
        if current_bad == 0:
            return coloring, 0
        
        end_time = time.time() + time_limit
        improved = True
        while improved and time.time() < end_time:
            improved = False
            for e_idx in range(num_edges):
                if time.time() > end_time:
                    break
                old_color = coloring[e_idx]
                best_new_color = old_color
                best_delta = 0
                
                affected = edge_to_k4s_arr[e_idx]
                old_bad = sum(1 for i in affected if len(set(coloring[k4_edges[i]])) < 5)
                
                for new_color in range(k):
                    if new_color == old_color:
                        continue
                    coloring[e_idx] = new_color
                    new_bad = sum(1 for i in affected if len(set(coloring[k4_edges[i]])) < 5)
                    delta = new_bad - old_bad
                    if delta < best_delta:
                        best_delta = delta
                        best_new_color = new_color
                
                coloring[e_idx] = old_color
                if best_delta < 0:
                    coloring[e_idx] = best_new_color
                    current_bad += best_delta
                    improved = True
                    if current_bad == 0:
                        return coloring, 0
        
        return coloring, current_bad

    # Try to find a solution with 13 colors first
    if time.time() < start_time + 50:
        sol, bad = run_sa(13, 45, restarts=3)
        if bad == 0:
            return sol.tolist()
        # Polish
        sol, bad = run_local_search(sol, 13, 5)
        if bad == 0:
            return sol.tolist()

    # Try 14 colors
    if time.time() < start_time + 90:
        sol, bad = run_sa(14, 35, restarts=2)
        if bad == 0:
            return sol.tolist()
        # Polish
        sol, bad = run_local_search(sol, 14, 5)
        if bad == 0:
            return sol.tolist()

    # Try 15 colors
    if time.time() < start_time + 120:
        sol, bad = run_sa(15, 25, restarts=2)
        if bad == 0:
            return sol.tolist()
        # Polish
        sol, bad = run_local_search(sol, 15, 5)
        if bad == 0:
            return sol.tolist()

    # Fallback: use a known valid coloring with more colors
    # Use (i + j) % 17 which uses 17 colors
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
