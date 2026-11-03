import time
import random
import math
from itertools import combinations

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 115.0
    
    # 1. Greedy Baseline
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = max(max(r) for r in best_rows)
    
    # Get all differences for a row
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Check if two rows are compatible (no shared differences)
    def rows_compatible(row1, row2):
        d1 = get_row_diffs(row1)
        d2 = get_row_diffs(row2)
        return len(d1 & d2) == 0
    
    # Maximum clique search using branch and bound with coloring
    def max_clique_search(M, time_limit):
        """Find a clique of size n_rows in the compatibility graph for scope M."""
        start_time = time.time()
        
        # Generate all valid rows (Golomb rulers) of length row_len with max <= M
        # A row is valid if all internal differences are distinct
        valid_rows = []
        valid_row_diffs = []
        
        # Generate combinations of k points from 1..M
        for combo in combinations(range(1, M + 1), k):
            row = [0] + list(combo)
            diffs = get_row_diffs(row)
            if len(diffs) == len(row) * (len(row) - 1) // 2:
                valid_rows.append(row)
                valid_row_diffs.append(diffs)
        
        if len(valid_rows) < n_rows:
            return None, None
        
        # Build adjacency: edge if rows are compatible
        n = len(valid_rows)
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if not (valid_row_diffs[i] & valid_row_diffs[j]):
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Greedy coloring for upper bound
        def color_graph(vertices):
            colors = {}
            color_class = []
            for v in vertices:
                placed = False
                for ci, cc in enumerate(color_class):
                    if not any(adj[v] & set(cc) for _ in [1]):
                        cc.append(v)
                        colors[v] = ci
                        placed = True
                        break
                if not placed:
                    color_class.append([v])
                    colors[v] = len(color_class) - 1
            return colors, len(color_class)
        
        # Branch and bound for maximum clique
        best_clique = []
        
        def expand_clique(candidates, current_clique):
            nonlocal best_clique
            if time.time() - start_time > time_limit:
                return
            
            if len(current_clique) > len(best_clique):
                best_clique = current_clique[:]
            
            if len(current_clique) == n_rows:
                return
            
            if not candidates:
                return
            
            # Color the candidates for upper bound
            colors, num_colors = color_graph(candidates)
            
            # Sort by color (higher color first)
            sorted_candidates = sorted(candidates, key=lambda v: -colors[v])
            
            for idx, v in enumerate(sorted_candidates):
                if len(current_clique) + colors[v] <= len(best_clique):
                    break
                
                new_candidates = [u for u in sorted_candidates[:idx] if u in adj[v]]
                expand_clique(new_candidates, current_clique + [v])
        
        # Start with all vertices
        all_vertices = list(range(n))
        expand_clique(all_vertices, [])
        
        if len(best_clique) >= n_rows:
            clique_rows = [valid_rows[i] for i in best_clique[:n_rows]]
            clique_max = max(max(r) for r in clique_rows)
            return clique_rows, clique_max
        return None, None
    
    # Simulated Annealing for polishing
    def get_cost(rows):
        diffs = []
        max_val = 0
        for r in rows:
            max_val = max(max_val, r[-1])
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.append(r[j] - r[i])
        unique_diffs = len(set(diffs))
        collisions = len(diffs) - unique_diffs
        return collisions * 10000 + max_val, max_val, collisions
    
    def simulate_annealing(initial_rows, time_limit):
        rows = [r[:] for r in initial_rows]
        current_cost, current_max, current_collisions = get_cost(rows)
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max
        
        T = 100.0
        cooling = 0.9999
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)
            
            lower = rows[r_idx][i_idx - 1] + 1
            upper_limit = best_max + 20
            if i_idx < row_len - 1:
                upper = rows[r_idx][i_idx + 1] - 1
                upper = min(upper, upper_limit)
            else:
                upper = upper_limit
            
            if lower > upper:
                continue
                
            new_val = random.randint(lower, upper)
            if new_val == rows[r_idx][i_idx]:
                continue
                
            old_val = rows[r_idx][i_idx]
            rows[r_idx][i_idx] = new_val
            
            new_cost, new_max, new_collisions = get_cost(rows)
            
            if new_cost < current_cost:
                current_cost = new_cost
                current_max = new_max
                current_collisions = new_collisions
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_max = new_max
                    best_local_rows = [r[:] for r in rows]
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    if random.random() < 0.5:
                        current_cost = new_cost
                        current_max = new_max
                        current_collisions = new_collisions
                else:
                    if delta < 1000:
                        p = math.exp(-delta / T)
                    else:
                        p = 0.0
                    
                    if random.random() < p:
                        current_cost = new_cost
                        current_max = new_max
                        current_collisions = new_collisions
                    else:
                        rows[r_idx][i_idx] = old_val
            
            T *= cooling
            if T < 0.1:
                T = 0.1
                
        return best_local_rows, best_local_max, best_local_cost
    
    # Main Loop: Try maximum clique search for decreasing M
    # Start from best_max and try to improve
    search_start = min(best_max, 120)
    
    # Try maximum clique for various M values
    for M in range(search_start, 100, -1):
        if time.time() > deadline - 10:
            break
        
        # Limit time for each M
        time_for_m = min(10.0, deadline - time.time() - 5)
        if time_for_m <= 0:
            break
        
        clique_rows, clique_max = max_clique_search(M, time_for_m)
        
        if clique_rows is not None and clique_max < best_max:
            best_max = clique_max
            best_rows = clique_rows
            if best_max <= 112:
                break
    
    # Polish with SA
    if time.time() < deadline - 5:
        sa_time = min(5.0, deadline - time.time() - 2)
        if sa_time > 0:
            local_rows, local_max, local_cost = simulate_annealing(best_rows, sa_time)
            if local_cost < 10000 and local_max < best_max:
                best_max = local_max
                best_rows = local_rows
    
    return best_rows
