import numpy as np
import itertools
import time
import random

def get_k6_cycles():
    cycles = set()
    # Permutations of 1,2,3,4,5
    for perm in itertools.permutations([1, 2, 3, 4, 5]):
        v = (0,) + perm
        # Edges: (v[i], v[i+1]) for i=0..4 and (v[5], v[0])
        edges = set()
        for i in range(6):
            u, w = v[i], v[(i + 1) % 6]
            if u > w: u, w = w, u
            edges.add((u, w))
        cycles.add(frozenset(edges))
    return [list(c) for c in cycles]

K6_CYCLES = get_k6_cycles() # Should be 60

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin for 120s limit
    
    # Helper to check validity
    def is_valid(n, colors):
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return False
        
        def get_idx(i, j):
            return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
        # Check all C6s
        # For efficiency, we can just check a subset or use the full list if n is small
        # For n=17, full check is ~740k cycles.
        # Let's just do it.
        
        c6_edges_list = []
        # Iterate over all combinations of 6 vertices
        for comb in itertools.combinations(range(n), 6):
            for cycle_edges in K6_CYCLES:
                global_edges = []
                for u, v in cycle_edges:
                    gu = comb[u]
                    gv = comb[v]
                    if gu > gv: gu, gv = gv, gu
                    global_edges.append(get_idx(gu, gv))
                c6_edges_list.append(global_edges)
        
        c6_edges = np.array(c6_edges_list, dtype=np.int32)
        c = np.array(colors, dtype=np.int8)
        c6_c = c[c6_edges]
        is_mono = np.all(c6_c == c6_c[:, 0:1], axis=1)
        return np.sum(is_mono) == 0

    def solve_n(n, time_limit):
        num_edges = n * (n - 1) // 2
        
        def get_idx(i, j):
            return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
        c6_edges_list = []
        for comb in itertools.combinations(range(n), 6):
            for cycle_edges in K6_CYCLES:
                global_edges = []
                for u, v in cycle_edges:
                    gu = comb[u]
                    gv = comb[v]
                    if gu > gv: gu, gv = gv, gu
                    global_edges.append(get_idx(gu, gv))
                c6_edges_list.append(global_edges)
        
        c6_edges = np.array(c6_edges_list, dtype=np.int32)
        
        # Simulated Annealing
        colors = np.random.randint(0, 4, num_edges, dtype=np.int8)
        
        def evaluate(c):
            c6_c = c[c6_edges]
            is_mono = np.all(c6_c == c6_c[:, 0:1], axis=1)
            return np.sum(is_mono)
        
        current_score = evaluate(colors)
        if current_score == 0:
            return colors.tolist()
            
        best_score = current_score
        best_colors = colors.copy()
        
        T = 10.0
        cooling = 0.9995
        min_T = 0.01
        
        while time.time() < start_time + time_limit and T > min_T:
            idx = np.random.randint(num_edges)
            old_color = colors[idx]
            new_color = np.random.randint(0, 3)
            if new_color >= old_color:
                new_color += 1
            
            colors[idx] = new_color
            new_score = evaluate(colors)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_colors = colors.copy()
                    if best_score == 0:
                        return best_colors.tolist()
            else:
                colors[idx] = old_color
                
            T *= cooling
            if T < min_T:
                break
                
        if best_score == 0:
            return best_colors.tolist()
        return None

    # Try n=17
    # Time budget: 100s
    result_17 = solve_n(17, 100.0)
    if result_17 is not None:
        return {'n': 17, 'colors': result_17}
        
    # Try n=16
    # Time budget: 10s
    if time.time() < deadline:
        result_16 = solve_n(16, 10.0)
        if result_16 is not None:
            return {'n': 16, 'colors': result_16}
            
    # Try n=15
    # Time budget: 5s
    if time.time() < deadline:
        result_15 = solve_n(15, 5.0)
        if result_15 is not None:
            return {'n': 15, 'colors': result_15}
            
    # Fallback to n=9 (known valid from seed)
    n_seed = 9
    return {'n': n_seed, 'colors': [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]}
