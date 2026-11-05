import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an edge colouring for K16 with minimal colours such that
    every K4 subgraph spans at least 5 distinct colours.
    Uses greedy construction + simulated annealing with delta-cost updates.
    """
    n = 16
    total_edges = n * (n - 1) // 2  # 120
    
    # Precompute edge index mapping
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            idx += 1
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return edge_map[(i, j)]
    
    # Precompute all K4s and their edges
    k4_list = []
    for a, b, c, d in combinations(range(n), 4):
        edges = [
            edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
            edge_idx(b, c), edge_idx(b, d), edge_idx(c, d)
        ]
        k4_list.append(edges)
    
    num_k4s = len(k4_list)  # 1820
    
    # Precompute for each edge, which K4s contain it
    edge_to_k4s = [[] for _ in range(total_edges)]
    for k4_idx, edges in enumerate(k4_list):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
    
    # Penalty for a K4: (5 - distinct_count)^2 if distinct_count < 5, else 0
    def k4_penalty(colors, k4_idx):
        edges = k4_list[k4_idx]
        c = [colors[e] for e in edges]
        d = len(set(c))
        return (5 - d) ** 2 if d < 5 else 0
    
    def total_cost(colors):
        total = 0
        for k4_idx in range(num_k4s):
            p = k4_penalty(colors, k4_idx)
            total += p
        return total
    
    def is_valid(colors):
        for k4_idx in range(num_k4s):
            if k4_penalty(colors, k4_idx) > 0:
                return False
        return True
    
    # Fallback: 17-colour cyclic construction
    def fallback():
        return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    # Simulated annealing with delta-cost
    def simulated_annealing(k, time_limit, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        start_time = time.time()
        
        # Initialize with greedy construction
        colors = np.zeros(total_edges, dtype=np.int32)
        # Greedy: assign colors to minimize local violations
        edge_order = list(range(total_edges))
        random.shuffle(edge_order)
        
        for e in edge_order:
            best_c = 0
            best_p = float('inf')
            for c in range(k):
                colors[e] = c
                p = 0
                for k4_idx in edge_to_k4s[e]:
                    p += k4_penalty(colors, k4_idx)
                if p < best_p:
                    best_p = p
                    best_c = c
            colors[e] = best_c
        
        cur_cost = total_cost(colors)
        if cur_cost == 0:
            return colors.tolist()
        
        best_colors = colors.copy()
        best_cost = cur_cost
        
        T = 100.0
        T_min = 0.1
        cooling = 0.9995
        
        while time.time() - start_time < time_limit:
            if T < T_min:
                T = T_min
            
            # Pick a random edge
            e_idx = random.randint(0, total_edges - 1)
            old_c = colors[e_idx]
            
            # Try a random new color
            new_c = random.randint(0, k - 1)
            if new_c == old_c:
                new_c = (new_c + 1) % k
            
            # Calculate delta cost using only affected K4s
            delta = 0
            for k4_idx in edge_to_k4s[e_idx]:
                old_pen = k4_penalty(colors, k4_idx)
                colors[e_idx] = new_c
                new_pen = k4_penalty(colors, k4_idx)
                colors[e_idx] = old_c
                delta += new_pen - old_pen
            
            # Accept if improvement, or with probability exp(-delta/T)
            if delta <= 0 or random.random() < np.exp(-delta / T):
                colors[e_idx] = new_c
                cur_cost += delta
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        return best_colors.tolist()
            
            T *= cooling
        
        # Return best found
        if best_cost < cur_cost:
            return best_colors.tolist()
        return colors.tolist()
    
    # Main search: try k=13, 14, 15 with multiple restarts
    time_start = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Try each k with decreasing time budget
    for k in [13, 14, 15]:
        remaining = deadline - (time.time() - time_start)
        if remaining < 5:
            break
        
        # Allocate time for this k
        if k == 13:
            alloc = min(30.0, remaining * 0.4)
        elif k == 14:
            alloc = min(40.0, remaining * 0.4)
        else:
            alloc = min(50.0, remaining * 0.5)
        
        # Multiple restarts
        num_restarts = max(1, int(alloc / 5.0))
        per_restart = alloc / num_restarts
        
        for restart in range(num_restarts):
            if time.time() - time_start > deadline - 5:
                break
            
            res = simulated_annealing(k, per_restart, seed=42 + k * 100 + restart)
            if res is not None and len(res) == total_edges:
                # Verify
                if is_valid(res):
                    return res
    
    # If we get here, try one more aggressive attempt with k=15
    if time.time() - time_start < deadline - 10:
        res = simulated_annealing(15, deadline - time.time() - 10, seed=999)
        if res is not None and len(res) == total_edges and is_valid(res):
            return res
    
    # Fallback
    return fallback()
# EVOLVE-BLOCK-END
