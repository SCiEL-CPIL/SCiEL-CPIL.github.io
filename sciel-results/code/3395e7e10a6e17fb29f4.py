import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Try to find a valid circulant graph for n from 40 down to 27
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
        
        max_s = n // 2
        candidates = []
        
        # Backtracking search for triangle-free S
        # S is a subset of {1, ..., max_s}
        # We build S incrementally.
        # To ensure we find dense graphs (small alpha), we can bias towards adding elements.
        # But we need to check alpha at the end.
        
        # Let's generate all triangle-free S and check alpha.
        # To speed up, we can sort candidates by size descending.
        
        def backtrack(current_S, next_val):
            if time.time() > deadline:
                return
            
            # Check if current_S is a valid candidate for alpha check
            # Only check if size is sufficient
            if len(current_S) >= n // 20:
                if check_alpha_less_than_10_circulant(n, current_S):
                    # Found a valid graph
                    adj, edges = build_circulant_graph(n, current_S)
                    # Store result and stop
                    nonlocal result
                    result = {'n': n, 'edges': edges}
                    return True
            
            for v in range(next_val, max_s + 1):
                # Check if adding v keeps S triangle-free
                if is_circulant_triangle_free_with(n, current_S, v):
                    if backtrack(current_S + [v], v + 1):
                        return True
            
            return False
        
        result = None
        # Start with empty set?
        # We want large sets, so maybe start by picking large sets?
        # The backtracking above explores all.
        # To find a large set quickly, we can try to fill S as much as possible.
        # This is a Maximum Independent Set on the "triangle conflict graph".
        # But let's just do a simple DFS.
        
        # Optimization: Try to find a solution with large |S| first.
        # We can iterate sizes from max_s down to min_size.
        # For each size, find a valid S.
        
        min_size = max(1, n // 20)
        
        for size in range(max_s, min_size - 1, -1):
            if time.time() > deadline:
                break
            
            # Find one triangle-free S of this size
            # We can use a simple random search or backtracking with size constraint.
            # Given the small max_s (20), backtracking is fine.
            
            def find_s_of_size(target_size, current_S, next_val):
                if time.time() > deadline:
                    return None
                
                if len(current_S) == target_size:
                    if check_alpha_less_than_10_circulant(n, current_S):
                        return current_S
                    return None
                
                # Pruning: if remaining slots > available numbers
                remaining = target_size - len(current_S)
                available = max_s - next_val + 1
                if remaining > available:
                    return None
                
                for v in range(next_val, max_s + 1):
                    if is_circulant_triangle_free_with(n, current_S, v):
                        res = find_s_of_size(target_size, current_S + [v], v + 1)
                        if res:
                            return res
                return None
            
            S = find_s_of_size(size, [], 1)
            if S:
                adj, edges = build_circulant_graph(n, S)
                return {'n': n, 'edges': edges}
        
    # Fallback to the seed graph (n=26)
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

def is_circulant_triangle_free_with(n, S, new_v):
    # Check if S U {new_v} is triangle-free
    # Triangle if exists a, b in S U {new_v} such that -(a+b) in S U {new_v}
    # We only need to check pairs involving new_v, as S is already triangle-free.
    # Pairs: (new_v, new_v) -> 2*new_v
    # Pairs: (new_v, a) for a in S -> new_v + a
    
    S_set = set(S)
    
    # Case 1: a = new_v, b = new_v
    c = -(2 * new_v) % n
    if c in S_set or c == new_v:
        # If c == new_v, then 3*new_v = 0 mod n.
        # This forms a triangle (0, new_v, 2*new_v).
        # If c in S_set, triangle (0, new_v, new_v+c).
        return False
        
    # Case 2: a = new_v, b in S
    for a in S:
        c = -(new_v + a) % n
        if c in S_set or c == new_v:
            return False
            
    return True

def check_alpha_less_than_10_circulant(n, S):
    # Build adj bitsets for circulant graph
    neighbors = [0] * n
    for i in range(n):
        mask = 0
        for s in S:
            j1 = (i + s) % n
            j2 = (i - s) % n
            mask |= (1 << j1)
            mask |= (1 << j2)
        neighbors[i] = mask
        
    # Use the same DFS as before
    found_large = [False]
    
    def dfs(vertices_mask, current_size):
        if found_large[0]:
            return
        if current_size >= 10:
            found_large[0] = True
            return
        
        if vertices_mask == 0:
            return
        
        v_count = bin(vertices_mask).count('1')
        if current_size + v_count < 10:
            return
            
        lsb = vertices_mask & -vertices_mask
        v = lsb.bit_length() - 1
        
        new_mask = vertices_mask & ~(neighbors[v] | lsb)
        dfs(new_mask, current_size + 1)
        
        if found_large[0]:
            return
            
        new_mask = vertices_mask & ~lsb
        dfs(new_mask, current_size)
        
    dfs((1 << n) - 1, 0)
    
    return not found_large[0]

def build_circulant_graph(n, S):
    adj = np.zeros((n, n), dtype=bool)
    edges = []
    for s in S:
        for i in range(n):
            j = (i + s) % n
            if i < j:
                adj[i, j] = True
                adj[j, i] = True
                edges.append([i, j])
    return adj, edges
