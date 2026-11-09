import time
import random

def has_simple_path_len_4(adj, c, u, w, v_limit):
    """
    Check if there is a simple path of length 4 from u to w in the graph of color c,
    using only vertices < v_limit.
    Path: u - a - b - d - w
    """
    mask_limit = (1 << v_limit) - 1
    
    nu = adj[c][u] & mask_limit
    # If u and w are not connected by a path of length 4, return False.
    # We need a, b, d distinct from u, w and each other.
    
    a_mask = nu
    while a_mask:
        a_bit = a_mask & -a_mask
        a = a_bit.bit_length() - 1
        a_mask ^= a_bit
        
        # a must be distinct from w
        if a == w:
            continue
            
        na = adj[c][a] & mask_limit
        # b in N(a) \ {u, w}
        b_mask = na & ~(1 << u) & ~(1 << w)
        
        while b_mask:
            b_bit = b_mask & -b_mask
            b = b_bit.bit_length() - 1
            b_mask ^= b_bit
            
            nb = adj[c][b] & mask_limit
            # d in N(b) \ {u, a, w}
            d_mask = nb & ~(1 << u) & ~(1 << a) & ~(1 << w)
            
            # We need w in N(d).
            # Check if any d in d_mask is connected to w.
            temp = d_mask
            while temp:
                d_bit = temp & -temp
                d = d_bit.bit_length() - 1
                temp ^= d_bit
                if adj[c][d] & (1 << w):
                    return True
            
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    target_n = 17
    best_n = 9
    # Seed coloring for n=9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Mode 1: Seed Extension
    # Try to extend the n=9 seed to n=17
    def try_seed_extension():
        nonlocal best_n, best_colors
        if time.time() > deadline:
            return False
            
        edge_color = [[-1] * target_n for _ in range(target_n)]
        adj = [[0] * target_n for _ in range(4)]
        
        n_seed = 9
        for i in range(n_seed):
            for j in range(i + 1, n_seed):
                c = min(j - i, n_seed - j + i) - 1
                edge_color[i][j] = c
                edge_color[j][i] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        def backtrack(v, u):
            if time.time() > deadline:
                return False
                
            if v == target_n:
                nonlocal best_n, best_colors
                cols = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        cols.append(edge_color[i][j])
                best_n = target_n
                best_colors = cols
                return True
                
            if u < 0:
                return backtrack(v + 1, v)
                
            # Try colors for (v, u)
            # Heuristic: Try colors that appear less often in (v, 0..u+1)?
            # Or just random.
            colors_to_try = [0, 1, 2, 3]
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                valid = True
                # Check against all w > u such that edge_color[v][w] == c
                for w in range(u + 1, v):
                    if edge_color[v][w] == c:
                        if has_simple_path_len_4(adj, c, u, w, v):
                            valid = False
                            break
                            
                if valid:
                    edge_color[v][u] = c
                    edge_color[u][v] = c
                    adj[c][v] |= (1 << u)
                    adj[c][u] |= (1 << v)
                    
                    if backtrack(v, u - 1):
                        return True
                        
                    edge_color[v][u] = -1
                    edge_color[u][v] = -1
                    adj[c][v] ^= (1 << u)
                    adj[c][u] ^= (1 << v)
                    
            return False
            
        return backtrack(n_seed, n_seed - 1)
        
    if try_seed_extension():
        return {'n': best_n, 'colors': best_colors}
        
    # Mode 2: Random Search from scratch
    # With symmetry breaking
    def try_random_search():
        nonlocal best_n, best_colors
        
        for restart in range(1000):
            if time.time() > deadline:
                break
                
            random.seed(restart) # Deterministic restarts for reproducibility? 
            # Actually, using time-based seed or just random is fine.
            # Let's use random.seed(restart) to ensure we cover different paths.
            
            edge_color = [[-1] * target_n for _ in range(target_n)]
            adj = [[0] * target_n for _ in range(4)]
            
            # Symmetry breaking: Fix c(0,1) = 0
            # We start with vertices 0 and 1.
            # Actually, we can start backtracking from v=1.
            # But we need to fix c(0,1).
            
            # Initialize with vertex 0.
            # Add vertex 1.
            edge_color[0][1] = 0
            edge_color[1][0] = 0
            adj[0][0] |= (1 << 1)
            adj[0][1] |= (1 << 0)
            
            def backtrack(v, u):
                if time.time() > deadline:
                    return False
                    
                if v == target_n:
                    nonlocal best_n, best_colors
                    cols = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            cols.append(edge_color[i][j])
                    best_n = target_n
                    best_colors = cols
                    return True
                    
                if u < 0:
                    return backtrack(v + 1, v)
                    
                colors_to_try = [0, 1, 2, 3]
                random.shuffle(colors_to_try)
                
                for c in colors_to_try:
                    valid = True
                    for w in range(u + 1, v):
                        if edge_color[v][w] == c:
                            if has_simple_path_len_4(adj, c, u, w, v):
                                valid = False
                                break
                                
                    if valid:
                        edge_color[v][u] = c
                        edge_color[u][v] = c
                        adj[c][v] |= (1 << u)
                        adj[c][u] |= (1 << v)
                        
                        if backtrack(v, u - 1):
                            return True
                            
                        edge_color[v][u] = -1
                        edge_color[u][v] = -1
                        adj[c][v] ^= (1 << u)
                        adj[c][u] ^= (1 << v)
                        
                return False
                
            # Start backtracking from v=2, u=1
            # v=1 is already connected to 0.
            # So next vertex is 2.
            if backtrack(2, 1):
                return True
                
        return False
        
    try_random_search()
    
    return {'n': best_n, 'colors': best_colors}
