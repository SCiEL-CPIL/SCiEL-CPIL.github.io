import time
import random
import numpy as np

def _count_k222_violations_numpy(adj, n):
    """
    Count the number of monochromatic K2,2,2 subgraphs in the graph defined by adj.
    adj: np.array of shape (n, n), boolean.
    Returns count of K2,2,2.
    """
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                continue
            C = adj[u] & adj[v]
            c_idx = np.where(C)[0]
            m = len(c_idx)
            if m < 4:
                continue
            
            # We need to find two disjoint non-edges (w,x) and (y,z) in C
            # such that all cross edges exist.
            
            # Iterate over pairs (w, x) in C that are non-edges
            # To speed up, we can iterate over w, and check x > w.
            for i in range(m):
                w = c_idx[i]
                # x must be in c_idx[i+1:]
                for j in range(i + 1, m):
                    x = c_idx[j]
                    if adj[w, x]:
                        continue
                    
                    # Candidates for y, z:
                    # Must be in C
                    # Must be neighbors of w and x
                    # Must not be w or x
                    cand_mask = C.copy()
                    cand_mask[w] = False
                    cand_mask[x] = False
                    cand_mask &= adj[w]
                    cand_mask &= adj[x]
                    
                    cand_idx = np.where(cand_mask)[0]
                    k = len(cand_idx)
                    if k < 2:
                        continue
                        
                    # Find non-adjacent pair (y, z) in cand_idx
                    for a in range(k):
                        y = cand_idx[a]
                        # z must be in cand_idx[a+1:]
                        for b in range(a + 1, k):
                            z = cand_idx[b]
                            if not adj[y, z]:
                                count += 1
    return count

def _is_valid_numpy(adj, n):
    return _count_k222_violations_numpy(adj, n) == 0

def _local_search_numpy(n, time_limit, max_iters=200000):
    """
    Local search for a valid coloring of K_n.
    """
    start_time = time.time()
    num_edges = n * (n - 1) // 2
    edge_map = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initialize random
    colors = np.random.randint(0, 2, size=num_edges)
    
    # Build initial adj
    adj = np.zeros((n, n), dtype=bool)
    for idx, (u, v) in enumerate(edge_map):
        if colors[idx] == 1:
            adj[u, v] = True
            adj[v, u] = True
            
    # Compute initial score
    v1 = _count_k222_violations_numpy(adj, n)
    adj0 = ~adj
    np.fill_diagonal(adj0, False)
    v0 = _count_k222_violations_numpy(adj0, n)
    total_violations = v1 + v0
    
    if total_violations == 0:
        return colors.tolist()
    
    best_colors = colors.copy()
    best_score = total_violations
    
    T = 1.0
    T_min = 0.01
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        idx = np.random.randint(0, num_edges)
        u, v = edge_map[idx]
        
        # Flip
        colors[idx] = 1 - colors[idx]
        adj[u, v] = not adj[u, v]
        adj[v, u] = not adj[v, u]
        
        # Compute new score
        # This is the expensive part
        v1_new = _count_k222_violations_numpy(adj, n)
        adj0 = ~adj
        np.fill_diagonal(adj0, False)
        v0_new = _count_k222_violations_numpy(adj0, n)
        total_violations_new = v1_new + v0_new
        
        delta = total_violations_new - total_violations
        
        if delta < 0 or (random.random() < np.exp(-delta / T)):
            total_violations = total_violations_new
            if total_violations < best_score:
                best_score = total_violations
                best_colors = colors.copy()
                if best_score == 0:
                    return best_colors.tolist()
        else:
            # Revert
            colors[idx] = 1 - colors[idx]
            adj[u, v] = not adj[u, v]
            adj[v, u] = not adj[v, u]
            
        T = max(T * 0.999, T_min)
        
    return best_colors.tolist()

def _paley(n, p):
    """Paley coloring using prime p >= n."""
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def construct():
    deadline = time.time() + 140
    
    # Try n=30
    # Use multiple restarts
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        t_limit = min(10, time_left - 2)
        if t_limit <= 0:
            break
            
        colors = _local_search_numpy(30, t_limit)
        
        if colors:
            adj = np.zeros((30, 30), dtype=bool)
            edge_map = [(i, j) for i in range(30) for j in range(i + 1, 30)]
            for idx, (u, v) in enumerate(edge_map):
                if colors[idx] == 1:
                    adj[u, v] = True
                    adj[v, u] = True
            if _is_valid_numpy(adj, 30):
                return {'n': 30, 'colors': colors}
                
    # Fallback to n=29
    colors_29 = _paley(29, 29)
    if colors_29:
        adj = np.zeros((29, 29), dtype=bool)
        edge_map = [(i, j) for i in range(29) for j in range(i + 1, 29)]
        for idx, (u, v) in enumerate(edge_map):
            if colors_29[idx] == 1:
                adj[u, v] = True
                adj[v, u] = True
        if _is_valid_numpy(adj, 29):
            return {'n': 29, 'colors': colors_29}
            
    # Try local search for 29 if Paley failed
    if time.time() < deadline - 10:
        colors_29 = _local_search_numpy(29, deadline - time.time() - 5)
        if colors_29:
            adj = np.zeros((29, 29), dtype=bool)
            edge_map = [(i, j) for i in range(29) for j in range(i + 1, 29)]
            for idx, (u, v) in enumerate(edge_map):
                if colors_29[idx] == 1:
                    adj[u, v] = True
                    adj[v, u] = True
            if _is_valid_numpy(adj, 29):
                return {'n': 29, 'colors': colors_29}
                
    return {'n': 1, 'colors': []}
