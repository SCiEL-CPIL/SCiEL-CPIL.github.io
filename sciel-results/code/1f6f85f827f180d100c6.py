import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # 1. Generate Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    
    # Verify Paley 37
    if not is_valid_graph(q, edges, 5, 5):
        # Should never happen, but fallback
        return q, edges

    best_n = q
    best_edges = edges
    
    # 2. Try to extend to 38 vertices
    # We will try to add vertex 37
    # We need to choose neighbors for vertex 37 among 0..36
    
    # Heuristic: Try random assignments with validation
    # To make it fast, we only check for K5 and Ind5 involving the new vertex
    # or we run full check? Full check is expensive.
    # But n=38 is small.
    
    # Let's try a few random configurations for the new vertex
    # If we find one, we return it.
    
    attempts = 0
    while time.time() < deadline and attempts < 1000:
        attempts += 1
        # Randomly decide edges for new vertex 37
        new_edges = list(edges)
        for i in range(q):
            if random.random() < 0.5:
                new_edges.append((i, 37))
        
        if is_valid_graph(q + 1, new_edges, 5, 5):
            best_n = q + 1
            best_edges = new_edges
            # Try to extend further?
            break # Or continue to try for 39?
            # Let's stick to 38 for now to be safe and fast.
            
    return best_n, best_edges

def is_valid_graph(n, edges, clique_size, ind_size):
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Check for clique of size clique_size
    # Use backtracking or simple check for small n
    # For n=38, checking all subsets of size 5 is C(38,5) ~ 5M.
    # 5M * 10 (edges in K5) = 50M ops. Might be slow in Python.
    # Better: Check each 5-tuple?
    # Or use bitsets?
    
    # Let's use a simple recursive check for K5 and Ind5.
    # Since n is small, we can use a simple DFS.
    
    def has_clique(v, size, current_clique):
        if len(current_clique) == size:
            return True
        # Optimization: if not enough vertices left, return False
        if len(current_clique) + (n - v) < size:
            return False
        
        for i in range(v, n):
            # Check if i connects to all in current_clique
            if all(adj[i][j] for j in current_clique):
                if has_clique(i + 1, size, current_clique + [i]):
                    return True
        return False
        
    def has_independent(v, size, current_ind):
        if len(current_ind) == size:
            return True
        if len(current_ind) + (n - v) < size:
            return False
        
        for i in range(v, n):
            # Check if i does NOT connect to any in current_ind
            if all(not adj[i][j] for j in current_ind):
                if has_independent(i + 1, size, current_ind + [i]):
                    return True
        return False

    # The above DFS might be too slow if it explores many branches.
    # For n=37, it's known to be free.
    # For n=38, if we add a vertex, the structure is slightly perturbed.
    
    # A faster check:
    # Iterate all combinations of 5 vertices?
    # C(38, 5) is 501,942.
    # For each, check 10 edges. 5M checks.
    # In Python, 5M iterations might take ~0.5s to 1s.
    # This is acceptable.
    
    from itertools import combinations
    
    for combo in combinations(range(n), clique_size):
        # Check if clique
        is_clique = True
        for i in range(clique_size):
            for j in range(i+1, clique_size):
                if not adj[combo[i]][combo[j]]:
                    is_clique = False
                    break
            if not is_clique:
                break
        if is_clique:
            return False
            
    for combo in combinations(range(n), ind_size):
        # Check if independent
        is_ind = True
        for i in range(ind_size):
            for j in range(i+1, ind_size):
                if adj[combo[i]][combo[j]]:
                    is_ind = False
                    break
            if not is_ind:
                break
        if is_ind:
            return False
            
    return True
