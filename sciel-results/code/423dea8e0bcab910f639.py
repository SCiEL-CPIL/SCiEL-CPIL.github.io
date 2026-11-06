import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Initial graph: Andrasfai graph And(9) on 26 vertices
    # Triangle-free, alpha = 9
    n0 = 26
    edges0 = []
    for i in range(n0):
        for j in range(i + 1, n0):
            if (j - i) % n0 % 3 == 1:
                edges0.append([i, j])
    
    best_graph = {'n': n0, 'edges': edges0}
    
    # Helper to build adjacency matrix from edge list
    def build_adj(n, edges):
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            if u < n and v < n:
                adj[u][v] = True
                adj[v][u] = True
        return adj

    # Check if graph is triangle-free and has no independent set of size 10
    def is_valid_graph(n, edges):
        adj = build_adj(n, edges)
        
        # Check triangles
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        
        # Check independent set of size 10
        # Use branch and bound to find if an independent set of size 10 exists
        # If it exists, graph is invalid.
        
        def has_independent_set_size_k(adj, k, n):
            # Returns True if there is an independent set of size k
            # We use a recursive search with pruning
            # State: current independent set size, available vertices
            # Optimization: If remaining vertices < k - current_size, return False
            # Heuristic: Pick vertex with min degree in available subgraph?
            
            # For speed, we limit the depth or use a simple heuristic
            # Since k=10 is small, B&B is fast if it finds one or proves none quickly.
            
            def dfs(available, current_size):
                if current_size == k:
                    return True
                if not available:
                    return False
                if len(available) < k - current_size:
                    return False
                
                # Pick a vertex
                # Randomly pick or pick first
                # To make it deterministic for validity check, pick first
                v = available[0]
                rest = available[1:]
                
                # Branch 1: Include v
                # New available: vertices in rest that are not adjacent to v
                new_avail = [u for u in rest if not adj[v][u]]
                if dfs(new_avail, current_size + 1):
                    return True
                
                # Branch 2: Exclude v
                if dfs(rest, current_size):
                    return True
                
                return False
            
            return dfs(list(range(n)), 0)
        
        if has_independent_set_size_k(adj, 10, n):
            return False
        
        return True

    # Heuristic check: Does G - S have an independent set of size 9?
    # If yes, adding a vertex connected to S would create an independent set of size 10 (1 + 9)
    # We use a randomized search to find such a set quickly.
    def has_independent_set_size_k_fast(adj, k, n, timeout=0.1):
        start_t = time.time()
        
        # We want to find an independent set of size k
        # Randomized DFS
        import sys
        sys.setrecursionlimit(10000)
        
        def dfs(available, current_size, path):
            if time.time() - start_t > timeout:
                return False, None
            if current_size == k:
                return True, path
            if not available:
                return False, None
            if len(available) < k - current_size:
                return False, None
            
            # Shuffle available to randomize search path
            # But for speed, maybe just pick a random vertex
            idx = random.randint(0, len(available) - 1)
            v = available[idx]
            rest = [u for i, u in enumerate(available) if i != idx]
            
            # Branch 1: Include v
            new_avail = [u for u in rest if not adj[v][u]]
            found, p = dfs(new_avail, current_size + 1, path + [v])
            if found:
                return True, p
            
            # Branch 2: Exclude v
            found, p = dfs(rest, current_size, path)
            if found:
                return True, p
            
            return False, None
        
        found, _ = dfs(list(range(n)), 0, [])
        return found

    # Main search loop
    while time.time() < deadline:
        current_n = best_graph['n']
        current_edges = [e[:] for e in best_graph['edges']]
        
        # Try to extend the current best graph
        # We will try to add vertices one by one
        extended = False
        for _ in range(20): # Try 20 times to add a single vertex
            if time.time() > deadline:
                break
            
            # Build adj for current graph
            adj = build_adj(current_n, current_edges)
            
            # Generate a random independent set S in the current graph
            # We use a simple greedy random independent set generation
            S = []
            available = list(range(current_n))
            random.shuffle(available)
            for v in available:
                # Check if v is independent from S
                if all(not adj[v][u] for u in S):
                    S.append(v)
            
            if not S:
                continue
                
            # Check if G - S has an independent set of size 9
            # If it does, we cannot add a vertex connected to S
            # We need alpha(G-S) <= 8
            # So we check if alpha(G-S) >= 9
            
            # Create subgraph G-S
            # Map original indices to new indices?
            # Easier: Just check in original adj, ignoring vertices in S
            
            # We need to check if there is an independent set of size 9 in G induced by V \ S
            # Let's create a list of vertices not in S
            non_S = [v for v in range(current_n) if v not in S]
            
            # Check if independent set of size 9 exists in non_S
            # We can reuse the fast check, but restricted to non_S
            # Let's write a specific check
            
            def has_ind_set_size_k_in_subset(adj, k, subset):
                if len(subset) < k:
                    return False
                start_t = time.time()
                def dfs(available, current_size):
                    if time.time() - start_t > 0.05: # Short timeout
                        return False
                    if current_size == k:
                        return True
                    if not available:
                        return False
                    if len(available) < k - current_size:
                        return False
                    
                    # Pick random vertex
                    idx = random.randint(0, len(available) - 1)
                    v = available[idx]
                    rest = [u for i, u in enumerate(available) if i != idx]
                    
                    # Include v
                    new_avail = [u for u in rest if not adj[v][u]]
                    if dfs(new_avail, current_size + 1):
                        return True
                    
                    # Exclude v
                    if dfs(rest, current_size):
                        return True
                    
                    return False
                
                return dfs(subset, 0)

            if has_ind_set_size_k_in_subset(adj, 9, non_S):
                continue
            else:
                # It seems safe to add a vertex connected to S
                new_v = current_n
                new_edges = current_edges.copy()
                for u in S:
                    new_edges.append([new_v, u])
                
                # Verify the new graph exactly
                new_n = current_n + 1
                if is_valid_graph(new_n, new_edges):
                    best_graph = {'n': new_n, 'edges': new_edges}
                    extended = True
                    break # Restart extension loop with new best
                else:
                    # Heuristic failed, try another S
                    continue
        
        if not extended:
            # Could not extend best_graph in this round
            # Try a different strategy?
            # For now, just loop again with new random seeds
            pass

    return best_graph

if __name__ == "__main__":
    g = construct()
    print(f"Found graph with n={g['n']}")
