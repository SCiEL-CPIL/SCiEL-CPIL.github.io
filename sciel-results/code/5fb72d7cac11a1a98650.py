import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def construct_A(k):
        """Construct Andrásfai graph A_k on 3k-2 vertices."""
        n = 3 * k - 2
        edges = []
        # i ~ j iff (j - i) % n is 1 mod 3
        # The set of differences D = {1, 4, ..., 1+3(k-1)}
        # We just check (j-i) % 3 == 1
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 3 == 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    def has_independent_set_of_size_10(graph):
        """Check if graph has an independent set of size 10."""
        n = graph['n']
        if n < 10:
            return False
        # Build adjacency matrix for speed
        adj = [[0] * n for _ in range(n)]
        for u, v in graph['edges']:
            adj[u][v] = 1
            adj[v][u] = 1
        
        # Branch and bound to find max independent set size
        # We just need to know if it's >= 10
        best = 0
        
        def dfs(candidates, current_set):
            nonlocal best
            if len(current_set) + len(candidates) <= best:
                return
            if len(current_set) >= 10:
                best = 10
                return
            
            # Heuristic: pick vertex with max degree in candidates?
            # Or just simple recursion
            if not candidates:
                if len(current_set) > best:
                    best = len(current_set)
                return
            
            v = candidates[0]
            rest = candidates[1:]
            
            # Option 1: Include v
            # New candidates must be non-neighbors of v in current candidates
            new_candidates = [u for u in rest if adj[v][u] == 0]
            dfs(new_candidates, current_set + [v])
            
            if best >= 10:
                return
                
            # Option 2: Exclude v
            dfs(rest, current_set)

        dfs(list(range(n)), [])
        return best >= 10

    def is_valid(graph):
        """Check triangle-free and alpha < 10."""
        # Triangle-free check
        n = graph['n']
        adj = [[0] * n for _ in range(n)]
        for u, v in graph['edges']:
            adj[u][v] = 1
            adj[v][u] = 1
            
        # Check triangles
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
                            
        # Check alpha
        if has_independent_set_of_size_10(graph):
            return False
            
        return True

    # Base case: A_9 (n=26)
    base_graph = construct_A(9)
    if is_valid(base_graph):
        best_graph = base_graph
    else:
        # Should not happen, but fallback
        best_graph = None

    # Try to improve by taking A_k for k > 9 and removing vertices
    # Target alpha <= 9
    # A_k has alpha = k
    # Remove k-9 vertices
    
    max_k = 20  # A_20 has n=58, remove 11 vertices -> n=47. Might be too big for search?
                 # But we stop if time runs out.
    
    for k in range(15, 8, -1):
        if time.time() > deadline:
            break
            
        n_base = 3 * k - 2
        m = k - 9
        if m < 0:
            continue
            
        n_final = n_base - m
        if n_final <= (best_graph['n'] if best_graph else 0):
            continue
            
        g_base = construct_A(k)
        
        # Generate subsets of vertices to remove
        # To make it faster, we can try random subsets first, then systematic if time permits
        # But for small m, systematic is fine.
        
        # Shuffle to find good ones faster?
        # Let's try systematic for small m, random for large m
        
        vertices = list(range(n_base))
        
        found = False
        if m <= 5:
            for remove_set in itertools.combinations(vertices, m):
                if time.time() > deadline:
                    break
                keep_set = [v for v in vertices if v not in remove_set]
                # Build subgraph
                # Map old indices to new
                # Or just filter edges
                sub_edges = []
                idx_map = {v: i for i, v in enumerate(keep_set)}
                for u, v in g_base['edges']:
                    if u in idx_map and v in idx_map:
                        sub_edges.append([idx_map[u], idx_map[v]])
                
                sub_graph = {'n': n_final, 'edges': sub_edges}
                
                if is_valid(sub_graph):
                    best_graph = sub_graph
                    found = True
                    break
        else:
            # Random search
            import random
            for _ in range(1000):
                if time.time() > deadline:
                    break
                remove_set = set(random.sample(vertices, m))
                keep_set = [v for v in vertices if v not in remove_set]
                idx_map = {v: i for i, v in enumerate(keep_set)}
                sub_edges = []
                for u, v in g_base['edges']:
                    if u in idx_map and v in idx_map:
                        sub_edges.append([idx_map[u], idx_map[v]])
                
                sub_graph = {'n': n_final, 'edges': sub_edges}
                
                if is_valid(sub_graph):
                    best_graph = sub_graph
                    found = True
                    break

    if best_graph is None:
        # Fallback to A_9
        return construct_A(9)
        
    return best_graph
