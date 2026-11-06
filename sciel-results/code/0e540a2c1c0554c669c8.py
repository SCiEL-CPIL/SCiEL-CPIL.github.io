import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def construct_A(k):
        """Construct Andrásfai graph A_k on 3k-2 vertices."""
        n = 3 * k - 2
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 3 == 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    def get_max_independent_sets(graph, target_size):
        """
        Find a few maximum independent sets of size target_size.
        Uses a simple backtracking search.
        """
        n = graph['n']
        adj = [[0] * n for _ in range(n)]
        for u, v in graph['edges']:
            adj[u][v] = 1
            adj[v][u] = 1
        
        sets_found = []
        
        def dfs(candidates, current_set, depth):
            if len(sets_found) >= 20: # Limit number of sets to find
                return
            if len(current_set) == target_size:
                sets_found.append(list(current_set))
                return
            if depth >= n:
                return
            if len(current_set) + len(candidates) < target_size:
                return
                
            v = candidates[0]
            rest = candidates[1:]
            
            # Include v
            new_candidates = [u for u in rest if adj[v][u] == 0]
            current_set.append(v)
            dfs(new_candidates, current_set, depth + 1)
            current_set.pop()
            
            # Exclude v
            dfs(rest, current_set, depth + 1)

        dfs(list(range(n)), [], 0)
        return sets_found

    def find_hitting_set(independent_sets, max_size):
        """
        Find a set of vertices of size <= max_size that intersects all given independent sets.
        Uses greedy approach: pick vertex that appears in most sets.
        """
        if not independent_sets:
            return []
        
        remaining_sets = [set(s) for s in independent_sets]
        hitting_set = []
        
        while remaining_sets:
            if len(hitting_set) >= max_size:
                return None # Failed to find hitting set of this size
            
            # Count frequency of each vertex in remaining sets
            freq = {}
            for s in remaining_sets:
                for v in s:
                    freq[v] = freq.get(v, 0) + 1
            
            if not freq:
                break
                
            # Pick vertex with max frequency
            best_v = max(freq, key=freq.get)
            hitting_set.append(best_v)
            
            # Remove all sets that contain best_v
            remaining_sets = [s for s in remaining_sets if best_v not in s]
            
        return hitting_set

    def is_valid(graph):
        """Check triangle-free and alpha < 10."""
        n = graph['n']
        adj = [[0] * n for _ in range(n)]
        for u, v in graph['edges']:
            adj[u][v] = 1
            adj[v][u] = 1
            
        # Triangle-free check
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
                            
        # Check alpha < 10
        # Use a quick branch and bound to see if alpha >= 10
        best = 0
        def dfs(candidates, current_set):
            nonlocal best
            if len(current_set) + len(candidates) <= best:
                return
            if len(current_set) >= 10:
                best = 10
                return
            if not candidates:
                if len(current_set) > best:
                    best = len(current_set)
                return
            v = candidates[0]
            rest = candidates[1:]
            new_candidates = [u for u in rest if adj[v][u] == 0]
            dfs(new_candidates, current_set + [v])
            if best >= 10:
                return
            dfs(rest, current_set)

        dfs(list(range(n)), [])
        return best < 10

    # Base case: A_9 (n=26)
    base_graph = construct_A(9)
    if is_valid(base_graph):
        best_graph = base_graph
    else:
        best_graph = None

    # Try to improve by taking A_k for k > 9 and removing vertices
    # Target alpha <= 9
    # A_k has alpha = k
    # Remove k-9 vertices
    
    # We want to maximize n_final = 3k - 2 - (k - 9) = 2k + 7
    # For k=16, n_final = 39.
    # For k=15, n_final = 37.
    # For k=14, n_final = 35.
    
    for k in range(16, 8, -1):
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
        
        # Find some max independent sets in A_k
        # To save time, we don't need ALL, just enough to find a hitting set
        # If the hitting set works, we verify the final graph.
        
        max_sets = get_max_independent_sets(g_base, k)
        
        if not max_sets:
            continue
            
        # Try to find a hitting set of size m
        # Greedy might not be optimal, but let's try
        # Also try random permutations of vertices in the hitting set selection?
        # For now, greedy.
        
        hitting_set = find_hitting_set(max_sets, m)
        
        if hitting_set is None:
            # Greedy failed, maybe try a different set of independent sets?
            # Or just skip this k
            continue
            
        # Remove hitting_set from g_base
        remove_set = set(hitting_set)
        keep_set = [v for v in range(n_base) if v not in remove_set]
        
        # Build subgraph
        idx_map = {v: i for i, v in enumerate(keep_set)}
        sub_edges = []
        for u, v in g_base['edges']:
            if u in idx_map and v in idx_map:
                sub_edges.append([idx_map[u], idx_map[v]])
                
        sub_graph = {'n': n_final, 'edges': sub_edges}
        
        if is_valid(sub_graph):
            best_graph = sub_graph
            # We found a valid graph with n_final.
            # Since we are iterating k downwards, and n_final increases with k,
            # if we find one for k=16, it's the best possible for this family.
            # But we should check if we can get higher n with other k?
            # n_final = 2k + 7. Higher k gives higher n.
            # So if k=16 works, we can break.
            break

    if best_graph is None:
        # Fallback to A_9
        return construct_A(9)
        
    return best_graph
