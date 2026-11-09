import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def _paley_graph(p):
    """Constructs the Paley graph on p vertices (p prime, p % 4 == 1)."""
    squares = {x * x % p for x in range(1, p)}
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                edges.append([i, j])
    return {'n': p, 'edges': edges}

def _has_k4(n, edges):
    """Checks if the graph contains a K4."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Check all 4-cliques
    for a in range(n):
        for b in range(a + 1, n):
            if b not in adj[a]:
                continue
            for c in range(b + 1, n):
                if c not in adj[a] or c not in adj[b]:
                    continue
                for d in range(c + 1, n):
                    if d not in adj[a] or d not in adj[b] or d not in adj[c]:
                        continue
                    return True
    return False

def _has_independent_set(n, edges, k):
    """Checks if the graph contains an independent set of size k."""
    # An independent set in G is a clique in the complement graph G_bar
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Build complement adjacency
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp_adj[i].add(j)
                comp_adj[j].add(i)
                
    # Find clique of size k in comp_adj
    # Use backtracking
    def dfs(candidates, current_size):
        if current_size == k:
            return True
        if not candidates:
            return False
        if len(candidates) + current_size < k:
            return False
            
        # Pick a vertex from candidates
        # Heuristic: pick vertex with max degree in induced subgraph?
        # Or just first one.
        v = candidates[0]
        rest = candidates[1:]
        
        # Branch 1: Include v
        # New candidates must be neighbors of v in comp_adj
        new_candidates = [u for u in rest if u in comp_adj[v]]
        if dfs(new_candidates, current_size + 1):
            return True
            
        # Branch 2: Exclude v
        if dfs(rest, current_size):
            return True
            
        return False

    return dfs(list(range(n)), 0)

def _is_valid(graph):
    n = graph['n']
    edges = graph['edges']
    if _has_k4(n, edges):
        return False
    if _has_independent_set(n, edges, 6):
        return False
    return True

def _reduce_graph(graph, target_alpha=5, max_removals=10):
    """
    Attempts to remove vertices to reduce independence number.
    Returns a new graph dict or None if it fails to meet constraints or takes too long.
    """
    n = graph['n']
    edges = graph['edges']
    
    # We need to ensure no independent set of size 6.
    # Current graph might have one.
    # We iteratively find an independent set of size 6 and remove a vertex from it.
    
    current_vertices = set(range(n))
    removed = set()
    
    # We need to re-check validity after each removal.
    # To speed up, we can just check for independent set of size 6.
    # K4 constraint is preserved by removing vertices.
    
    for _ in range(max_removals):
        # Check if current induced subgraph has independent set of size 6
        # We need to map vertices to 0..n-1 for the checker
        # Or just write a checker that works on subsets.
        
        # Let's create a subgraph
        sub_edges = []
        for u, v in edges:
            if u in current_vertices and v in current_vertices:
                sub_edges.append([u, v])
                
        # Check if subgraph has independent set of size 6
        # We can reuse _has_independent_set if we remap, but it's easier to just pass the set
        # Let's modify _has_independent_set to take a list of active vertices?
        # No, let's just remap.
        
        active_list = sorted(current_vertices)
        if len(active_list) < 6:
             break
             
        remap = {v: i for i, v in enumerate(active_list)}
        sub_edges_remap = []
        for u, v in sub_edges:
            sub_edges_remap.append([remap[u], remap[v]])
            
        if not _has_independent_set(len(active_list), sub_edges_remap, 6):
            # Valid
            # Construct final graph with original labels
            final_edges = []
            for u, v in sub_edges:
                final_edges.append([u, v])
            return {'n': len(active_list), 'edges': final_edges}
            
        # Find the independent set to remove a vertex from
        # We need to find WHICH set.
        # Modify _has_independent_set to return the set?
        # Let's write a helper to find one.
        
        def find_one_independent_set(n, edges, k):
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            comp_adj = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if j not in adj[i]:
                        comp_adj[i].add(j)
                        comp_adj[j].add(i)
                        
            def dfs(candidates, current_set):
                if len(current_set) == k:
                    return current_set
                if not candidates:
                    return None
                if len(candidates) + len(current_set) < k:
                    return None
                    
                v = candidates[0]
                rest = candidates[1:]
                
                new_candidates = [u for u in rest if u in comp_adj[v]]
                res = dfs(new_candidates, current_set + [v])
                if res:
                    return res
                res = dfs(rest, current_set)
                if res:
                    return res
                return None
                
            return dfs(list(range(n)), [])

        found_set = find_one_independent_set(len(active_list), sub_edges_remap, 6)
        if not found_set:
            # Should not happen if _has_independent_set returned True
            break
            
        # Remove the first vertex in the found set (mapped back to original)
        v_to_remove = active_list[found_set[0]]
        current_vertices.remove(v_to_remove)
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try Paley 37
    p37 = _paley_graph(37)
    if time.time() < deadline:
        if _is_valid(p37):
            return p37
        # Try to reduce
        reduced = _reduce_graph(p37)
        if reduced and reduced['n'] > 29:
            # Verify reduced is valid (just in case)
            if _is_valid(reduced):
                return reduced

    # Try Paley 29
    if time.time() < deadline:
        p29 = _paley_graph(29)
        if _is_valid(p29):
            return p29
        # Try to reduce? Unlikely to be invalid if theory holds, but check
        reduced29 = _reduce_graph(p29)
        if reduced29 and _is_valid(reduced29):
            return reduced29

    # Fallback to 17
    p17 = _paley_graph(17)
    return p17

# EVOLVE-BLOCK-END
