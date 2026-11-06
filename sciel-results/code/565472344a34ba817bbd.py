import random
import time
from itertools import combinations

def _find_triangle(adj):
    """Find a triangle in the graph. Returns (u, v, w) or None."""
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if v > u: # Avoid duplicates
                # Check common neighbors of u and v
                common = adj[u] & adj[v]
                if common:
                    w = next(iter(common))
                    return (u, v, w)
    return None

def _find_independent_set(adj, size):
    """Find an independent set of given size. Returns list of vertices or None."""
    n = len(adj)
    
    # Backtracking to find independent set of size 'size'
    def backtrack(candidates, current_set):
        if len(current_set) == size:
            return current_set
        if not candidates:
            return None
        
        # Heuristic: pick vertex with min degree in candidates to prune faster?
        # Or random. Random helps escape local minima in the search tree.
        v_list = list(candidates)
        random.shuffle(v_list)
        
        for v in v_list:
            # Try including v
            new_candidates = candidates - {v} - adj[v]
            result = backtrack(new_candidates, current_set + [v])
            if result:
                return result
        return None

    all_vertices = set(range(n))
    return backtrack(all_vertices, [])

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    def try_build(n, time_limit):
        """Try to build a valid graph on n vertices within time_limit."""
        if time.time() > start_time + time_limit:
            return None
            
        # Initialize random graph
        p = 0.15
        
        for _ in range(50): # 50 restarts
            if time.time() > start_time + time_limit:
                return None
                
            # Init graph
            adj = [set() for _ in range(n)]
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < p:
                        adj[u].add(v)
                        adj[v].add(u)
            
            # Local Search
            max_iter = 2000
            for _ in range(max_iter):
                if time.time() > start_time + time_limit:
                    return None
                    
                # 1. Remove Triangles
                tri = _find_triangle(adj)
                if tri:
                    u, v, w = tri
                    # Remove random edge from triangle
                    r = random.random()
                    if r < 0.33:
                        adj[u].remove(v); adj[v].remove(u)
                    elif r < 0.66:
                        adj[v].remove(w); adj[w].remove(v)
                    else:
                        adj[u].remove(w); adj[w].remove(u)
                    continue # Restart loop to find next violation
                
                # 2. Check Independent Set
                indep_set = _find_independent_set(adj, 10)
                if not indep_set:
                    # Success
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                
                # 3. Break Independent Set
                # Try to add a safe edge between two vertices in indep_set
                # Safe edge (u,v) means no common neighbor.
                added = False
                random.shuffle(indep_set)
                for i in range(len(indep_set)):
                    for j in range(i+1, len(indep_set)):
                        u = indep_set[i]
                        v = indep_set[j]
                        if v not in adj[u]: # Not already connected
                            # Check if safe
                            if not (adj[u] & adj[v]):
                                adj[u].add(v)
                                adj[v].add(u)
                                added = True
                                break
                    if added:
                        break
                
                if not added:
                    # Stuck. Try removing a random edge to create space?
                    all_edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                all_edges.append((u, v))
                    if all_edges:
                        u, v = random.choice(all_edges)
                        adj[u].remove(v)
                        adj[v].remove(u)
                    else:
                        break # Empty graph, alpha=n, stuck.
                    
        return None

    # Try n=40
    result = try_build(40, 60)
    if result:
        return result
        
    # Try n=39
    if time.time() < deadline:
        result = try_build(39, 40)
        if result:
            return result
            
    # Try n=38
    if time.time() < deadline:
        result = try_build(38, 20)
        if result:
            return result
            
    # Fallback: Return a small valid graph
    # K_{9,9} on 18 vertices is triangle free, alpha=9.
    n = 18
    edges = [[i, j] for i in range(9) for j in range(9, 18)]
    return {'n': n, 'edges': edges}
