import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n = 16
    all_edges = list(combinations(range(n), 2))
    
    def is_triangle_free(edge_set):
        """Check if graph with given edges is triangle-free."""
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            return False
        return True
    
    def is_3_colorable(edge_set, max_time=None):
        """Check if graph is 3-colorable using backtracking."""
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        color = [-1] * n
        
        def backtrack(vertex):
            if time.time() > deadline:
                return False  # timeout, assume not colorable (conservative)
            if vertex == n:
                return True
            for c in range(3):
                if all(color[neighbor] != c for neighbor in adj[vertex]):
                    color[vertex] = c
                    if backtrack(vertex + 1):
                        return True
                    color[vertex] = -1
            return False
        
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        
        # Remap
        new_order = [0] * n
        for i, v in enumerate(order):
            new_order[v] = i
        
        adj_remapped = [set() for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                adj_remapped[new_order[u]].add(new_order[v])
        
        color = [-1] * n
        def backtrack2(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            vertex = order[idx]
            for c in range(3):
                if all(color[neighbor] != c for neighbor in adj[vertex]):
                    color[vertex] = c
                    if backtrack2(idx + 1):
                        return True
                    color[vertex] = -1
            return False
        
        return backtrack2(0)
    
    def valid_graph(edge_list):
        edge_set = set(tuple(sorted(e)) for e in edge_list)
        if not is_triangle_free(edge_set):
            return False
        if is_3_colorable(edge_list):
            return False
        return True
    
    # Start with Grötzsch graph on vertices 0-10
    groetzsch_edges = []
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, w in cycle:
        groetzsch_edges.append([u, w])
    for u, w in cycle:
        groetzsch_edges.append([u, 5 + w])
        groetzsch_edges.append([w, 5 + u])
    for u in range(5):
        groetzsch_edges.append([5 + u, 10])
    
    best_edges = groetzsch_edges[:]
    best_count = len(groetzsch_edges)
    
    # Try to add edges to improve
    def try_add_edges(current_edges, num_trials=1000):
        nonlocal best_edges, best_count
        current = current_edges[:]
        current_set = set(tuple(sorted(e)) for e in current)
        
        # Candidate edges not already in graph
        candidates = []
        for u, v in all_edges:
            if (u, v) not in current_set and (v, u) not in current_set:
                candidates.append([u, v])
        
        random.shuffle(candidates)
        
        for edge in candidates:
            if time.time() > deadline:
                break
            u, v = edge
            new_edges = current + [edge]
            new_set = current_set | {tuple(sorted(edge))}
            
            # Quick triangle check: only check triangles involving u or v
            adj = [set() for _ in range(n)]
            for a, b in new_set:
                adj[a].add(b)
                adj[b].add(a)
            
            has_triangle = False
            for w in adj[u]:
                if w in adj[v]:
                    has_triangle = True
                    break
            if has_triangle:
                continue
            
            # Check 3-colorability
            if is_3_colorable(new_edges):
                continue
            
            # Valid! Add it
            current = new_edges
            current_set = new_set
            if len(current) > best_count:
                best_edges = current[:]
                best_count = len(current)
        
        return current
    
    # Multiple restarts with different strategies
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Start from best found so far
        current = best_edges[:]
        
        # Add some randomness: remove a few random edges and re-add
        if restart > 0 and random.random() < 0.3:
            current = best_edges[:]
            # Remove 2-5 random edges to explore
            num_remove = random.randint(2, 5)
            to_remove = random.sample(current, min(num_remove, len(current)))
            current = [e for e in current if e not in to_remove]
        
        try_add_edges(current)
    
    return best_edges
