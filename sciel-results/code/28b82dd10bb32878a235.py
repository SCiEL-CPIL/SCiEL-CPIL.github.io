import time
import random
from collections import defaultdict

def instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def find_cycle(adj, start):
    # Returns a cycle as a list of vertices or None
    visited = set()
    parent = {}
    stack = []
    
    def dfs(u):
        visited.add(u)
        stack.append(u)
        for v in adj[u]:
            if v not in visited:
                parent[v] = u
                if dfs(v):
                    return True
            elif v in stack:
                # Cycle found
                # Extract cycle from v to u in stack
                idx = stack.index(v)
                cycle = stack[idx:]
                return cycle
        stack.pop()
        return None

    # Randomize neighbors to get different cycles in different attempts
    for u in adj:
        random.shuffle(adj[u])
        
    result = dfs(start)
    if result:
        return result
    return None

def decompose(edges, time_limit=10.0):
    start_time = time.time()
    best_pieces = None
    best_len = float('inf')
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    
    # We will run multiple attempts with randomization
    # Number of attempts depends on time
    max_attempts = 100
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        current_edges = set(edges_set)
        pieces = []
        
        while current_edges:
            # Build adjacency list
            adj = defaultdict(list)
            for u, v in current_edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # Pick a random start vertex with degree > 0
            vertices = [v for v in adj if adj[v]]
            if not vertices:
                break
                
            random.shuffle(vertices)
            cycle_found = None
            # Try a few random starts if the first fails? 
            # Actually, if the graph has a cycle, DFS from any vertex in that component will find it.
            # If the start vertex is not in a cyclic component, it returns None.
            # So we should iterate through vertices or pick one carefully.
            # Better: Just pick one random vertex. If no cycle found, it means that component is a tree.
            # But there might be other components with cycles.
            # So we need to check all components or just pick a vertex that is likely in a cycle.
            # Simpler: Just iterate through all vertices as start points until a cycle is found.
            # But that's expensive.
            
            # Optimized: 
            # 1. Find connected components.
            # 2. For each component, check if it has a cycle (edges >= vertices).
            # 3. If yes, run DFS from a vertex in that component.
            
            # Let's implement component check for efficiency and correctness.
            visited_comp = set()
            cycle_to_remove = None
            
            for v in vertices:
                if v in visited_comp:
                    continue
                # BFS/DFS to find component
                comp = []
                stack_c = [v]
                visited_comp.add(v)
                while stack_c:
                    node = stack_c.pop()
                    comp.append(node)
                    for nb in adj[node]:
                        if nb not in visited_comp:
                            visited_comp.add(nb)
                            stack_c.append(nb)
                
                # Check if component has a cycle
                # Count edges in component
                comp_set = set(comp)
                comp_edges = 0
                for u, w in current_edges:
                    if u in comp_set and w in comp_set:
                        comp_edges += 1
                        
                if comp_edges >= len(comp):
                    # Has cycle
                    # Run DFS from v to find a specific cycle
                    # We need to be careful with the DFS implementation to handle the specific component
                    # Or just run the global DFS which will find *a* cycle in the whole graph.
                    # Since we are in the component, it will find one here.
                    
                    # Let's just use the simple DFS on the whole graph but starting from v.
                    # If it finds a cycle, great.
                    # If not, it means my component logic was wrong or DFS failed?
                    # No, if edges >= vertices, there is a cycle. DFS must find it.
                    
                    # Reset adj shuffling for this attempt?
                    # We shuffled adj at the beginning of the while loop? No, inside.
                    # Let's shuffle adj[u] for all u before DFS.
                    for u in adj:
                        random.shuffle(adj[u])
                    
                    # Custom DFS that returns the cycle
                    visited = set()
                    parent = {}
                    path = []
                    
                    def dfs(u):
                        visited.add(u)
                        path.append(u)
                        for w in adj[u]:
                            if w not in visited:
                                parent[w] = u
                                res = dfs(w)
                                if res:
                                    return res
                            elif w in path:
                                idx = path.index(w)
                                return path[idx:]
                        path.pop()
                        return None
                    
                    res = dfs(v)
                    if res:
                        cycle_to_remove = res
                        break # Found a cycle, stop looking for other components
                    
            if cycle_to_remove:
                # Add cycle to pieces
                # Ensure unique vertices
                if len(set(cycle_to_remove)) < 3:
                    # Should not happen for simple cycle
                    continue
                pieces.append(cycle_to_remove)
                
                # Remove edges of the cycle
                for i in range(len(cycle_to_remove)):
                    u = cycle_to_remove[i]
                    v = cycle_to_remove[(i + 1) % len(cycle_to_remove)]
                    edge = tuple(sorted((u, v)))
                    current_edges.discard(edge)
            else:
                # No cycle found in any component?
                # Then all remaining edges are in trees.
                # Add all remaining edges as single pieces
                for u, v in current_edges:
                    pieces.append([u, v])
                break
                
        if len(pieces) < best_len:
            best_len = len(pieces)
            best_pieces = pieces
            
    if best_pieces is None:
        # Fallback: all edges
        return [list(e) for e in edges]
        
    return best_pieces

def construct():
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    total_deadline = 100.0
    start_time = time.time()
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        # Allocate time based on remaining budget
        remaining = total_deadline - (time.time() - start_time)
        # Be conservative, use 1/3 of remaining for each, or fixed if enough time
        time_limit = min(30.0, remaining / 3.0)
        if time_limit < 5.0:
             time_limit = 5.0 # Minimum effort
        
        pieces = decompose(edges, time_limit)
        results.append(pieces)
        
    return results
