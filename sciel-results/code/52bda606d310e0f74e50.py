import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    
    for n, p, seed in specs:
        if time.time() > deadline:
            results.append([])
            continue
            
        graph = generate_instance(n, p, seed)
        pieces = decompose(graph, deadline)
        results.append(pieces)
        
    return results

def generate_instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = set()
    for i in range(n - 1):
        u, v = order[i], order[i+1]
        if u > v: u, v = v, u
        edges.add((u, v))
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return edges

def decompose(edges, deadline):
    remaining_edges = set(edges)
    if not remaining_edges:
        return []
    
    n = max([max(u, v) for u, v in edges] + [0]) + 1
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in remaining_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components
    components = find_components(adj, n)
    
    all_pieces = []
    
    for comp in components:
        if time.time() > deadline:
            # Fallback: add remaining edges as single pieces
            for u, v in remaining_edges:
                all_pieces.append([u, v])
            return all_pieces
        
        comp_edges = set()
        for u, v in remaining_edges:
            if u in comp and v in comp:
                comp_edges.add((u, v))
        
        if not comp_edges:
            continue
        
        # Decompose this component
        comp_pieces = decompose_component(comp_edges, comp, deadline)
        all_pieces.extend(comp_pieces)
        
        # Remove used edges from remaining
        for piece in comp_pieces:
            if len(piece) == 2:
                u, v = piece
                e = (u, v) if u < v else (v, u)
                remaining_edges.discard(e)
            else:
                for i in range(len(piece)):
                    u, v = piece[i], piece[(i + 1) % len(piece)]
                    e = (u, v) if u < v else (v, u)
                    remaining_edges.discard(e)
    
    # Add any remaining edges as single pieces
    for u, v in remaining_edges:
        all_pieces.append([u, v])
    
    return all_pieces

def find_components(adj, n):
    visited = set()
    components = []
    
    for node in range(n):
        if node not in visited:
            # BFS/DFS to find component
            component = set()
            stack = [node]
            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                component.add(v)
                for nb in adj[v]:
                    if nb not in visited:
                        stack.append(nb)
            if component:
                components.append(component)
    
    return components

def decompose_component(comp_edges, comp, deadline):
    """Decompose a connected component into cycles and edges."""
    remaining = set(comp_edges)
    if not remaining:
        return []
    
    # Build local adjacency
    adj = defaultdict(set)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Find a cycle in the remaining graph
        cycle = find_cycle(remaining, adj, deadline)
        
        if cycle:
            pieces.append(cycle)
            # Remove cycle edges
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = (u, v) if u < v else (v, u)
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
        else:
            # No cycle found, add remaining edges as single pieces
            for u, v in remaining:
                pieces.append([u, v])
            break
    
    return pieces

def find_cycle(remaining, adj, deadline):
    """Find a cycle in the graph using randomized DFS with multiple restarts."""
    best_cycle = None
    best_len = 0
    
    # Get all vertices with degree >= 2
    candidates = [v for v in adj if len(adj[v]) >= 2]
    if not candidates:
        return None
    
    # Sort by degree descending
    candidates.sort(key=lambda x: len(adj[x]), reverse=True)
    
    # Try multiple random restarts
    rng = random.Random()
    
    for _ in range(20):  # 20 restarts
        if time.time() > deadline:
            break
        
        # Pick a random candidate
        start = rng.choice(candidates[:min(10, len(candidates))])
        
        # Try to find a cycle starting from this vertex
        cycle = dfs_find_cycle(start, adj, deadline)
        if cycle and len(cycle) > best_len:
            best_cycle = cycle
            best_len = len(cycle)
            
            # If we found a very long cycle, we can stop early
            if best_len >= 10:
                break
    
    return best_cycle

def dfs_find_cycle(start, adj, deadline):
    """Find a cycle starting from 'start' using randomized DFS."""
    path = [start]
    visited = {start}
    
    def dfs(u):
        if time.time() > deadline:
            return None
        
        # Check if we can close the cycle
        for v in adj[u]:
            if v == start and len(path) >= 3:
                return path[:]
        
        # Try to extend the path
        neighbors = list(adj[u])
        random.shuffle(neighbors)  # Randomize for diversity
        
        for v in neighbors:
            if v not in visited:
                visited.add(v)
                path.append(v)
                result = dfs(v)
                if result:
                    return result
                path.pop()
                visited.discard(v)
        
        return None
    
    return dfs(start)
