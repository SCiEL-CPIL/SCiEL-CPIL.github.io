import time
import random
from collections import defaultdict, deque

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycle(adj, used_edges):
    """Find a cycle in the remaining graph using DFS."""
    n = max(adj.keys()) + 1 if adj else 0
    visited = [False] * n
    parent = [-1] * n
    cycle = []

    def dfs(u):
        visited[u] = True
        for v in adj[u]:
            if (u, v) in used_edges or (v, u) in used_edges:
                continue
            if not visited[v]:
                parent[v] = u
                if dfs(v):
                    return True
            elif v != parent[u]:
                # Found a back edge, extract cycle
                cycle.clear()
                cur = u
                while cur != v:
                    cycle.append(cur)
                    cur = parent[cur]
                cycle.append(v)
                return True
        return False

    for u in range(n):
        if not visited[u]:
            if dfs(u):
                return cycle
    return None

def _find_long_cycle(adj, used_edges, max_len=None):
    """Try to find a longer cycle using randomized greedy approach."""
    n = max(adj.keys()) + 1 if adj else 0
    if n == 0:
        return None
    
    # Try multiple random starts
    best_cycle = None
    for attempt in range(20):
        start = random.randint(0, n - 1)
        visited = [False] * n
        path = [start]
        visited[start] = True
        
        current = start
        while True:
            # Find unvisited neighbors not using used edges
            neighbors = []
            for v in adj[current]:
                if (current, v) in used_edges or (v, current) in used_edges:
                    continue
                if not visited[v]:
                    neighbors.append(v)
                
            if not neighbors:
                break
            
            # Prefer neighbors that lead to longer paths or create cycles
            if len(neighbors) == 1:
                next_v = neighbors[0]
            else:
                # Random choice with some heuristic
                next_v = random.choice(neighbors)
            
            # Check if this creates a cycle
            if next_v == start and len(path) >= 3:
                cycle = path.copy()
                if max_len is None or len(cycle) <= max_len:
                    if best_cycle is None or len(cycle) > len(best_cycle):
                        best_cycle = cycle
                break
            
            if visited[next_v] and next_v != start:
                # Found a cycle but not back to start
                # Extract the cycle
                if next_v in path:
                    idx = path.index(next_v)
                    cycle = path[idx:]
                    if len(cycle) >= 3:
                        if max_len is None or len(cycle) <= max_len:
                            if best_cycle is None or len(cycle) > len(best_cycle):
                                best_cycle = cycle
                break
            
            visited[next_v] = True
            path.append(next_v)
            current = next_v
        
        if best_cycle and (max_len is not None and len(best_cycle) >= max_len):
            break
    
    return best_cycle

def _decompose_once(edges, time_limit=None, start_time=None):
    """Decompose edges into cycles and single edges, minimizing total pieces."""
    if start_time is None:
        start_time = time.time()
    
    if time_limit is not None and time.time() - start_time > time_limit:
        return [[e[0], e[1]] for e in edges]
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    remaining = set()
    for u, v in edges:
        remaining.add(tuple(sorted((u, v))))
    
    pieces = []
    
    # Repeatedly find cycles
    while remaining:
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Try to find a cycle
        found = False
        for _ in range(5):  # Multiple attempts
            cycle = _find_long_cycle(adj, set())
            if cycle and len(cycle) >= 3:
                # Check if all edges in cycle are available
                valid = True
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    edge = tuple(sorted((u, v)))
                    if edge not in remaining:
                        valid = False
                        break
                if valid:
                    # Use this cycle
                    pieces.append(cycle)
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        remaining.discard(tuple(sorted((u, v))))
                    found = True
                    break
        
        if not found:
            # No cycle found, take a single edge
            if remaining:
                edge = next(iter(remaining))
                remaining.discard(edge)
                pieces.append([edge[0], edge[1]])
    
    return pieces

def _decompose_multi(edges, time_budget=100.0):
    """Try multiple decomposition strategies and return the best."""
    start_time = time.time()
    best_pieces = None
    best_count = float('inf')
    
    # Strategy 1: Greedy cycle finding
    pieces = _decompose_once(edges, time_limit=time_budget * 0.3, start_time=start_time)
    if len(pieces) < best_count:
        best_count = len(pieces)
        best_pieces = pieces
    
    # Strategy 2: Multiple random restarts
    for restart in range(3):
        if time.time() - start_time > time_budget * 0.7:
            break
        random.seed()  # Reset random state for different exploration
        pieces = _decompose_once(edges, time_limit=time_budget * 0.15, start_time=start_time)
        if len(pieces) < best_count:
            best_count = len(pieces)
            best_pieces = pieces
    
    # Final pass: try to improve by combining small pieces
    if best_pieces is None:
        best_pieces = [[e[0], e[1]] for e in edges]
    
    return best_pieces

def construct():
    start_time = time.time()
    total_budget = 140.0  # Safety margin before 150s limit
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for spec in specs:
        if time.time() - start_time > total_budget:
            # Not enough time, return trivial solution
            edges = _instance(*spec)
            results.append([[e[0], e[1]] for e in edges])
            continue
        
        edges = _instance(*spec)
        # Allocate time proportionally to graph size
        time_budget = min(40.0, (total_budget - (time.time() - start_time)) / 3)
        pieces = _decompose_multi(edges, time_budget=time_budget)
        results.append(pieces)
    
    return results
