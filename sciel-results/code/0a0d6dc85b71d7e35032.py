import time
import random
from collections import defaultdict

_DEADLINE_SECONDS = 145.0

def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_longest_cycle(adj, start, n):
    """
    Find the longest simple cycle starting and ending at 'start'.
    """
    best_cycle = None
    best_len = 0
    
    visited = [False] * n
    path = [start]
    visited[start] = True
    
    def dfs(u, current_len):
        nonlocal best_cycle, best_len
        if current_len > best_len:
            if start in adj[u] and current_len >= 2:
                if current_len > best_len:
                    best_len = current_len
                    best_cycle = path[:]
        
        for v in adj[u]:
            if v == start:
                continue
            if not visited[v]:
                visited[v] = True
                path.append(v)
                dfs(v, current_len + 1)
                path.pop()
                visited[v] = False
                
    dfs(start, 1)
    return best_cycle

def _greedy_longest_decompose(edges, n, rng):
    """
    Greedily find edge-disjoint cycles, preferring the longest one.
    rng is used to shuffle node order for tie-breaking/restarts.
    """
    if not edges:
        return []
    
    remaining = set((min(u, v), max(u, v)) for u, v in edges)
    cycles = []
    
    while remaining:
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        best_cycle = None
        best_len = 0
        
        nodes = list(range(n))
        rng.shuffle(nodes)
        
        for start in nodes:
            if not adj[start]:
                continue
            cycle = _find_longest_cycle(adj, start, n)
            if cycle and len(cycle) > best_len:
                best_cycle = cycle
                best_len = len(cycle)
                
        if best_cycle:
            cycles.append(best_cycle)
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
        else:
            break
            
    return cycles

def construct():
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    
    time_per_graph = (_DEADLINE_SECONDS - 10) / 3.0
    
    for idx, (n, p, seed) in enumerate(specs):
        elapsed = time.time() - start_time
        remaining_time = _DEADLINE_SECONDS - elapsed
        if remaining_time < 5:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        allowed_time = min(time_per_graph * 0.8, remaining_time - 5)
        
        rng = random.Random(42 + idx)
        
        best_pieces = None
        best_score = float('inf')
        
        # Run multiple restarts
        while (time.time() - start_time) < start_time + allowed_time:
            cycles = _greedy_longest_decompose(edges, n, rng)
            
            pieces = []
            covered_edges = set()
            
            for cycle in cycles:
                pieces.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    covered_edges.add(e)
                    
            for u, v in edges:
                e = (min(u, v), max(u, v))
                if e not in covered_edges:
                    pieces.append([u, v])
                    
            score = len(pieces)
            if score < best_score:
                best_score = score
                best_pieces = pieces
                
        if best_pieces is None:
            best_pieces = [[u, v] for u, v in edges]
            
        results.append(best_pieces)
        
    return results
