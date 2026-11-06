"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
import time
import random
from collections import defaultdict

_DEADLINE_SECONDS = 140.0  # Safety margin from 150s limit

def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycle_greedy(adj, deg, n, start, time_limit):
    """
    Find a cycle starting from 'start' using a greedy DFS that prefers
    unvisited neighbors with high degree. Returns a list of vertices or None.
    """
    start_time = time.time()
    visited = {start}
    path = [start]
    parent = {start: None}
    
    # Stack for DFS: (node, iterator_over_neighbors)
    # We simulate recursion to avoid stack limits and allow time checks
    # But for a single path, a simple loop is often faster if we don't backtrack much.
    # However, to find a cycle, we need to return to start.
    # Let's use a standard DFS with backtracking but limited depth/time.
    
    # Actually, a simpler approach: Randomized DFS with backtracking.
    # We want to find a path from start to start of length >= 3.
    
    # Let's try a heuristic: 
    # 1. Start at 'start'.
    # 2. Pick a neighbor.
    # 3. Continue to unvisited neighbors.
    # 4. If stuck, backtrack.
    # 5. If we hit 'start' with len(path) >= 3, we found a cycle.
    
    # To make it efficient, we limit the number of backtracks or use a greedy choice.
    # Given the small size (n<=30), we can afford some backtracking.
    
    def dfs(u, path, visited):
        if time.time() - start_time > time_limit:
            return None
        
        if u == start and len(path) >= 3:
            return path[:]
        
        neighbors = list(adj[u])
        # Sort neighbors: prefer unvisited, then high degree
        neighbors.sort(key=lambda x: (x in visited, -deg[x]))
        
        for v in neighbors:
            if v in visited:
                continue
            visited.add(v)
            path.append(v)
            result = dfs(v, path, visited)
            if result:
                return result
            path.pop()
            visited.remove(v)
            
        return None

    return dfs(start, path, visited)

def _decompose_graph(edges, n, time_limit):
    """
    Decompose graph by repeatedly finding and removing cycles.
    """
    if not edges:
        return []

    adj = defaultdict(set)
    deg = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    pieces = []
    start_time = time.time()
    rng = random.Random(42)

    while True:
        if time.time() - start_time > time_limit:
            break

        # Check remaining edges
        remaining = sum(len(adj[u]) for u in range(n)) // 2
        if remaining == 0:
            break

        # Find a cycle
        cycle = None
        candidates = [v for v in range(n) if deg[v] >= 2]
        
        if not candidates:
            # No cycles left, take an edge
            edge_found = None
            for u in range(n):
                if adj[u]:
                    v = next(iter(adj[u]))
                    edge_found = (u, v)
                    break
            if edge_found is None:
                break
            u, v = edge_found
            pieces.append([u, v])
            adj[u].discard(v)
            adj[v].discard(u)
            deg[u] -= 1
            deg[v] -= 1
            continue

        # Try to find a cycle from a few random candidates
        # Prioritize high degree vertices
        candidates.sort(key=lambda v: -deg[v])
        
        # Try top few candidates
        found = False
        for start in candidates[:5]:
            if time.time() - start_time > time_limit:
                break
                
            # Allow a small time budget per start attempt
            cycle = _find_cycle_greedy(adj, deg, n, start, 0.05)
            if cycle:
                found = True
                break
        
        if found and cycle:
            pieces.append(cycle)
            # Remove cycle edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                adj[u].discard(v)
                adj[v].discard(u)
                deg[u] -= 1
                deg[v] -= 1
        else:
            # No cycle found, take an edge
            edge_found = None
            for u in range(n):
                if adj[u]:
                    v = next(iter(adj[u]))
                    edge_found = (u, v)
                    break
            if edge_found is None:
                break
            u, v = edge_found
            pieces.append([u, v])
            adj[u].discard(v)
            adj[v].discard(u)
            deg[u] -= 1
            deg[v] -= 1

    # Add remaining edges
    for u in range(n):
        for v in adj[u]:
            if u < v:
                pieces.append([u, v])

    return pieces

def _merge_edges_into_cycles(pieces, n, time_limit):
    """
    Post-processing: Try to merge single-edge pieces into cycles.
    Specifically, look for triangles formed by 3 single edges.
    """
    start_time = time.time()
    
    # Identify single edge pieces
    edge_pieces = {} # (u,v) -> index
    for i, piece in enumerate(pieces):
        if len(piece) == 2:
            u, v = piece
            edge_pieces[(min(u, v), max(u, v))] = i

    if not edge_pieces:
        return pieces

    # Build adjacency of edge pieces
    adj_single = defaultdict(set)
    for (u, v) in edge_pieces.keys():
        adj_single[u].add(v)
        adj_single[v].add(u)
        
    deg_single = [len(adj_single[v]) for v in range(n)]

    used_pieces = set()
    new_cycles = []
    
    changed = True
    while changed:
        if time.time() - start_time > time_limit:
            break
        changed = False
        
        # Find triangles in the single-edge graph
        for u in range(n):
            if deg_single[u] < 2:
                continue
            
            neighbors = list(adj_single[u])
            for i in range(len(neighbors)):
                v = neighbors[i]
                if (min(u, v), max(u, v)) in used_pieces:
                    continue
                for j in range(i + 1, len(neighbors)):
                    w = neighbors[j]
                    if (min(u, w), max(u, w)) in used_pieces:
                        continue
                    # Check if (v, w) is an edge piece
                    edge_vw = (min(v, w), max(v, w))
                    if edge_vw in edge_pieces:
                        if edge_vw in used_pieces:
                            continue
                        
                        # Found triangle u, v, w
                        e_uv = (min(u, v), max(u, v))
                        e_uw = (min(u, w), max(u, w))
                        
                        idx_uv = edge_pieces.get(e_uv)
                        idx_uw = edge_pieces.get(e_uw)
                        idx_vw = edge_pieces.get(edge_vw)
                        
                        if idx_uv is not None and idx_uw is not None and idx_vw is not None:
                            if idx_uv not in used_pieces and idx_uw not in used_pieces and idx_vw not in used_pieces:
                                # Merge them
                                used_pieces.add(idx_uv)
                                used_pieces.add(idx_uw)
                                used_pieces.add(idx_vw)
                                new_cycles.append([u, v, w])
                                changed = True
                                break
                if changed:
                    break
            if changed:
                break

    if not changed:
        return pieces

    # Rebuild pieces
    final_pieces = []
    for i, piece in enumerate(pieces):
        if len(piece) > 2:
            final_pieces.append(piece)
        elif len(piece) == 2:
            u, v = piece
            edge = (min(u, v), max(u, v))
            idx = edge_pieces.get(edge)
            if idx is not None and idx not in used_pieces:
                final_pieces.append(piece)
        
    final_pieces.extend(new_cycles)
    
    return final_pieces

def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining = _DEADLINE_SECONDS - elapsed
        if remaining < 10:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        edges_set = set((min(u, v), max(u, v)) for u, v in edges)

        # Allocate time for this graph
        graph_time = remaining * 0.9

        # Strategy: Greedy Cycle Removal
        pieces = _decompose_graph(edges, n, graph_time * 0.8)
        
        # Post-processing: Merge edge pieces into cycles
        pieces = _merge_edges_into_cycles(pieces, n, graph_time * 0.2)

        results.append(pieces)

    return results
