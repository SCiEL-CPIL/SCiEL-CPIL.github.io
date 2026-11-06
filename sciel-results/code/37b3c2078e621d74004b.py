"""Rewritten program: decompose each instance graph into cycles + leftover edges, minimizing piece count."""
import time
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _find_cycle(adj, n, used_edges, start=None):
    """Find a cycle using DFS. Returns list of vertices in cycle order or None."""
    # Try to find a cycle starting from a vertex with enough unused degree
    candidates = [v for v in range(n) if len([u for u in adj[v] if (min(u,v),max(u,v)) not in used_edges]) >= 2]
    if start is not None and start in candidates:
        candidates = [start] + [c for c in candidates if c != start]
    if not candidates:
        return None
    
    for start_v in candidates:
        # DFS to find a cycle back to start_v
        stack = [(start_v, [start_v], set())]
        while stack:
            node, path, visited = stack.pop()
            for neighbor in adj[node]:
                edge_key = (min(node, neighbor), max(node, neighbor))
                if edge_key in used_edges:
                    continue
                if neighbor == start_v and len(path) >= 3:
                    return path
                if neighbor not in visited:
                    new_visited = visited | {node}
                    stack.append((neighbor, path + [neighbor], new_visited))
    return None

def _decompose_graph(n, edges):
    """Decompose graph into cycles and single edges. Returns list of pieces."""
    adj = _build_adjacency(n, edges)
    used_edges = set()
    for u, v in edges:
        used_edges.add((min(u, v), max(u, v)))
    
    remaining_edges = set(used_edges)
    pieces = []
    
    # Repeatedly find cycles
    while remaining_edges:
        # Find a cycle
        cycle = _find_cycle(adj, n, used_edges - remaining_edges)
        if cycle is None:
            break
        
        # Mark cycle edges as used
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge_key = (min(u, v), max(u, v))
            if edge_key in remaining_edges:
                remaining_edges.discard(edge_key)
            else:
                # This shouldn't happen if our cycle is valid
                cycle = None
                break
        
        if cycle is not None and len(cycle) >= 3:
            pieces.append(cycle)
    
    # Add remaining edges as single-edge pieces
    for u, v in sorted(remaining_edges):
        pieces.append([u, v])
    
    return pieces

def construct():
    deadline = time.time() + 145  # 5 second safety margin before 150s kill
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return trivial decomposition for remaining graphs
            n, p, seed = spec
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = _instance(*spec)
        n = spec[0]
        pieces = _decompose_graph(n, edges)
        results.append(pieces)
    
    return results
