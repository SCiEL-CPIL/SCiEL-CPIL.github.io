"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random as _random

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose_graph(edges):
    """Decompose a graph into cycles and single edges, minimizing piece count."""
    if not edges:
        return []
    
    # Build adjacency
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    
    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    deadline = time.time() + 140  # safety margin
    
    while remaining:
        if time.time() > deadline:
            # Fallback: emit remaining edges as single-edge pieces
            for e in remaining:
                pieces.append([e[0], e[1]])
            break
        
        # Find a cycle in the remaining graph using DFS
        cycle = _find_longest_cycle(remaining, adj)
        
        if cycle and len(cycle) >= 3:
            pieces.append(cycle)
            # Remove edges of the cycle from remaining
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                if e in remaining:
                    remaining.discard(e)
                    # Update adj
                    adj[u].discard(v)
                    adj[v].discard(u)
                    if not adj[u]:
                        del adj[u]
                    if not adj[v]:
                        del adj[v]
        else:
            # No cycle found; pick a high-degree vertex and emit its edges
            # as single-edge pieces, or find a 2-edge path to remove
            # Find a vertex with max degree in remaining
            deg = {}
            for u, v in remaining:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            if deg:
                max_deg_vertex = max(deg, key=lambda x: (deg[x], x))
                # Emit all edges incident to this vertex as single pieces
                to_remove = []
                for e in list(remaining):
                    if max_deg_vertex in e:
                        to_remove.append(e)
                        pieces.append([e[0], e[1]])
                        u, v = e
                        adj[u].discard(v)
                        adj[v].discard(u)
                        if not adj[u]:
                            adj[u] = adj.get(u, set())
                            if not adj[u]:
                                del adj[u]
                        if not adj[v]:
                            adj[v] = adj.get(v, set())
                            if not adj[v]:
                                del adj[v]
                for e in to_remove:
                    remaining.discard(e)
            else:
                # Should not happen, but safety
                e = next(iter(remaining))
                pieces.append([e[0], e[1]])
                remaining.discard(e)
    
    return pieces


def _find_longest_cycle(remaining, adj, max_nodes=200):
    """Find a long cycle in the graph defined by remaining edges and adj."""
    if not remaining:
        return None
    
    # Get vertices with degree >= 2
    candidates = [v for v in adj if len(adj[v]) >= 2]
    if not candidates:
        return None
    
    # Sort by degree descending for better cycles
    candidates.sort(key=lambda v: (-len(adj[v]), v))
    
    best_cycle = None
    
    for start in candidates[:30]:  # Try up to 30 starting vertices
        # DFS to find a cycle back to start
        stack = [(start, [start])]
        visited = {start}
        found = None
        
        while stack:
            node, path = stack.pop()
            if len(path) > max_nodes:
                continue
            
            neighbors = list(adj.get(node, []))
            # Shuffle for variety
            _random.shuffle(neighbors)
            
            for nb in neighbors:
                # Check edge exists in remaining
                e = tuple(sorted((node, nb)))
                if e not in remaining:
                    continue
                
                if nb == start and len(path) >= 3:
                    # Found a cycle
                    cycle = path
                    if best_cycle is None or len(cycle) > len(best_cycle):
                        best_cycle = cycle
                    found = cycle
                    break
                elif nb not in visited:
                    visited.add(nb)
                    stack.append((nb, path + [nb]))
            
            if found:
                break
        
        if best_cycle and len(best_cycle) >= 8:
            break  # Good enough
    
    return best_cycle


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        edges = _instance(*spec)
        pieces = _decompose_graph(edges)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
