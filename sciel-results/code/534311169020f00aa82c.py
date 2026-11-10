"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycle_from_start(start, adj, remaining, max_len=8):
    """Find a cycle starting from start using BFS/DFS, preferring longer cycles."""
    best_cycle = None
    best_len = 0
    
    # Try BFS to find shortest cycle first (fast), then try to extend
    # Actually, let's try DFS with limited depth to find good cycles
    visited = {start: [start]}
    stack = [(start, [start], 0)]
    
    while stack:
        node, path, depth = stack.pop()
        if depth >= max_len:
            continue
        
        for nb in adj[node]:
            if nb == start and len(path) >= 3:
                # Found a cycle
                cycle = path
                # Verify all edges are in remaining
                valid = True
                for i in range(len(cycle)):
                    a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((a, b)))
                    if e not in remaining:
                        valid = False
                        break
                if valid and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
            elif nb not in visited and nb != start:
                new_path = path + [nb]
                visited[nb] = new_path
                stack.append((nb, new_path, depth + 1))
    
    return best_cycle

def _decompose_graph(edges, deadline):
    """Decompose edges into cycles and single edges efficiently."""
    if not edges:
        return []
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    adj = defaultdict(set)
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    remaining = set(edges_set)
    
    # Phase 1: Find cycles greedily
    # Strategy: repeatedly find a cycle, preferring longer ones
    while remaining and time.time() < deadline:
        # Find vertices with degree >= 2, sorted by degree descending
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            break
        
        # Sort by degree descending to prefer high-degree vertices
        candidates.sort(key=lambda v: -len(adj[v]))
        
        best_cycle = None
        best_len = 0
        
        # Try each candidate start vertex
        for start in candidates[:30]:  # Limit to top 30 candidates
            if len(adj[start]) < 2:
                continue
            
            cycle = _find_cycle_from_start(start, adj, remaining, max_len=10)
            if cycle and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle
            
            # If we found a reasonably long cycle, stop searching
            if best_len >= 6:
                break
        
        if best_cycle and best_len >= 3:
            pieces.append(best_cycle)
            # Remove cycle edges from remaining and adj
            for i in range(len(best_cycle)):
                a, b = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
                e = tuple(sorted((a, b)))
                remaining.discard(e)
                adj[a].discard(b)
                adj[b].discard(a)
        else:
            break
    
    # Phase 2: Add remaining edges as single-edge pieces
    for e in remaining:
        pieces.append([e[0], e[1]])
    
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    # Evaluator kills at 150s, leave 10s margin
    deadline = time.time() + 140.0
    
    for i, spec in enumerate(specs):
        # Allocate time proportionally: ~45s per graph
        graph_deadline = deadline - (len(specs) - 1 - i) * 5.0  # Leave 5s per remaining graph
        
        edges = _instance(*spec)
        pieces = _decompose_graph(edges, graph_deadline)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
