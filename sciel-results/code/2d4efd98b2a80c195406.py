import time
import random
from random import Random

DEADLINE_MARGIN = 10  # seconds before 150s

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def build_adj(edges):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj

def cycle_edges(cycle):
    """Given a list of vertices forming a cycle (>=3 distinct), return set of edges."""
    if len(cycle) < 3:
        return set()
    edges = set()
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        e = (u, v) if u < v else (v, u)
        edges.add(e)
    return edges

def find_longest_cycle(remaining, adj, rng, max_iter=50000):
    """
    Find a long simple cycle in the graph using randomized DFS with backtracking.
    Returns the cycle as a list of vertices, or None if no cycle found.
    """
    if not remaining:
        return None
    
    nodes = list(set([u for (u, v) in remaining] + [v for (u, v) in remaining]))
    if len(nodes) < 3:
        return None
    
    best_cycle = None
    best_len = 0
    iter_count = 0
    
    # Try multiple random starts
    for _ in range(20):
        if iter_count > max_iter:
            break
        start_node = rng.choice(nodes)
        if start_node not in adj or len(adj[start_node]) < 2:
            continue
        
        # DFS with path tracking
        visited = {start_node}
        path = [start_node]
        stack = [(start_node, iter( sorted(adj[start_node], key=lambda x: rng.random()) ))]
        
        found = False
        while stack and iter_count < max_iter:
            node, neighbors = stack[-1]
            iter_count += 1
            
            advanced = False
            for nb in neighbors:
                e = (node, nb) if node < nb else (nb, node)
                if e not in remaining:
                    continue
                
                if nb == start_node and len(path) >= 3:
                    # Found a cycle
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                    found = True
                    break
                
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    stack.append((nb, iter(sorted(adj[nb], key=lambda x: rng.random()))))
                    advanced = True
                    break
            
            if found:
                break
            if not advanced:
                # Backtrack
                stack.pop()
                if path:
                    last = path.pop()
                    visited.discard(last)
    
    return best_cycle

def decompose_max_cycles(edges, max_time=40.0):
    """
    Decompose edges into fewest pieces by repeatedly extracting the longest cycle.
    Remaining edges (forest) become single-edge pieces.
    """
    start = time.time()
    all_edges_set = set(tuple(sorted(e)) for e in edges)
    remaining = set(all_edges_set)
    pieces = []
    rng = Random(42)
    
    while remaining:
        cur_time = time.time()
        if cur_time - start > max_time:
            break
        
        adj = build_adj(list(remaining))
        
        # Try to find a long cycle
        cycle = find_longest_cycle(remaining, adj, rng, max_iter=30000)
        
        if cycle and len(cycle) >= 3:
            cyc_edges = cycle_edges(cycle)
            # Verify all edges are in remaining
            if cyc_edges <= remaining:
                pieces.append(cycle[:])
                for e in cyc_edges:
                    remaining.discard(e)
            else:
                # Edge mismatch, take a single edge
                e = next(iter(remaining))
                pieces.append([e[0], e[1]])
                remaining.discard(e)
        else:
            # No cycle found; take a single edge
            if remaining:
                e = next(iter(remaining))
                pieces.append([e[0], e[1]])
                remaining.discard(e)
    
    # Add remaining edges as single-edge pieces
    for e in remaining:
        pieces.append([e[0], e[1]])
    
    return pieces

def construct():
    start = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    per_graph_time = max(1.0, (DEADLINE_MARGIN - 5) / 3)
    
    for spec in specs:
        edges = instance(*spec)
        cur_time = time.time()
        remaining_time = min(per_graph_time, DEADLINE_MARGIN - (cur_time - start))
        if remaining_time < 1.0:
            # No time left, return all edges as single pieces
            results.append([list(e) for e in edges])
            continue
        
        pieces = decompose_max_cycles(edges, max_time=remaining_time)
        results.append(pieces)
        
        # Check deadline
        cur_time = time.time()
        if cur_time - start > DEADLINE_MARGIN - 3:
            # Return remaining as individual edges
            for remaining_spec in specs[specs.index(spec) + 1:]:
                e = instance(*remaining_spec)
                results.append([list(e) for e in e])
            break
    
    # Ensure we have exactly 3 results
    while len(results) < 3:
        # Should not happen, but safeguard
        results.append([])
    
    return results[:3]
