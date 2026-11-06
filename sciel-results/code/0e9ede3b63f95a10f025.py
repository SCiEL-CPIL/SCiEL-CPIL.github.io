"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_long_cycle(adj, n, rng, max_steps=500000):
    """Find a long cycle using randomized DFS with backtracking."""
    best_cycle = None
    for attempt in range(20):
        # Pick a random start vertex with degree >= 2
        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        if not candidates:
            break
        start = rng.choice(candidates)
        
        # DFS to find a cycle
        path = [start]
        visited = {start}
        stack = [(start, list(adj[start]))]
        found = False
        steps = 0
        
        while stack and steps < max_steps:
            steps += 1
            node, neighbors = stack[-1]
            
            # Find an unvisited neighbor
            next_node = None
            for nb in neighbors:
                if nb not in visited:
                    next_node = nb
                    break
            
            if next_node is None:
                # Backtrack
                stack.pop()
                if len(stack) > 0:
                    # Remove the edge from parent's adjacency list temporarily
                    parent = stack[-1][0]
                    # We don't actually modify adj, just track visited
                    pass
                continue
            
            # Check if we can close a cycle
            if next_node == start and len(path) >= 3:
                found = True
                break
            
            # Move to next node
            path.append(next_node)
            visited.add(next_node)
            
            # Get neighbors of next_node
            next_neighbors = list(adj[next_node])
            # Shuffle for randomization
            rng.shuffle(next_neighbors)
            stack.append((next_node, next_neighbors))
        
        if found:
            cycle = path[:]
            if best_cycle is None or len(cycle) > len(best_cycle):
                best_cycle = cycle
    
    return best_cycle

def _decompose(edges, n, deadline):
    """Decompose edges into cycles and single edges, minimizing total pieces."""
    if not edges:
        return []
    
    # Build adjacency list
    adj = defaultdict(set)
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
    
    pieces = []
    remaining_edges = set(edge_set)
    remaining_adj = {v: set() for v in adj}
    for u, v in edges:
        remaining_adj[u].add(v)
        remaining_adj[v].add(u)
    
    # Greedy: repeatedly extract longest cycle
    while remaining_edges:
        if time.time() > deadline:
            break
        
        # Find longest cycle
        best_cycle = None
        for _ in range(5):  # Try a few random starts
            cycle = _find_long_cycle(remaining_adj, n, random.Random(), max_steps=200000)
            if cycle and (best_cycle is None or len(cycle) > len(best_cycle)):
                best_cycle = cycle
        
        if best_cycle and len(best_cycle) >= 3:
            pieces.append(best_cycle)
            # Remove cycle edges
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                e = (min(u, v), max(u, v))
                if e in remaining_edges:
                    remaining_edges.remove(e)
                    remaining_adj[u].discard(v)
                    remaining_adj[v].discard(u)
        else:
            # No cycle found, take a single edge
            u, v = next(iter(remaining_edges))
            pieces.append([u, v])
            remaining_edges.remove((u, v))
            remaining_adj[u].discard(v)
            remaining_adj[v].discard(u)
    
    return pieces

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = _instance(*spec)
        n = spec[0]
        pieces = _decompose(edges, n, deadline)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
