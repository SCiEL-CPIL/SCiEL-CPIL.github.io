"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 138  # safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose_graph(edges, n):
        """Decompose graph into minimum pieces (cycles + edges)."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []
        
        best_pieces = None
        best_count = len(edge_set) + 1
        
        # Try multiple randomized restarts
        max_restarts = 300
        rng = random.Random(42)
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Build adjacency list
            adj = defaultdict(set)
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            
            remaining = set(edge_set)
            pieces = []
            
            # Iteratively extract cycles
            while remaining:
                # Find a cycle in the remaining graph using randomized DFS
                # Try multiple strategies to find long cycles
                
                cycle = None
                
                # Strategy 1: Randomized DFS with bias toward unvisited nodes
                # Try from a high-degree vertex
                degs = {u: len([nb for nb in adj[u] if (min(u, nb), max(u, nb)) in remaining]) for u in adj}
                if degs:
                    # Sort vertices by degree descending, with some randomization
                    verts = sorted(degs.keys(), key=lambda u: (-degs[u], rng.random()))
                    
                    for start in verts[:min(10, len(verts))]:  # Try top 10 vertices
                        if degs[start] < 2:
                            continue
                        
                        # Randomized DFS to find a cycle back to start
                        c = _find_cycle_dfs(start, adj, remaining, rng, max_depth=80, attempts=15)
                        if c and (cycle is None or len(c) > len(cycle)):
                            cycle = c
                        
                        if cycle and len(cycle) >= 6:
                            break  # Good enough for this restart step
                
                # Strategy 2: Greedy approach
                c2 = _find_cycle_greedy(adj, remaining, rng)
                if c2 and (cycle is None or len(c2) > len(cycle)):
                    cycle = c2
                
                if cycle and len(cycle) >= 3:
                    # Remove cycle edges
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        remaining.discard(e)
                    pieces.append(cycle)
                else:
                    # No cycle found; take a single edge
                    e = tuple(sorted(rng.choice(list(remaining))))
                    remaining.discard(e)
                    pieces.append(list(e))
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [p[:] for p in pieces]
                
                # If we've reached the theoretical minimum (all edges in cycles, no single edges), stop
                if best_count == 1:
                    break
        
        return best_pieces if best_pieces else [list(e) for e in edge_set]
    
    def _find_cycle_dfs(start, adj, remaining, rng, max_depth=80, attempts=10):
        """Find a cycle through start using randomized DFS."""
        best_cycle = None
        
        for _ in range(attempts):
            path = [start]
            visited = {start}
            current = start
            found = False
            
            for _ in range(max_depth):
                neighbors = [nb for nb in adj[current] if (min(current, nb), max(current, nb)) in remaining]
                if not neighbors:
                    break
                
                unvisited = [nb for nb in neighbors if nb not in visited]
                
                if unvisited:
                    # Randomly pick an unvisited neighbor
                    next_node = rng.choice(unvisited)
                elif start in neighbors:
                    # Close the cycle
                    next_node = start
                    found = True
                else:
                    break
                
                if found:
                    if len(path) >= 3:
                        if best_cycle is None or len(path) > len(best_cycle):
                            best_cycle = path[:]
                    break
                
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if found and len(path) >= 3:
                if best_cycle is None or len(path) > len(best_cycle):
                    best_cycle = path[:]
        
        return best_cycle
    
    def _find_cycle_greedy(adj, remaining, rng):
        """Find a long cycle using greedy approach with backtracking."""
        # Find a good starting vertex (high degree)
        degs = {u: len([nb for nb in adj[u] if (min(u, nb), max(u, nb)) in remaining]) for u in adj}
        if not degs:
            return None
        
        # Try several starting vertices
        start_candidates = sorted(degs.keys(), key=lambda u: (-degs[u], rng.random()))
        
        best_cycle = None
        
        for start in start_candidates[:min(5, len(start_candidates))]:
            if degs[start] < 2:
                continue
            
            # Try to build a long cycle from this start
            for _ in range(5):  # Multiple attempts
                path = [start]
                visited = {start}
                current = start
                found = False
                
                for _ in range(100):  # Max depth
                    neighbors = [nb for nb in adj[current] if (min(current, nb), max(current, nb)) in remaining]
                    if not neighbors:
                        break
                    
                    unvisited = [nb for nb in neighbors if nb not in visited]
                    
                    if unvisited:
                        # Prefer neighbors with higher degree for longer cycles
                        unvisited_sorted = sorted(unvisited, key=lambda nb: (-degs.get(nb, 0), rng.random()))
                        next_node = unvisited_sorted[0]
                    elif start in neighbors and len(path) >= 3:
                        next_node = start
                        found = True
                    else:
                        break
                    
                    if found:
                        if len(path) >= 3:
                            if best_cycle is None or len(path) > len(best_cycle):
                                best_cycle = path[:]
                        break
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                
                if found and len(path) >= 3:
                    if best_cycle is None or len(path) > len(best_cycle):
                        best_cycle = path[:]
        
        return best_cycle
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose_graph(edges, n)
        results.append(pieces)
        if time.time() > deadline - 3:
            # Fallback for remaining graphs
            for spec2 in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))[len(results):]:
                n2, p2, seed2 = spec2
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
# EVOLVE-BLOCK-END
