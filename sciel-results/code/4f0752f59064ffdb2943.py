"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict
import random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_best_decomposition(edges, n, max_iters=200):
        """Find decomposition with minimum pieces using randomized greedy cycle extraction."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []
        
        best_pieces = None
        best_count = len(edge_set) + 1
        
        # Increase iterations to find better solutions, but guard with time
        for iteration in range(500):
            if time.time() > deadline - 10:
                break
            
            # Build adjacency
            adj = defaultdict(set)
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            
            remaining = set(edge_set)
            pieces = []
            
            # Greedy: repeatedly find a cycle in the remaining graph
            # Try to find longer cycles first by using BFS/DFS with depth preference
            while remaining:
                if not remaining:
                    break
                
                # Pick the vertex with the highest degree in the remaining graph
                # This is more likely to be part of a long cycle
                max_deg = -1
                start = None
                for u in adj:
                    deg = len([nb for nb in adj[u] if (min(u, nb), max(u, nb)) in remaining])
                    if deg > max_deg:
                        max_deg = deg
                        start = u
                
                if start is None or max_deg == 0:
                    # No edges left or isolated vertices, take a random edge
                    start_edge = random.choice(list(remaining))
                    e = tuple(sorted(start_edge))
                    remaining.discard(e)
                    pieces.append(list(e))
                    continue
                
                # Try to find a cycle starting from 'start'
                cycle = _find_cycle_greedy(start, adj, remaining)
                
                if cycle and len(cycle) >= 3:
                    # Remove cycle edges from remaining
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        remaining.discard(e)
                    pieces.append(cycle)
                else:
                    # No cycle found; take one edge as a piece
                    # Pick an edge incident to start if possible, else random
                    incident_edges = [e for e in remaining if start in e]
                    if incident_edges:
                        e = tuple(sorted(random.choice(incident_edges)))
                    else:
                        e = tuple(sorted(random.choice(list(remaining))))
                    remaining.discard(e)
                    pieces.append(list(e))
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [p[:] for p in pieces]
        
        return best_pieces if best_pieces else [list(e) for e in edge_set]
    
    def _find_cycle_greedy(start, adj, remaining):
        """Find a cycle starting at 'start' using randomized DFS with bias for longer cycles."""
        best_cycle = None
        
        # Try multiple random walks to find a cycle back to start
        # We want to find the longest possible cycle among these attempts
        for attempt in range(80):
            path = [start]
            visited = {start}
            current = start
            found = False
            
            for _ in range(150):  # max path length increased
                neighbors = [nb for nb in adj[current] if (min(current, nb), max(current, nb)) in remaining]
                if not neighbors:
                    break
                
                # Prefer unvisited neighbors for longer cycles
                unvisited = [nb for nb in neighbors if nb not in visited]
                
                if unvisited:
                    # Sort unvisited neighbors by degree (descending) to prefer dense areas
                    # This helps find longer cycles
                    unvisited.sort(key=lambda x: len(adj[x]), reverse=True)
                    # Randomize slightly to avoid getting stuck in local optima
                    if random.random() < 0.3:
                        next_node = random.choice(unvisited)
                    else:
                        next_node = unvisited[0]
                elif start in neighbors:
                    # Close the cycle only if we have no other options or if the cycle is already long
                    # If we have a choice, we might have missed a longer cycle by not exploring unvisited
                    # But if unvisited is empty, we must close or break.
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
        pieces = find_best_decomposition(edges, n)
        results.append(pieces)
        if time.time() > deadline - 5:
            # Fallback: just use individual edges for remaining graphs
            for spec2 in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))[len(results):]:
                n2, p2, seed2 = spec2
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
# EVOLVE-BLOCK-END
