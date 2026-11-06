"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        remaining_edges = set(edges)
        
        # Helper to find a long cycle using greedy path extension
        def find_greedy_cycle(start):
            if start not in adj or len(adj[start]) == 0:
                return []
            
            visited = {start}
            path = [start]
            current = start
            
            while True:
                # Find neighbors of current not in visited
                nexts = [n for n in adj[current] if n not in visited]
                
                if nexts:
                    # Greedy: pick a neighbor with highest degree to keep options open
                    nexts.sort(key=lambda x: len(adj[x]), reverse=True)
                    next_node = nexts[0]
                    visited.add(next_node)
                    path.append(next_node)
                    current = next_node
                else:
                    # Stuck. Check if current connects to any node in path
                    # to form a cycle
                    cycle_found = None
                    for i, node in enumerate(path):
                        if current in adj[node]:
                            # Cycle is path[i:] + [current]
                            # Length is len(path) - i + 1
                            # We want the longest cycle, so we want the smallest i
                            # Actually, the cycle is path[i] -> ... -> current -> path[i]
                            # The vertices are path[i], path[i+1], ..., current
                            # The edges are (path[i], path[i+1]), ..., (current, path[i])
                            # This is a valid cycle if len(path) - i >= 2
                            if len(path) - i >= 2:
                                cycle_found = path[i:] + [node]
                                break
                            
                    if cycle_found:
                        return cycle_found
                    else:
                        return []

        while remaining_edges:
            if time.time() > deadline:
                # Fallback: add all remaining edges as singletons
                pieces.extend([list(e) for e in remaining_edges])
                break
            
            best_cycle = []
            best_len = 0
            
            # Try starting from each vertex with degree > 1
            candidates = [v for v in adj if len(adj[v]) > 0]
            # Randomize order to break ties and find different cycles
            random.shuffle(candidates)
            
            for start in candidates:
                cycle = find_greedy_cycle(start)
                if len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                # Early exit if we found a very long cycle?
                # Max possible is N. If we find N, it's a Hamiltonian cycle.
                if best_len >= len(adj):
                    break
            
            if best_cycle:
                pieces.append(best_cycle)
                # Remove edges from best_cycle
                for i in range(len(best_cycle) - 1):
                    u, v = best_cycle[i], best_cycle[i+1]
                    edge = tuple(sorted((u, v)))
                    if edge in remaining_edges:
                        remaining_edges.remove(edge)
                        # Update adj
                        adj[u].remove(v)
                        adj[v].remove(u)
            else:
                # No cycle found, all remaining edges are singletons
                # This happens if the graph is a forest
                pieces.extend([list(e) for e in remaining_edges])
                break
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        decomposition = decompose(edges)
        results.append(decomposition)
        
        if time.time() > deadline:
            # If we run out of time, fill rest with empty lists (invalid) or best effort
            # But we should have finished.
            pass

    return results
# EVOLVE-BLOCK-END
