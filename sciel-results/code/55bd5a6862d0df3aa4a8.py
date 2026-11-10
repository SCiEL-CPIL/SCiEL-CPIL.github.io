"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import random

def construct():
    deadline = time.time() + 138  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_2_core(edge_set, n):
        """Compute the 2-core of the graph (subgraph where all vertices have degree >= 2)."""
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Iteratively remove vertices with degree < 2
        in_core = set(range(n))
        changed = True
        while changed:
            changed = False
            for u in list(in_core):
                deg = sum(1 for nb in adj[u] if nb in in_core)
                if deg < 2:
                    in_core.discard(u)
                    changed = True
        
        # Return only edges whose both endpoints are in the core
        core_edges = set()
        for u, v in edge_set:
            if u in in_core and v in in_core:
                core_edges.add(tuple(sorted((u, v))))
        return core_edges, in_core

    def find_longest_cycle(start, adj, remaining, max_depth=60):
        """Find a long cycle starting and ending at 'start' using randomized DFS."""
        best_cycle = None
        
        for attempt in range(30):
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
                    # Prefer neighbors with higher degree in remaining graph
                    def degree_in_remaining(nb):
                        return sum(1 for x in adj[nb] if (min(nb, x), max(nb, x)) in remaining)
                    
                    unvisited.sort(key=degree_in_remaining, reverse=True)
                    # Mix of greedy and random
                    if random.random() < 0.25:
                        next_node = random.choice(unvisited)
                    else:
                        next_node = unvisited[0]
                elif start in neighbors:
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

    def decompose_graph(edges, n):
        """Decompose graph into minimum pieces (cycles + edges) using 2-core guided search."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []
        
        best_pieces = None
        best_count = len(edge_set) + 1
        
        # Multiple restarts with different random seeds
        for restart in range(300):
            if time.time() > deadline - 5:
                break
            
            remaining = set(edge_set)
            pieces = []
            
            while remaining:
                # Get 2-core of remaining graph to focus on cycle-supporting subgraph
                core_edges, core_vertices = get_2_core(remaining, n)
                
                if not core_edges:
                    # No cycles possible, take all remaining edges as single edges
                    for e in remaining:
                        pieces.append(list(e))
                    break
                
                # Build adjacency for the core
                adj = defaultdict(set)
                for u, v in core_edges:
                    adj[u].add(v)
                    adj[v].add(u)
                
                # Pick a start vertex: prefer high-degree vertices in the core
                candidates = list(core_vertices)
                if not candidates:
                    candidates = list(set(u for e in remaining for u in e))
                
                def core_degree(u):
                    return sum(1 for nb in adj.get(u, []) if nb in core_vertices)
                
                candidates.sort(key=core_degree, reverse=True)
                
                # Try several start vertices to find the best cycle
                best_cycle = None
                best_start = None
                
                num_starts = min(5, len(candidates))
                for i in range(num_starts):
                    start = candidates[i]
                    cycle = find_longest_cycle(start, adj, remaining)
                    if cycle and (best_cycle is None or len(cycle) > len(best_cycle)):
                        best_cycle = cycle
                        best_start = start
                
                if best_cycle and len(best_cycle) >= 3:
                    # Remove cycle edges from remaining
                    for i in range(len(best_cycle)):
                        u, v = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
                        e = tuple(sorted((u, v)))
                        remaining.discard(e)
                    pieces.append(best_cycle)
                else:
                    # No cycle found; take one edge
                    e = tuple(sorted(random.choice(list(remaining))))
                    remaining.discard(e)
                    pieces.append(list(e))
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [p[:] for p in pieces]
                
                # Early termination if we've reached a good solution
                # Theoretical lower bound: edges / max_cycle_length
                if best_count <= len(edge_set) // 5 + 2:
                    break
        
        return best_pieces if best_pieces else [list(e) for e in edge_set]

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for idx, spec in enumerate(specs):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose_graph(edges, n)
        results.append(pieces)
        
        if time.time() > deadline - 3:
            # Fallback for remaining graphs
            for spec2 in specs[idx+1:]:
                n2, p2, seed2 = spec2
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
# EVOLVE-BLOCK-END
