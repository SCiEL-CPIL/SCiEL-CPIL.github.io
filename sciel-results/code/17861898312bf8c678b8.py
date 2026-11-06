import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(edges, n):
        if not edges:
            return []
        
        # Build adjacency list for vertices
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Precompute conflict sets for edges
        # Two edges e1, e2 conflict if they share a vertex or are connected by an edge.
        # e1=(u1,v1), e2=(u2,v2)
        # Conflict if {u1,v1} & {u2,v2} != empty
        # OR if (u1 in adj[u2] or u1 in adj[v2] or v1 in adj[u2] or v1 in adj[v2])
        
        edge_set = set(edges)
        conflicts = {}
        for e in edges:
            u1, v1 = e
            conf = set()
            for e2 in edges:
                if e == e2: continue
                u2, v2 = e2
                # Share vertex?
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conf.add(e2)
                # Connected by edge?
                elif (u1 in adj[u2] or u1 in adj[v2] or v1 in adj[u2] or v1 in adj[v2]):
                    conf.add(e2)
            conflicts[e] = conf
            
        def find_large_induced_matching(remaining_edges, rng, iterations=20):
            best_m = set()
            best_size = 0
            remaining_list = list(remaining_edges)
            for _ in range(iterations):
                rng.shuffle(remaining_list)
                m = set()
                for e in remaining_list:
                    # Check if e can be added to m
                    # e is compatible with m if no edge in m is in conflicts[e]
                    # Efficient check:
                    # if not any(f in conflicts[e] for f in m):
                    # This is O(|m|) per check, so O(|M|^2) total per iteration.
                    # |M| is small (max matching size <= n/2).
                    if not (m & conflicts[e]):
                        m.add(e)
                if len(m) > best_size:
                    best_size = len(m)
                    best_m = m
            return best_m

        colouring = {}
        remaining = set(edges)
        colour_id = 0
        rng = random.Random(42) # Fixed seed for reproducibility in heuristic, or use time? Fixed is safer for deterministic output if needed, but task implies one run.
        
        while remaining:
            if time.time() > deadline:
                # Fallback: just colour remaining edges individually
                for e in remaining:
                    colouring[e] = colour_id
                    colour_id += 1
                break
                
            # Find a large induced matching in remaining
            m = find_large_induced_matching(remaining, rng, iterations=50)
            
            if not m:
                # Should not happen if remaining is not empty, but safety
                e = next(iter(remaining))
                m = {e}
                
            for e in m:
                colouring[e] = colour_id
            remaining -= m
            colour_id += 1
            
        # Return colours in sorted edge order
        return [colouring[e] for e in edges]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        colours = solve_instance(edges, n)
        out.append(colours)
        
    return out

from random import Random
