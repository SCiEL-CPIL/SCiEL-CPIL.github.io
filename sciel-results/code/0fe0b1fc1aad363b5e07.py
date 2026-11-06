"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_longest_path(remaining_edges, adj, max_steps=500000):
        """Beam search for a long simple path using only remaining edges."""
        if not remaining_edges:
            return []
        
        # Build adjacency from remaining edges
        rem_adj = defaultdict(set)
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Get a starting vertex (highest degree in remaining graph)
        if not rem_adj:
            return []
        
        start = max(rem_adj.keys(), key=lambda x: len(rem_adj[x]))
        
        # Beam search: each state is (path_tuple, visited_frozenset)
        # We keep top-K paths by length
        beam_size = 200
        beam = [(tuple([start]), frozenset([start]))]
        
        best_path = [start]
        
        steps = 0
        while steps < max_steps:
            steps += 1
            if time.time() - start_time > DEADLINE:
                break
            
            new_beam = []
            for path, visited in beam:
                last = path[-1]
                for nxt in rem_adj.get(last, []):
                    if nxt not in visited:
                        new_path = path + (nxt,)
                        new_visited = visited | {nxt}
                        new_beam.append((new_path, new_visited))
            
            if not new_beam:
                break
            
            # Keep only the longest paths (beam_size)
            new_beam.sort(key=lambda x: len(x[0]), reverse=True)
            beam = new_beam[:beam_size]
            
            # Track best
            if len(beam[0][0]) > len(best_path):
                best_path = list(beam[0][0])
            
            # If beam has only one path and it can't extend, stop
            if len(beam) == 1:
                last = beam[0][0][-1]
                if not any(nxt not in beam[0][1] for nxt in rem_adj.get(last, [])):
                    break
        
        return best_path

    def decompose(edges):
        """Decompose edges into edge-disjoint simple paths, minimizing count."""
        if not edges:
            return []
        
        remaining = set(tuple(e) for e in edges)
        adj = build_adjacency(edges)
        paths = []
        
        while remaining:
            # Find longest path in remaining graph
            path = find_longest_path(remaining, adj)
            
            if not path or len(path) < 2:
                # Single isolated edge
                # Find any remaining edge
                u, v = next(iter(remaining))
                path = [u, v]
            
            # Remove edges used by this path
            for i in range(len(path) - 1):
                e1 = (path[i], path[i+1])
                e2 = (path[i+1], path[i])
                if e1 in remaining:
                    remaining.discard(e1)
                elif e2 in remaining:
                    remaining.discard(e2)
            
            paths.append(path)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() - start_time > DEADLINE:
            # Fallback: each edge as its own path
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
