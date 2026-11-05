import random
from collections import defaultdict

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_path_greedy(remaining_edges, start, rng):
        """
        Find a long simple path starting from 'start' using a greedy heuristic.
        Uses rng for tie-breaking to allow variation.
        """
        if not remaining_edges:
            return []
        
        adj = defaultdict(set)
        for u, v in remaining_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        if start not in adj or not adj[start]:
            return []
            
        path = [start]
        visited = {start}
        current = start
        
        while True:
            neighbors = [n for n in adj[current] if n not in visited]
            if not neighbors:
                break
            
            # Heuristic: Prefer neighbors with lower degree (Warnsdorff's rule for Hamiltonian path)
            # to avoid getting stuck.
            # We add a bit of randomness by sorting with a small random noise or just shuffling ties.
            # Better: Sort by degree, then pick randomly from the top few?
            # Let's sort by degree ascending (prefer low degree) to extend path deeper.
            neighbors.sort(key=lambda x: len(adj[x]))
            
            # Randomize choice among top candidates to allow different paths
            # Pick randomly from the first 3 (or all if fewer)
            k = min(3, len(neighbors))
            next_node = rng.choice(neighbors[:k])
            
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            
        return path

    def decompose(edges):
        remaining = set(edges)
        paths = []
        rng = random.Random(42) # Fixed seed for reproducibility in attempts
        
        while remaining:
            # Build adjacency for candidates
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)
            
            candidates = [v for v in adj if adj[v]]
            if not candidates:
                break
                
            best_path = []
            
            # Try multiple starting vertices and multiple randomized walks
            # To save time, we can limit the number of attempts per step
            # N is small (<=30), edges ~ 100-200.
            # We can try all starts, but with 1-2 random variations each.
            
            for start in candidates:
                # Try 2 different random paths from this start
                for _ in range(2):
                    path = find_path_greedy(remaining, start, rng)
                    if len(path) > len(best_path):
                        best_path = path
                        
            if not best_path:
                # Fallback: pick any edge
                u, v = next(iter(remaining))
                best_path = [u, v]
                
            # Remove edges from remaining
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i+1]
                e = tuple(sorted((u, v)))
                remaining.discard(e)
            
            paths.append(best_path)
            
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    
    return results
