"""Path decomposition via randomized greedy with Warnsdorff heuristic."""
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """
        Find edge-disjoint paths covering all edges, minimizing path count.
        Uses randomized greedy with Warnsdorff heuristic and multiple restarts.
        """
        edge_set = set(tuple(sorted(e)) for e in edges)
        n = max(max(e) for e in edge_set) + 1 if edge_set else 0
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        best_paths = None
        best_count = len(edge_set)  # worst case: each edge is its own path
        
        # Multiple restarts with different random seeds
        import random
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            rng = random.Random(attempt * 12345 + 678)
            
            # Copy edge set
            remaining = set(edge_set)
            paths = []
            
            # Greedy: repeatedly find a long path from remaining edges
            # Start from a random edge or vertex
            while remaining:
                # Pick a starting vertex: prefer one with high degree in remaining
                # Weighted random selection
                degs = {}
                for u, v in remaining:
                    degs[u] = degs.get(u, 0) + 1
                    degs[v] = degs.get(v, 0) + 1
                
                # Pick start vertex with some randomness but biased toward high degree
                vertices = list(degs.keys())
                weights = [degs[v] ** 2 for v in vertices]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                start = vertices[0]
                for i, v in enumerate(vertices):
                    cum += weights[i]
                    if r <= cum:
                        start = v
                        break
                
                # Build path using Warnsdorff: always go to neighbor with fewest unvisited edges
                path = [start]
                used_edges = set()
                
                while True:
                    # Find neighbors of last vertex via remaining edges
                    last = path[-1]
                    neighbors = []
                    for u, v in remaining:
                        if u == last and (u, v) not in used_edges:
                            neighbors.append(v)
                        elif v == last and (u, v) not in used_edges:
                            neighbors.append(u)
                    
                    if not neighbors:
                        break
                    
                    # Warnsdorff: choose neighbor with fewest available edges
                    def count_avail(nbr):
                        cnt = 0
                        for u, v in remaining:
                            if u == nbr and (u, v) not in used_edges:
                                cnt += 1
                            elif v == nbr and (u, v) not in used_edges:
                                cnt += 1
                        return cnt
                    
                    # Add some randomness: pick from top few with least available
                    scored = [(count_avail(nbr), nbr) for nbr in neighbors]
                    scored.sort()
                    
                    # Pick randomly among the best few (ties or close ties)
                    min_score = scored[0][0]
                    # Allow ties within a small range for diversity
                    candidates = [s for s in scored if s[0] <= min_score + 1]
                    chosen = rng.choice(candidates)[1]
                    
                    # Add edge to used
                    edge_tuple = tuple(sorted((last, chosen)))
                    used_edges.add(edge_tuple)
                    path.append(chosen)
                
                paths.append(path)
                
                # Remove used edges from remaining
                for e in used_edges:
                    if e in remaining:
                        remaining.discard(e)
            
            # Validate and count
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [list(p) for p in paths]
        
        if best_paths is None:
            # Fallback: each edge as its own path
            return [list(e) for e in edges]
        
        return best_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        if time.time() > deadline:
            # Emergency: fill remaining with single-edge paths
            while len(results) < 3:
                edges_left = instance(*results[len(results)]) if len(results) < len(((20,0.3,1),(25,0.3,2),(30,0.25,3))) else []
                results.append([list(e) for e in edges_left])
            break
    
    return results
