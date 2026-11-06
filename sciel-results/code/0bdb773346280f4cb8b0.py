import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        # Build adjacency as dict of sets for dynamic removal
        adj = {}
        for u, v in edges:
            if u not in adj: adj[u] = set()
            if v not in adj: adj[v] = set()
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        
        def deg(v):
            return len(adj.get(v, set()))
            
        def remove_path(path):
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                adj[u].remove(v)
                adj[v].remove(u)
                if not adj[u]: del adj[u]
                if not adj[v]: del adj[v]

        def beam_search_longest_path(start, beam_width=80, max_depth=30):
            """Beam search for longest simple path starting from start."""
            beam = [([start], {start})]
            best_path = [start]
            best_len = 1
            
            for _ in range(max_depth):
                if time.time() > deadline - 5.0:
                    break
                new_beam = []
                for path, visited in beam:
                    current = path[-1]
                    neighbors = [n for n in adj.get(current, set()) if n not in visited]
                    if not neighbors:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_path = path
                        continue
                    neighbors.sort(key=lambda x: (deg(x), x))
                    # Limit branching per beam entry
                    max_branch = max(1, beam_width // max(1, len(beam)))
                    for n in neighbors[:max_branch]:
                        new_path = path + [n]
                        new_visited = visited | {n}
                        new_beam.append((new_path, new_visited))
                        if len(new_path) > best_len:
                            best_len = len(new_path)
                            best_path = new_path
                if not new_beam:
                    break
                new_beam.sort(key=lambda x: -len(x[0]))
                beam = new_beam[:beam_width]
                if not beam:
                    break
            return best_path

        while adj:
            if time.time() > deadline - 2.0:
                # Fallback: greedy finish
                while adj:
                    u = next(iter(adj))
                    v = next(iter(adj[u]))
                    paths.append([u, v])
                    remove_path([u, v])
                break

            candidates = list(adj.keys())
            candidates.sort(key=lambda v: (deg(v) % 2 == 0, -deg(v)))
            
            best_path = None
            best_len = -1
            
            num_starts = min(len(candidates), 8)
            for start in candidates[:num_starts]:
                p = beam_search_longest_path(start)
                if len(p) > best_len:
                    best_len = len(p)
                    best_path = p
                    
            if best_path is None or len(best_path) < 2:
                u = candidates[0]
                v = next(iter(adj[u]))
                best_path = [u, v]
            
            paths.append(best_path)
            remove_path(best_path)
            
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
    return results
# EVOLVE-BLOCK-END
