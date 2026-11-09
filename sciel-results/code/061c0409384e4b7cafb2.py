import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before the 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_best_cycle(adj, rng):
        """
        Finds a good cycle in the graph.
        Returns a cycle (list of vertices) or None.
        """
        candidates = [u for u, neighbors in adj.items() if len(neighbors) > 1]
        if not candidates:
            return None

        best_cycle = None
        best_score = -1
        
        # Sort by degree to bias towards hubs, but add randomness
        candidates.sort(key=lambda x: len(adj[x]), reverse=True)
        
        # Pick top 5 and some random ones
        starts = candidates[:5]
        if len(candidates) > 5:
            starts += rng.sample(candidates, min(5, len(candidates)))
        
        starts = list(set(starts))
        rng.shuffle(starts)

        # We will collect a few candidate cycles and pick the best one
        candidate_cycles = []

        for start in starts:
            # Iterative randomized DFS
            color = {start: 1}
            parent = {start: -1}
            path = [start]
            path_index = {start: 0}
            
            def get_shuffled_iter(neighs):
                lst = list(neighs)
                rng.shuffle(lst)
                return iter(lst)
            
            stack = [(start, get_shuffled_iter(adj[start]))]
            
            while stack:
                u, it = stack[-1]
                found_next = False
                for w in it:
                    if w == parent[u]:
                        continue
                    if color.get(w, 0) == 0:
                        color[w] = 1
                        parent[w] = u
                        path.append(w)
                        path_index[w] = len(path) - 1
                        stack.append((w, get_shuffled_iter(adj[w])))
                        found_next = True
                        break
                    elif color.get(w, 0) == 1:
                        # Back edge found
                        if w in path_index:
                            idx = path_index[w]
                            cyc = path[idx:]
                            if len(cyc) >= 3:
                                candidate_cycles.append(cyc)
                
                if not found_next:
                    stack.pop()
                    color[u] = 2
                    if path and path[-1] == u:
                        path.pop()
                        if u in path_index:
                            del path_index[u]
            
            # Early stop if we found a very long cycle
            if candidate_cycles:
                max_len = max(len(c) for c in candidate_cycles)
                if max_len > 15: # Threshold for "good enough"
                    break

        if not candidate_cycles:
            return None

        # Score the candidates
        # Heuristic: Longer is better, but also prefer cycles that include degree-2 vertices
        # This helps "resolve" constrained vertices.
        best_c = None
        best_s = -1
        for c in candidate_cycles:
            score = len(c)
            # Count degree 2 vertices in the current adj
            deg2_count = sum(1 for v in c if len(adj[v]) == 2)
            score += 0.5 * deg2_count
            if score > best_s:
                best_s = score
                best_c = c

        return best_c

    def decompose(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        best_pieces = None
        best_count = float('inf')
        
        max_restarts = 800 # Increased restarts, assuming faster per-restart or better quality
        
        for i in range(max_restarts):
            if time.time() - start_time > DEADLINE:
                break
                
            rng = random.Random(i)
            # Deep copy adj for this restart
            current_adj = {u: set(v) for u, v in adj.items()}
            current_edges = set(edges)
            pieces = []
            
            while True:
                if time.time() - start_time > DEADLINE:
                    break
                
                # Check if any edges remain
                if not any(len(neigh) > 0 for neigh in current_adj.values()):
                    break
                
                cycle = find_best_cycle(current_adj, rng)
                if cycle is None:
                    break
                
                # Remove cycle edges
                for j in range(len(cycle)):
                    a, b = cycle[j], cycle[(j + 1) % len(cycle)]
                    e = (min(a, b), max(a, b))
                    current_edges.discard(e)
                    if b in current_adj[a]:
                        current_adj[a].discard(b)
                    if a in current_adj[b]:
                        current_adj[b].discard(a)
                
                pieces.append(list(cycle))
            
            # Add remaining edges as single pieces
            for e in current_edges:
                pieces.append([e[0], e[1]])
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [list(p) for p in pieces]
                
        if best_pieces is None:
            return [[u, v] for u, v in edges]
            
        return best_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results
