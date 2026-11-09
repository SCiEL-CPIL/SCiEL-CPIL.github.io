import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _find_longest_cycle_dfs(adj, n, start, max_depth=100):
        """Find the longest cycle through start using DFS with backtracking."""
        best_cycle = None
        best_len = 0
        
        visited = {start}
        path = [start]
        stack = [(start, list(adj[start]))]
        
        while stack:
            node, neighbors = stack[-1]
            if not neighbors:
                stack.pop()
                path.pop()
                continue
            nxt = neighbors.pop()
            if nxt == start and len(path) >= 3:
                if len(path) > best_len:
                    best_len = len(path)
                    best_cycle = list(path)
                break
            if nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                stack.append((nxt, list(adj[nxt])))
                if len(path) > max_depth:
                    stack.pop()
                    path.pop()
        
        return best_cycle

    def _max_cycle_packing(edges, n, rng):
        """
        Find a set of edge-disjoint cycles using iterative longest-cycle extraction.
        """
        if not edges:
            return [], set()
        
        cycles = []
        current_edges = set(edges)
        adj = defaultdict(set)
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Iteratively extract cycles
        max_iterations = 100
        for _ in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break
            if not current_edges:
                break
            
            # Find vertices with degree >= 2
            vertices_with_degree = [v for v in adj if len(adj[v]) >= 2]
            if not vertices_with_degree:
                break
            
            # Try to find the longest cycle
            best_cycle = None
            best_len = 0
            
            # Try multiple starting vertices (shuffled for diversity)
            shuffled_vertices = vertices_with_degree[:]
            rng.shuffle(shuffled_vertices)
            
            for start in shuffled_vertices[:30]:
                cycle = _find_longest_cycle_dfs(adj, n, start)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
            
            if best_cycle:
                cycles.append(best_cycle)
                # Remove edges from current_edges
                for i in range(len(best_cycle)):
                    u, v = best_cycle[i], best_cycle[(i+1) % len(best_cycle)]
                    e = (min(u,v), max(u,v))
                    current_edges.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
            else:
                break
        
        return cycles, current_edges

    def _decompose(edges, n):
        """
        Decompose using iterative longest-cycle extraction with multiple restarts.
        """
        if not edges:
            return []
        
        best_pieces = None
        best_score = float('inf')
        
        # Try multiple random orderings
        num_restarts = 40
        
        for r in range(num_restarts):
            if time.time() - start_time > DEADLINE:
                break
            
            rng = random.Random(1000 + r)
            
            cycles, remaining = _max_cycle_packing(edges, n, rng)
            
            score = len(cycles) + len(remaining)
            
            if score < best_score:
                best_score = score
                best_pieces = []
                for c in cycles:
                    best_pieces.append(list(c))
                for e in remaining:
                    best_pieces.append([e[0], e[1]])
        
        if best_pieces is None:
            return [[u, v] for u, v in edges]
            
        return best_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        results.append(_decompose(edges, n))
        
        # Check time
        if time.time() - start_time > DEADLINE:
            break

    # Ensure we return 3 lists
    while len(results) < 3:
        results.append([])

    return results
