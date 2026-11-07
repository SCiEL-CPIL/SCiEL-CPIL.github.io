import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(a, b):
    return (a, b) if a < b else (b, a)

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin for 60s limit
    n = P['n']
    
    # Build adjacency list and edge set
    adj = [set() for _ in range(n)]
    full_edges = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        full_edges.add(_edge_key(u, v))

    class TimeExceeded(Exception):
        pass

    def find_hamiltonian_cycle(adj_cur, start, deadline, rng):
        n2 = len(adj_cur)
        visited = [False] * n2
        path = [start]
        visited[start] = True
        step_count = [0]
        
        # Pre-compute initial degrees for Warnsdorff? No, degrees change as we visit.
        # We need dynamic degree of unvisited neighbors.
        
        def dfs():
            step_count[0] += 1
            if step_count[0] % 1000 == 0 and time.time() > deadline:
                raise TimeExceeded()
            
            if len(path) == n2:
                return start in adj_cur[path[-1]]
                
            u = path[-1]
            # Get unvisited neighbors
            nbrs = [w for w in adj_cur[u] if not visited[w]]
            
            if not nbrs:
                return False
            
            # Warnsdorff's rule: choose neighbor with fewest unvisited neighbors
            # This is often sufficient for Hamiltonian paths/cycles
            def warnsdorff_score(w):
                return sum(1 for x in adj_cur[w] if not visited[x])
            
            # Sort by score, then random tie-breaker
            nbrs.sort(key=lambda w: (warnsdorff_score(w), rng.random()))
            
            for w in nbrs:
                visited[w] = True
                path.append(w)
                if dfs():
                    return True
                path.pop()
                visited[w] = False
            return False

        try:
            if dfs():
                return list(path)
        except TimeExceeded:
            return None
        return None

    def verify_cycles(cycles):
        used = set()
        for cyc in cycles:
            if len(cyc) < 3:
                return False
            if len(set(cyc)) != len(cyc):
                return False
            for i in range(len(cyc)):
                a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                k = _edge_key(a, b)
                if k in used:
                    return False
                if k not in full_edges:
                    return False
                used.add(k)
        return True

    best = None
    rng = random.Random(42)
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        # Copy adjacency
        adj_cur = [set(s) for s in adj]
        cycles = []
        ok = True

        # Try to find 3 cycles
        for c in range(3):
            # Choose start vertex
            # Strategy: Pick vertex with highest remaining degree
            starts = list(range(n))
            starts.sort(key=lambda v: (-len(adj_cur[v]), v))
            # Pick from top 5 for diversity, or random if few
            top_k = min(5, len(starts))
            start = starts[rng.randrange(top_k)]
            
            cyc = find_hamiltonian_cycle(adj_cur, start, deadline, rng)
            if cyc is None:
                ok = False
                break
            
            cycles.append(cyc)
            
            # Remove edges from adj_cur
            for i in range(len(cyc)):
                a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                adj_cur[a].discard(b)
                adj_cur[b].discard(a)

        if ok and len(cycles) == 3:
            if verify_cycles(cycles):
                return [[int(v) for v in cyc] for cyc in cycles]

        # If we didn't get 3, keep the best found
        if len(cycles) > 0 and verify_cycles(cycles):
            if best is None or len(cycles) > len(best):
                best = [[int(v) for v in cyc] for cyc in cycles]
        
        # Update random seed for next attempt
        rng = random.Random(attempt * 7919 + 42)

    if best is not None:
        return best

    # Fallback: single cycle
    adj_cur = [set(s) for s in adj]
    rng_fb = random.Random(0)
    cyc = find_hamiltonian_cycle(adj_cur, 0, deadline, rng_fb)
    if cyc is not None:
        return [[int(v) for v in cyc]]
    
    # Should not happen, but return empty list if all fails? 
    # Task says "nonempty list". 
    # If we fail to find any, we return a dummy? 
    # But we must return valid cycles. 
    # If no cycle found, we return [] which is invalid. 
    # However, Hamiltonian cycle always exists in this graph.
    return []
