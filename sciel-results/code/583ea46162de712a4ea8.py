import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    
    edge_set = set()
    adj = {i: set() for i in range(n)}
    for u, v in P['edges']:
        edge_set.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    def find_hamiltonian_cycle(available, rng):
        """Find a Hamiltonian cycle using only edges in available.
        Uses randomized DFS with strong pruning."""
        start = 0
        path = [start]
        visited = [False] * n
        visited[0] = True
        # Precompute which neighbors are available
        avail_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                if edge_key(u, v) in available:
                    avail_adj[u].add(v)
        
        # Count degrees in available graph
        deg = [len(avail_adj[i]) for i in range(n)]
        
        def backtrack(current, depth):
            if time.time() > deadline:
                return None
            if depth == n:
                last = path[-1]
                if edge_key(last, start) in available:
                    return list(path)
                return None
            
            last = path[-1]
            # Get available neighbors not yet visited
            candidates = [v for v in avail_adj[last] if not visited[v]]
            if not candidates:
                return None
            
            # Warnsdorff's rule: visit neighbor with fewest onward connections first
            candidates.sort(key=lambda v: len([w for w in avail_adj[v] if not visited[w] and w != last]))
            
            # Pruning: check if any unvisited vertex has degree 0 in remaining subgraph
            # (only counting edges to unvisited vertices + edge to current if applicable)
            unvisited_count = n - depth
            for u in range(n):
                if visited[u]:
                    continue
                # Count available edges from u to other unvisited vertices
                avail_to_unvisited = sum(1 for v in avail_adj[u] if not visited[v] and v != u)
                # Also count edge to 'last' if it's available (u could be next)
                # But we're checking feasibility of the remaining path
                if avail_to_unvisited == 0 and unvisited_count > 1:
                    # u has no edges to other unvisited, so it must be adjacent to last
                    # and must be visited next, but then it would be isolated
                    if unvisited_count > 1:
                        return None
            
            # Additional pruning: check that no unvisited vertex is completely isolated
            # from the rest (considering it needs to connect to both its predecessor and successor)
            for u in range(n):
                if visited[u]:
                    continue
                # In the remaining path, u needs at least 2 connections (or 1 if it's the last before start)
                # Count edges from u to unvisited vertices
                edges_to_unvisited = sum(1 for v in avail_adj[u] if not visited[v])
                # Plus potentially edge to 'last' (if u is next) or to 'start' (if u is last)
                # This is complex, so let's use a simpler check
                if edges_to_unvisited == 0:
                    # u can only connect to visited vertices
                    # It must be adjacent to 'last' (to be visited next) or to 'start' (to be last)
                    if not (edge_key(u, last) in available or edge_key(u, start) in available):
                        return None
            
            for nb in candidates:
                path.append(nb)
                visited[nb] = True
                result = backtrack(nb, depth + 1)
                if result is not None:
                    return result
                path.pop()
                visited[nb] = False
            
            return None
        
        return backtrack(start, 1)
    
    # Greedily find edge-disjoint Hamiltonian cycles
    cycles = []
    available = set(edge_set)
    rng = random.Random(42)
    
    # Try to find up to 3 cycles (theoretical max)
    for attempt in range(3):
        if time.time() > deadline:
            break
        found = None
        # Try multiple seeds for randomized restarts
        for seed_val in range(500):
            if time.time() > deadline:
                break
            rng_local = random.Random(seed_val + attempt * 1000)
            found = find_hamiltonian_cycle(available, rng_local)
            if found is not None:
                break
        if found is None:
            break
        # Remove used edges
        for i in range(len(found)):
            u = found[i]
            v = found[(i + 1) % len(found)]
            available.discard(edge_key(u, v))
        cycles.append(found)
    
    if cycles:
        return [list(c) for c in cycles]
    else:
        # Fallback: return a single valid cycle
        # Find any Hamiltonian cycle in the full graph
        for seed_val in range(100):
            rng_local = random.Random(seed_val)
            found = find_hamiltonian_cycle(edge_set, rng_local)
            if found is not None:
                return [list(found)]
        return [list(range(n))]
# EVOLVE-BLOCK-END
