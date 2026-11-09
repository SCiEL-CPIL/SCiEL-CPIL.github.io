import time
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = P['n']
    base_edges = [(e[0], e[1]) for e in P['edges']]
    base_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in base_edges)
    
    # Precompute all possible non-base edges
    all_possible = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_set:
                all_possible.append((u, v))
    
    # Sort candidate edges. 
    # Heuristic: Edges that are "long" reduce diameter more effectively.
    # We want to find a small set of edges. 
    # A good ordering for DFS might prioritize edges that cover large gaps.
    # However, triangle-free constraint is tight.
    # Let's try to find the minimum k by iterating k from 1 upwards.
    
    def _build_adj(n, edge_list):
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _is_triangle_free(n, edge_list):
        adj = _build_adj(n, edge_list)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True

    def _diameter(n, edge_list):
        adj = _build_adj(n, edge_list)
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            if -1 in dist:
                return None
            max_dist = max(max_dist, max(dist))
        return max_dist

    def _validate(n, base_edges, added_edges):
        all_edges = list(base_edges) + list(added_edges)
        seen = set()
        for u, v in all_edges:
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        if not _is_triangle_free(n, all_edges):
            return False
        d = _diameter(n, all_edges)
        if d is None or d > 4:
            return False
        return True

    # Strategy: Iterative Deepening DFS
    # We try to find a solution with exactly k edges.
    # If found, return it.
    # The search space is C(~276, k). For k=5, that's ~10^10, too big for naive DFS.
    # We need strong pruning.
    
    # Pruning 1: Triangle-free. When adding an edge (u,v), check if it creates a triangle.
    # Pruning 2: Diameter. We can check if the current partial set of edges *can* possibly 
    # lead to diameter <= 4. This is hard to check incrementally without full BFS.
    # However, we can check if the graph is connected and if the "current" diameter 
    # (with only selected edges) is already > 4 + (remaining_edges * max_reduction_per_edge).
    # This is complex.
    
    # Alternative: Use the fact that the base graph is a path.
    # Adding an edge (i, j) effectively creates a shortcut.
    # Let's try a specific structured search first, which is likely to find the optimum.
    # The optimum is likely small (e.g., 5-8 edges).
    
    # Let's use a randomized local search with a strong restart strategy, 
    # but guided by the "triangle-free" constraint more aggressively.
    
    import random
    random.seed(42)
    
    best_added = None
    best_count = float('inf')
    
    def try_candidate(added_edges):
        nonlocal best_added, best_count
        if len(added_edges) >= best_count:
            return
        if _validate(n, base_edges, added_edges):
            best_added = [list(e) for e in added_edges]
            best_count = len(added_edges)
    
    # 1. Try small k values with a randomized approach
    # We know the answer is likely small. Let's try to find a valid set of size k.
    
    # Heuristic for candidate generation:
    # To reduce diameter of a path of length 23 to 4, we need to "fold" the path.
    # Edges should be long.
    # Let's prioritize edges with length >= 5.
    
    long_edges = [(u, v) for u, v in all_possible if v - u >= 5]
    
    # Try to find a solution with k edges
    for k in range(1, 10):
        if time.time() > deadline:
            break
            
        found = False
        # Randomized restarts
        max_restarts = 5000 if k < 6 else 1000
        for _ in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Select k random long edges
            if len(long_edges) < k:
                continue
            added = random.sample(long_edges, k)
            
            # Quick check: Triangle free?
            # We can do this fast
            adj = _build_adj(n, base_edges + added)
            tf = True
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        if adj[u] & adj[v]:
                            tf = False
                            break
                if not tf:
                    break
            if not tf:
                continue
                
            # Check diameter
            d = _diameter(n, base_edges + added)
            if d is not None and d <= 4:
                try_candidate(added)
                found = True
                break
        
        if found:
            # We found a solution with k edges. 
            # Since we iterate k from 1 upwards, this is the minimum.
            break
            
    # If we didn't find a solution via random sampling (or k went up to 9),
    # try a more directed search or return the best found.
    
    # Fallback: If best_added is still None, try a greedy approach to find *any* valid solution
    # and then minimize.
    if best_added is None:
        # Greedy: Add edges that reduce diameter most
        current_edges = list(base_edges)
        added = []
        added_set = set()
        
        while time.time() < deadline:
            d = _diameter(n, current_edges)
            if d is not None and d <= 4:
                break
                
            adj = _build_adj(n, current_edges)
            # Find farthest pair
            max_pair_dist = 0
            max_pair = None
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                q = deque([src])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            q.append(v)
                for dst in range(src + 1, n):
                    if dist[dst] > max_pair_dist:
                        max_pair_dist = dist[dst]
                        max_pair = (src, dst)
                        
            if max_pair is None:
                break
                
            u, v = max_pair
            key = (min(u, v), max(u, v))
            if key in base_set or key in added_set:
                # If we can't add the best edge, we might be stuck or need to try others.
                # For simplicity in fallback, break.
                break
                
            new_edges = current_edges + [(u, v)]
            if _is_triangle_free(n, new_edges):
                added.append((u, v))
                added_set.add(key)
                current_edges = new_edges
            else:
                break
                
        if added:
            try_candidate(added)
            
    if best_added is not None:
        return [[int(u), int(v)] for u, v in best_added]
        
    # If nothing found, return empty (invalid, but must return something)
    return []

# EVOLVE-BLOCK-END
