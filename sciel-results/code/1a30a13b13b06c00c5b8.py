import time
from collections import deque
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
        
    # Precompute base adjacency and distances
    base_adj = {i: set() for i in range(n)}
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
        
    def get_base_dist(src):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in base_adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    # Precompute base distances for all pairs
    base_dists = [[0]*n for _ in range(n)]
    for i in range(n):
        d = get_base_dist(i)
        for j in range(n):
            base_dists[i][j] = d[j]
            
    # Generate candidate edges
    # An edge (u, v) is a candidate if:
    # 1. It's not in base.
    # 2. It doesn't create a triangle with base edges (i.e., u and v have no common neighbor in base).
    #    In a path, common neighbor exists iff |u-v| == 2.
    # 3. It's "useful": base_dist(u, v) > 4? No, even shorter ones might help if combined.
    #    But typically, to reduce diameter from 23 to 4, we need long jumps.
    #    Let's include all non-triangle-creating edges.
    
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_set:
                continue
            # Check triangle with base
            # Common neighbor in base?
            # Neighbors of u in base: u-1, u+1 (if exist)
            # Neighbors of v in base: v-1, v+1 (if exist)
            # Intersection non-empty?
            # Since it's a path, intersection is non-empty iff |u-v| == 2.
            if abs(u - v) == 2:
                continue
            candidates.append((u, v))
            
    # Sort candidates by "potential utility" (larger base distance first?) 
    # Or just keep them all. For k=10, 24 choose 2 is ~276 edges. 
    # Triangle-free removes ~23 edges. ~250 candidates.
    # C(250, 10) is huge. We need pruning.
    
    # Pruning Strategy:
    # To achieve diameter 4, every pair (i, j) with base_dist > 4 must have a path of length <= 4 in the augmented graph.
    # This implies that for any such pair, there must be a "bridge" edge.
    
    # Let's use a DFS with strong pruning.
    # State: current set of edges S.
    # Pruning 1: If current graph diameter > 4, we can still add edges. 
    #            But if the *minimum possible* diameter with remaining edges > 4, prune.
    #            Hard to compute min possible diameter quickly.
    # Pruning 2: Triangle-free check. When adding edge (u, v), ensure no w in S or Base is adjacent to both.
    
    # Better approach: Bidirectional search or meet-in-the-middle?
    # Or, since we suspect the answer is small (around 8-12), we can do a DFS that adds edges 
    # only if they significantly reduce the current diameter or fix a specific "worst pair".
    
    # Let's try a "Greedy with Backtracking" (Limited Discrepancy Search).
    # 1. Start with empty set.
    # 2. Find worst pair (u, v) with dist > 4.
    # 3. Generate all valid edges that reduce dist(u, v).
    # 4. Try each, recurse.
    
    # To make this fast, we precompute the "effect" of each candidate edge on all pairs.
    # Actually, dynamic BFS is fast enough for n=24.
    
    def get_diameter_and_worst(adj):
        max_d = 0
        worst_u, worst_v = -1, -1
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
            for dst in range(n):
                if dist[dst] > max_d:
                    max_d = dist[dst]
                    worst_u = src
                    worst_v = dst
        return max_d, worst_u, worst_v

    def is_compatible(adj, u, v):
        # Check if adding (u, v) creates a triangle.
        # u and v must not have a common neighbor in current adj.
        # Common neighbor w: w in adj[u] and w in adj[v].
        for w in adj[u]:
            if w in adj[v]:
                return False
        return True

    # Initial adjacency
    init_adj = {i: set(base_adj[i]) for i in range(n)}
    
    best_solution = None
    best_size = 100
    
    def dfs(current_adj, current_edges, depth, max_depth):
        nonlocal best_solution, best_size
        if time.time() > deadline:
            return
        
        # Prune if current depth is already >= best_size
        if len(current_edges) >= best_size:
            return
            
        diam, wu, wv = get_diameter_and_worst(current_adj)
        
        if diam <= 4:
            if len(current_edges) < best_size:
                best_size = len(current_edges)
                best_solution = list(current_edges)
            return
            
        if depth >= max_depth:
            return
            
        # Find candidates that reduce distance between wu and wv
        # Or just any candidate that is "good".
        # Let's prioritize candidates that are far apart in base?
        # Or candidates that are adjacent to wu or wv?
        
        # Heuristic: 
        # To reduce dist(wu, wv), we should add an edge (x, y) such that 
        # dist(wu, x) + 1 + dist(y, wv) < dist(wu, wv)
        # or similar.
        
        # Let's just iterate through candidates in a smart order.
        # Sort candidates by how much they reduce the worst pair distance.
        
        valid_candidates = []
        for (u, v) in candidates:
            if (min(u,v), max(u,v)) in set(map(lambda e: (min(e[0],e[1]), max(e[0],e[1])), current_edges)):
                continue
            if not is_compatible(current_adj, u, v):
                continue
            
            # Estimate reduction
            # We need dist in current graph. 
            # Instead of full BFS for each candidate (too slow), 
            # we can use a heuristic: 
            # If u is closer to wu and v closer to wv (or vice versa), it's good.
            
            # For now, let's just add all compatible candidates and let the recursion prune.
            # But we need to limit branching.
            # Let's pick the top K candidates based on a heuristic score.
            
            # Heuristic: Sum of base distances from endpoints to wu and wv?
            # Actually, let's just try the ones that are "long" shortcuts.
            valid_candidates.append((u, v))
            
        # Sort candidates by base distance descending (longer jumps first)
        valid_candidates.sort(key=lambda x: base_dists[x[0]][x[1]], reverse=True)
        
        # Limit branching factor to avoid explosion
        # If we have too many, take top 10?
        # This is a heuristic. 
        # To be safe and find the optimum, we might need to search more.
        # But with max_depth ~ 12, and branching 10, 10^12 is too big.
        
        # Better: Use the "worst pair" to filter.
        # Only consider edges (u, v) that actually shorten the path between wu and wv.
        
        # We need dist_current(wu, u) and dist_current(wv, v).
        # Let's compute dist from wu and wv in current graph once.
        dist_wu = [-1] * n
        dist_wu[wu] = 0
        q = deque([wu])
        while q:
            u = q.popleft()
            for v in current_adj[u]:
                if dist_wu[v] == -1:
                    dist_wu[v] = dist_wu[u] + 1
                    q.append(v)
                    
        dist_wv = [-1] * n
        dist_wv[wv] = 0
        q = deque([wv])
        while q:
            u = q.popleft()
            for v in current_adj[u]:
                if dist_wv[v] == -1:
                    dist_wv[v] = dist_wv[u] + 1
                    q.append(v)
                    
        filtered_candidates = []
        for (u, v) in valid_candidates:
            # Check if adding (u, v) helps wu-wv
            # Path wu -> ... -> u -> v -> ... -> wv
            # dist = dist_wu[u] + 1 + dist_wv[v]
            # Or wu -> ... -> v -> u -> ... -> wv
            # dist = dist_wu[v] + 1 + dist_wv[u]
            
            d1 = dist_wu[u] + 1 + dist_wv[v] if dist_wu[u] != -1 and dist_wv[v] != -1 else 100
            d2 = dist_wu[v] + 1 + dist_wv[u] if dist_wu[v] != -1 and dist_wv[u] != -1 else 100
            
            if min(d1, d2) < diam:
                filtered_candidates.append((u, v, min(d1, d2)))
                
        # Sort by new distance (smaller is better)
        filtered_candidates.sort(key=lambda x: x[2])
        
        # Branch on top few
        # If no candidates, dead end
        for u, v, _ in filtered_candidates[:8]: # Limit to 8 branches
            # Add edge
            current_adj[u].add(v)
            current_adj[v].add(u)
            current_edges.append((u, v))
            
            dfs(current_adj, current_edges, depth + 1, max_depth)
            
            # Remove edge
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            current_edges.pop()
            
            if time.time() > deadline:
                return

    # Start DFS
    # Max depth: We know 10 is possible. Let's search up to 15 to be safe, but prune by best_size.
    # We want to find the MINIMUM. 
    # If we find a solution of size k, we can stop searching branches of size >= k.
    
    # Let's try increasing max_depth until we find a solution?
    # No, DFS with pruning is better.
    
    # Initialize best_size to a known good value (10) to allow pruning.
    # We can run a quick greedy to get an initial upper bound.
    
    def greedy_init():
        adj = {i: set(base_adj[i]) for i in range(n)}
        edges = []
        while True:
            diam, wu, wv = get_diameter_and_worst(adj)
            if diam <= 4:
                break
            # Find best edge
            best_e = None
            best_d = diam + 1
            for (u, v) in candidates:
                if (min(u,v), max(u,v)) in set(map(lambda e: (min(e[0],e[1]), max(e[0],e[1])), edges)):
                    continue
                if not is_compatible(adj, u, v):
                    continue
                # Tentative add
                adj[u].add(v)
                adj[v].add(u)
                new_diam, _, _ = get_diameter_and_worst(adj)
                adj[u].remove(v)
                adj[v].remove(u)
                
                if new_diam < best_d:
                    best_d = new_diam
                    best_e = (u, v)
            if best_e is None:
                break
            u, v = best_e
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
        return edges, len(edges)
        
    init_edges, init_len = greedy_init()
    best_size = init_len
    best_solution = init_edges
    
    # Now run DFS to try to beat init_len
    # We search for solutions of size < init_len
    # So max_depth can be init_len - 1
    
    if init_len > 1:
        dfs(init_adj, [], 0, init_len - 1)
        
    if best_solution is None:
        # Fallback to greedy
        return [[int(u), int(v)] for u, v in init_edges]
        
    return [[int(u), int(v)] for u, v in best_solution]

# EVOLVE-BLOCK-END
