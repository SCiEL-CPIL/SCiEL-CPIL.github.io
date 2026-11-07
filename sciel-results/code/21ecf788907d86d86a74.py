import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    
    # Precompute all-pairs shortest paths (BFS from each vertex)
    # dist[i][j] = shortest distance from i to j
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[s][v] == -1:
                    dist[s][v] = dist[s][u] + 1
                    q.append(v)
    
    # Check if two vertices are adjacent
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    def is_adjacent(u, v):
        return (min(u,v), max(u,v)) in edge_set
    
    # Find k vertex-disjoint paths between k pairs of branch vertices
    # Uses max-flow formulation: split each vertex into in/out with capacity 1
    # For branches, allow capacity k (they can be used by multiple paths)
    # Actually, simpler: use the standard vertex-disjoint paths via max flow
    # with vertex splitting.
    
    def find_disjoint_paths(branches):
        """Find vertex-disjoint paths for all pairs of branches.
        Returns list of paths or None."""
        k = len(branches)
        pairs = list(combinations(range(k), 2))
        
        # Build flow network with vertex splitting
        # Each vertex v becomes v_in (2*v) and v_out (2*v+1)
        # Edge v_in -> v_out has capacity 1 (for non-branch) or k (for branch)
        # For each graph edge (u,v): u_out -> v_in with capacity 1, v_out -> u_in with capacity 1
        # Source connects to branch_in with capacity 1, branch_out to sink with capacity 1
        # Actually, for finding disjoint paths between specific pairs, we need a different approach.
        
        # Better approach: use the standard reduction.
        # Create a flow network where:
        # - Each vertex v has an edge v_in -> v_out with capacity 1 (or infinity for branches)
        # - Each original edge (u,v) becomes edges u_out -> v_in and v_out -> u_in with capacity 1
        # - For each pair (b_i, b_j) in our pairs, we add a super source S connected to b_i_in 
        #   with capacity 1, and b_j_out connected to super sink T with capacity 1.
        #   Wait, this doesn't work directly for multiple pairs.
        
        # Actually, the standard approach for "k vertex-disjoint paths connecting k specified pairs"
        # is to use a flow network with vertex splitting and check if max flow = k.
        # But we need to specify WHICH pairs. 
        # 
        # Alternative: Use the fact that we want paths between ALL pairs of branches.
        # This is equivalent to finding a subdivision of K_k.
        # 
        # Let me use a different approach: iterative BFS with backtracking, but with 
        # better heuristics.
        
        # Actually, let me use the max-flow approach properly.
        # For a given set of branches B = {b_0, ..., b_{k-1}}, we want to find
        # vertex-disjoint paths p_{ij} for all pairs (i,j) with i<j.
        # 
        # This is a "disjoint paths" problem with k(k-1)/2 pairs.
        # 
        # One approach: reduce to a single max-flow problem.
        # Create a network where:
        # - Split each vertex v into v_in and v_out, with edge v_in -> v_out capacity 1
        #   (except for branch vertices which get capacity k)
        # - For each original edge (u,v): add edges u_out -> v_in and v_out -> u_in with capacity 1
        # - Add a super source S and super sink T
        # - For each pair (i,j), connect S -> b_i_in with capacity 1 and b_j_out -> T with capacity 1
        #   But this doesn't enforce which pair goes where...
        #
        # Actually, the correct formulation for finding disjoint paths for SPECIFIED pairs
        # is more complex. Let me use a simpler approach:
        # 
        # Since k is small (at most ~6-7), we can use backtracking with good pruning.
        # Key insight: order the pairs by "difficulty" (shortest path length in residual graph).
        
        def bfs_path(s, t, blocked):
            """Find shortest path from s to t avoiding blocked vertices."""
            if s in blocked or t in blocked:
                return None
            parent = {s: None}
            q = deque([s])
            while q:
                u = q.popleft()
                if u == t:
                    # Reconstruct path
                    path = []
                    cur = t
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    return path
                for v in adj[u]:
                    if v not in parent and v not in blocked:
                        parent[v] = u
                        q.append(v)
            return None
        
        def find_k_paths(branches):
            k = len(branches)
            pairs = list(combinations(range(k), 2))
            
            # Order pairs by shortest path length (in original graph, ignoring blocked)
            # to process "easy" pairs first
            def pair_key(p):
                i, j = p
                return dist[branches[i]][branches[j]]
            
            pairs_sorted = sorted(pairs, key=pair_key)
            
            used = set()
            paths = {}
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == len(pairs_sorted):
                    return True
                
                i, j = pairs_sorted[idx]
                u, v = branches[i], branches[j]
                
                # Find multiple candidate paths
                # Use BFS to find shortest, then try variations
                candidate_paths = []
                
                # BFS for shortest path
                path = bfs_path(u, v, used)
                if path is None:
                    return False
                candidate_paths.append(path)
                
                # Try to find alternative paths by blocking the shortest path's interior
                # and finding another path
                blocked = set(used)
                for _ in range(3):
                    p = bfs_path(u, v, blocked)
                    if p is None:
                        break
                    candidate_paths.append(p)
                    # Block the interior of this path to find another
                    for mid in p[1:-1]:
                        blocked.add(mid)
                
                # Try each candidate
                for p in candidate_paths:
                    interiors = set(p[1:-1])
                    if interiors & used:
                        continue
                    used.update(interiors)
                    paths[(i, j)] = p
                    if backtrack(idx + 1):
                        return True
                    used.difference_update(interiors)
                    del paths[(i, j)]
                
                return False
            
            if backtrack(0):
                # Convert to list format
                result = []
                for i, j in pairs:
                    result.append(paths[(i, j)])
                return result
            return None
        
        return find_k_paths(branches)
    
    # Search for maximum branch set
    # Strategy: try large sets first, using various heuristics for candidate selection
    
    best_branches = None
    best_paths = None
    best_k = 2
    
    # Fallback: find an edge
    if edges:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]
        best_k = 2
    
    # Try to find triangles
    for i in range(n):
        if time.time() > deadline - 2:
            break
        for j in range(i + 1, n):
            if not is_adjacent(i, j):
                continue
            for k in range(j + 1, n):
                if is_adjacent(j, k) and is_adjacent(i, k):
                    best_branches = [i, j, k]
                    best_paths = [[i, j], [j, k], [i, k]]
                    best_k = 3
                    break
            if best_k == 3:
                break
        if best_k == 3:
            break
    
    # Try larger sets
    # Use multiple strategies for candidate generation
    strategies = []
    
    # Strategy 1: Top degree vertices
    sorted_by_deg = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    strategies.append(sorted_by_deg[:12])
    
    # Strategy 2: Vertices with high "cliquishness" (many neighbors that are also connected)
    def cliquish_score(v):
        score = 0
        for u in adj[v]:
            for w in adj[u]:
                if w > v and w != u and is_adjacent(v, w):
                    score += 1
        return score
    
    sorted_by_cliquish = sorted(range(n), key=lambda x: cliquish_score(x), reverse=True)
    strategies.append(sorted_by_cliquish[:12])
    
    # Strategy 3: Mix - some top degree, some high betweenness-like
    # Use distance-based: vertices that are "central" (low average distance)
    avg_dist = []
    for v in range(n):
        d = [dist[v][u] for u in range(n) if dist[v][u] > 0]
        avg_dist.append(sum(d) / len(d) if d else 999)
    sorted_by_central = sorted(range(n), key=lambda x: avg_dist[x])
    strategies.append(sorted_by_central[:12])
    
    # Strategy 4: Random samples
    for _ in range(5):
        random_sample = random.sample(range(n), 12)
        strategies.append(random_sample)
    
    # Try k from 6 down to 4
    for k in range(6, 3, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for strat in strategies:
            if time.time() > deadline - 5:
                break
            
            # Generate combinations from this strategy
            # Limit the number of combinations we try
            max_combos = 500
            combo_count = 0
            for combo in combinations(strat, k):
                if time.time() > deadline - 3:
                    break
                if combo_count >= max_combos:
                    break
                combo_count += 1
                
                branches = list(combo)
                paths = find_disjoint_paths(branches)
                if paths is not None:
                    best_branches = branches
                    best_paths = paths
                    best_k = k
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
    
    # Also try k=4 with more combinations if not found yet
    if best_k < 4 and time.time() < deadline - 10:
        # Try all combinations from a broader set
        top_candidates = sorted(range(n), key=lambda x: degrees[x], reverse=True)[:15]
        combo_count = 0
        max_combos = 2000
        for combo in combinations(top_candidates, 4):
            if time.time() > deadline - 3:
                break
            if combo_count >= max_combos:
                break
            combo_count += 1
            
            branches = list(combo)
            paths = find_disjoint_paths(branches)
            if paths is not None:
                best_branches = branches
                best_paths = paths
                best_k = 4
                break
    
    # Final validation and type conversion
    if best_paths is None or best_k < 2:
        if edges:
            u, v = edges[0]
            best_branches = [int(u), int(v)]
            best_paths = [[int(u), int(v)]]
        else:
            best_branches = [0, 1]
            best_paths = [[0, 1]]
    
    # Ensure all types are correct
    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(v) for v in p] for p in best_paths]
    
    return {
        'branches': branches_out,
        'paths': paths_out
    }
