import random
import time
import sys

# Set recursion limit higher just in case, though n=24 is small
sys.setrecursionlimit(10000)

def construct():
    n = 24
    # 1. Generate the edge list order
    rng = random.Random(614)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            edges.append((u, v))
    rng.shuffle(edges)
    
    # Map edge (u, v) to its index in the shuffled list
    # idx[u][v] is 0-based index
    idx = [[0] * n for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        idx[u][v] = i
        idx[v][u] = i
        
    total_edges = len(edges)
    
    # 2. Binary search on K (1-based arrival index)
    # We want the smallest K such that G_K has a Hamiltonian cycle.
    # G_K contains edges with index < K (0-based).
    # So if we find min K, the score is K.
    
    def has_hamiltonian_cycle(K):
        # K is the 1-based limit. Edges with index < K are allowed.
        # Build adjacency lists
        adj = [[] for _ in range(n)]
        deg = [0] * n
        
        for u in range(n):
            for v in range(u + 1, n):
                if idx[u][v] < K:
                    adj[u].append(v)
                    adj[v].append(u)
                    deg[u] += 1
                    deg[v] += 1
        
        # Pruning 1: Degree check
        for i in range(n):
            if deg[i] < 2:
                return False
        
        # Pruning 2: Connectivity check
        # BFS from 0
        visited = [False] * n
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        if count != n:
            return False
            
        # Pruning 3: DFS for Hamiltonian Cycle
        # Fix start at 0
        # path[0] = 0
        # We need to find a permutation of 1..n-1 such that edges exist
        # and last vertex connects to 0.
        
        # To make DFS faster, sort neighbors by degree in the current graph?
        # Or just use a fixed order.
        # Let's use a bitmask for visited.
        # n=24 fits in an int.
        
        # Heuristic: Always start with 0.
        # Next vertex: try neighbors of 0.
        # To break symmetry, we can assume the second vertex is the smallest neighbor?
        # No, that might miss the cycle if the cycle goes 0 -> big -> ... -> small -> 0.
        # But we can fix the direction? No, undirected.
        # We can just iterate all neighbors of 0.
        
        # Optimization:
        # Pre-compute neighbors sorted by their degree in H_K (ascending) to fail first.
        for u in range(n):
            adj[u].sort(key=lambda x: deg[x])
            
        # DFS
        visited_mask = 1 << 0  # 0 is visited
        path = [0]
        
        # Time limit for this specific check?
        # The binary search will call this ~9 times (log2(276)).
        # If one call is too slow, we might TLE.
        # n=24 is small enough that DFS with pruning should be fast for non-trivial K.
        # For very small K, it fails at degree/connectivity.
        # For very large K, it succeeds quickly.
        # The hard part is near the threshold.
        
        def dfs(current, depth):
            if depth == n:
                # Check if current connects to start (0)
                # Since we built adj based on K, we just check if 0 is in adj[current]
                # But we need to ensure the edge (current, 0) is valid.
                # It is in adj if idx[current][0] < K.
                return 0 in adj[current]
            
            # Try neighbors
            # Heuristic: pick neighbor with fewest unvisited neighbors?
            # This requires dynamic calculation.
            # Simpler: just iterate adj[current].
            for next_v in adj[current]:
                if not (visited_mask & (1 << next_v)):
                    # Check if next_v can lead to a solution
                    # Pruning: If next_v is the last vertex to be visited (depth == n-1),
                    # it must connect to 0.
                    if depth == n - 1:
                        if 0 in adj[next_v]:
                            return True
                    else:
                        # Optional: Check if next_v has enough unvisited neighbors?
                        # If next_v has 0 unvisited neighbors and depth < n-1, it's a dead end.
                        # Count unvisited neighbors of next_v
                        # This is O(n) per step.
                        # Let's try without this first, add if needed.
                        
                        if dfs(next_v, depth + 1):
                            return True
                        visited_mask ^= (1 << next_v) # Unvisit
                        continue # backtrack
                    visited_mask ^= (1 << next_v) # Unvisit if returning
                    # Wait, the logic above is messy.
                    
            return False
        
        # Let's rewrite DFS cleanly with backtracking
        def dfs2(current, depth):
            if depth == n:
                return 0 in adj[current]
            
            # Optional Pruning: 
            # If any unvisited vertex has no neighbors in the unvisited set, 
            # and it's not the last one to be added, it might be stuck.
            # But "unvisited set" changes.
            
            for next_v in adj[current]:
                if not (visited_mask & (1 << next_v)):
                    # If this is the last vertex, it must connect to 0
                    if depth == n - 1:
                        if 0 in adj[next_v]:
                            return True
                    else:
                        visited_mask |= (1 << next_v)
                        if dfs2(next_v, depth + 1):
                            return True
                        visited_mask &= ~(1 << next_v)
            return False

        # Start DFS from 0
        # To reduce symmetry, we can fix the second node to be the smallest neighbor?
        # No, just iterate.
        # But we can optimize by trying neighbors of 0 in order of their degree.
        
        # One more pruning: 
        # If the graph is not 2-connected, it might not have a HC.
        # Checking 2-connectivity is expensive.
        
        # Let's just run dfs2
        # We need to pass visited_mask by reference or return it.
        # In Python, integers are immutable. We need to update the outer scope.
        # Let's use a list or nonlocal.
        
        visited_mask = [1 << 0]
        
        def dfs3(current, depth):
            if depth == n:
                return 0 in adj[current]
            
            for next_v in adj[current]:
                if not (visited_mask[0] & (1 << next_v)):
                    if depth == n - 1:
                        if 0 in adj[next_v]:
                            return True
                    else:
                        visited_mask[0] |= (1 << next_v)
                        if dfs3(next_v, depth + 1):
                            return True
                        visited_mask[0] &= ~(1 << next_v)
            return False

        return dfs3(0, 1)

    # Binary Search
    lo, hi = 1, total_edges
    best_K = total_edges
    
    # Deadline guard
    start_time = time.time()
    deadline = 80.0  # Leave 10s margin
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        if has_hamiltonian_cycle(mid):
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    # Return the cycle?
    # The task asks for a permutation of vertices 0..23 describing the cycle.
    # The score is calculated by the evaluator.
    # We just need to return *a* valid cycle.
    # If best_K is found, we can extract the cycle from the last successful DFS?
    # No, `has_hamiltonian_cycle` doesn't return the cycle.
    # We need to return a permutation.
    # If we just return `list(range(n))`, it might not be a valid cycle in G_best_K?
    # Wait, the evaluator checks the score of the *returned* cycle.
    # If we return an invalid cycle (not a Hamiltonian cycle in the complete graph? No, any permutation is a Hamiltonian cycle in K_n).
    # The score is the max index of its edges.
    # If we return a cycle that has a max index > best_K, the score will be worse.
    # So we MUST return a cycle that actually exists in G_best_K.
    
    # So we need to modify `has_hamiltonian_cycle` to also retrieve the cycle when found.
    
    # Let's refactor to find the cycle.
    
    def find_cycle(K):
        adj = [[] for _ in range(n)]
        deg = [0] * n
        
        for u in range(n):
            for v in range(u + 1, n):
                if idx[u][v] < K:
                    adj[u].append(v)
                    adj[v].append(u)
                    deg[u] += 1
                    deg[v] += 1
        
        for i in range(n):
            if deg[i] < 2:
                return None
        
        visited_bfs = [False] * n
        queue = [0]
        visited_bfs[0] = True
        count = 1
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited_bfs[v]:
                    visited_bfs[v] = True
                    count += 1
                    queue.append(v)
        if count != n:
            return None
            
        for u in range(n):
            adj[u].sort(key=lambda x: deg[x])
            
        path = [0]
        visited_mask = [1 << 0]
        
        def dfs4(current, depth):
            if depth == n:
                if 0 in adj[current]:
                    path.append(current) # Wait, path already has n-1 vertices?
                    # path starts with [0]. Depth 1 means 1 vertex in path.
                    # When depth == n, we have n-1 vertices in path?
                    # Let's trace:
                    # dfs(0, 1) -> path=[0]. depth=1.
                    # ...
                    # dfs(u, n) -> path has n-1 vertices?
                    # If depth is the number of vertices in path, then:
                    # Start: path=[0], depth=1.
                    # Add v: path=[0,v], depth=2.
                    # ...
                    # Add u: path=[0,...,u], depth=n.
                    # Then check if u connects to 0.
                    # If yes, cycle is path + [0] (implicit).
                    # The permutation is path.
                    return True
                return False
            
            for next_v in adj[current]:
                if not (visited_mask[0] & (1 << next_v)):
                    if depth == n - 1:
                        if 0 in adj[next_v]:
                            path.append(next_v)
                            return True
                    else:
                        visited_mask[0] |= (1 << next_v)
                        path.append(next_v)
                        if dfs4(next_v, depth + 1):
                            return True
                        path.pop()
                        visited_mask[0] &= ~(1 << next_v)
            return False

        if dfs4(0, 1):
            return path
        return None

    # Re-run binary search, but store the best K and then find the cycle for that K?
    # Or find cycle during binary search?
    # It's safer to find the min K first, then find the cycle for that K.
    # But `find_cycle` is expensive.
    # We can just use the previous `has_hamiltonian_cycle` logic to find K,
    # and then run `find_cycle` once for the final K.
    
    # However, the previous `has_hamiltonian_cycle` didn't store the cycle.
    # Let's just use `find_cycle` in the binary search?
    # If `find_cycle` returns a cycle, we have it.
    # If not, we continue.
    # But we need the *minimal* K.
    
    # Strategy:
    # 1. Binary search for min K using a fast check (degree+conn+dfs existence).
    # 2. Once min K is found, run `find_cycle(min_K)` to get the actual permutation.
    # 3. If `find_cycle` fails (shouldn't happen if check was correct), fallback to something?
    
    # Let's combine: The DFS in `has_hamiltonian_cycle` can be the same as in `find_cycle`,
    # just without storing the path.
    # Actually, storing the path is cheap.
    # Let's just use `find_cycle` in the binary search.
    # If it returns a cycle, we update `best_K` and `best_cycle`.
    
    best_cycle = None
    lo, hi = 1, total_edges
    best_K = total_edges
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        cycle = find_cycle(mid)
        if cycle is not None:
            best_K = mid
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle is None:
        # Fallback: return a default cycle (might have high score)
        # This should not happen as K_n always has HC.
        # But if time ran out, we might not have found one.
        # In that case, we should return the best found so far?
        # We didn't store intermediate cycles.
        # We should store the best cycle found in any successful check.
        # Since we search downwards (hi = mid - 1), the last successful check is the best.
        # But if we break due to time, we might have a valid `best_cycle` from a previous iteration.
        
        # If no cycle found at all (should not happen for large K), return identity?
        return list(range(n))
        
    return best_cycle

# Test locally?
# No, just output.
