"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import collections
sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # 1. Build Bipartite Graph for Matching
    # Left side: 0..n-1 (u_out), Right side: n..2n-1 (v_in)
    # Edge (u, v) in G -> (u, n+v) and (v, n+u) in Bipartite
    
    bip_adj = [[] for _ in range(2*n)]
    for u, v in edges:
        bip_adj[u].append(n + v)
        bip_adj[v].append(n + u)
        
    # Hopcroft-Karp Algorithm for Maximum Bipartite Matching
    # U is left side (0..n-1), V is right side (n..2n-1)
    pair_u = [-1] * n
    pair_v = [-1] * n
    dist = [0] * n
    
    from collections import deque
    
    def bfs():
        q = deque()
        found = False
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        while q:
            u = q.popleft()
            for v in bip_adj[u]:
                u2 = v - n
                if pair_v[u2] == -1:
                    found = True
                elif dist[u2] == -1:
                    dist[u2] = dist[u] + 1
                    q.append(u2)
        return found

    def dfs(u):
        for v in bip_adj[u]:
            u2 = v - n
            if pair_v[u2] == -1 or (dist[u2] == dist[u] + 1 and dfs(u2)):
                pair_u[u] = v
                pair_v[u2] = u
                return True
        dist[u] = -1
        return False

    matching_size = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u):
                    matching_size += 1

    # 2. Extract Disjoint Paths from Matching
    # The matching defines edges in the original graph.
    # Build the subgraph H from matched edges.
    # If (u, n+v) is in matching, then (u, v) is an edge in H.
    # If (v, n+u) is in matching, then (v, u) is an edge in H.
    
    h_adj = [set() for _ in range(n)]
    for u in range(n):
        if pair_u[u] != -1:
            v = pair_u[u] - n
            h_adj[u].add(v)
            h_adj[v].add(u)
            
    # Find connected components in H
    visited = [False] * n
    paths = []
    
    for i in range(n):
        if not visited[i] and len(h_adj[i]) > 0:
            # Start a path
            # Find an endpoint (degree 1) or just any node if cycle
            # Since H is a collection of paths and cycles, we can traverse
            path = []
            current = i
            prev = -1
            while True:
                visited[current] = True
                path.append(current)
                next_node = -1
                for nei in h_adj[current]:
                    if nei != prev:
                        next_node = nei
                        break
                if next_node == -1:
                    break
                prev, current = current, next_node
            paths.append(path)
            
        elif not visited[i] and len(h_adj[i]) == 0:
            # Isolated vertex in H (not matched)
            paths.append([i])
            visited[i] = True

    # Note: The above traversal might miss cycles or handle them incorrectly if we start in the middle.
    # Let's refine path extraction.
    # We need to ensure we get maximal paths.
    # The structure of H is a set of disjoint paths and cycles.
    
    # Reset visited
    visited = [False] * n
    paths = []
    
    for i in range(n):
        if not visited[i]:
            if len(h_adj[i]) == 0:
                paths.append([i])
                visited[i] = True
            else:
                # Check if it's a cycle
                # Pick a node, traverse, see if we close the loop
                # Or just traverse until we hit a node with no unvisited neighbors in H
                # Since max degree is 2, it's simple.
                
                # Find an endpoint of this component
                start = i
                # If degree is 2, it might be a cycle. If degree is 1, it's a path end.
                # Let's just pick i and go to a neighbor.
                # To get a full path, we should start from an endpoint if it's a path.
                # If it's a cycle, any start works, but we need to break it.
                
                # Find all nodes in this component of H
                comp = []
                stack = [i]
                visited[i] = True
                while stack:
                    node = stack.pop()
                    comp.append(node)
                    for nei in h_adj[node]:
                        if not visited[nei]:
                            visited[nei] = True
                            stack.append(nei)
                
                # Now, within comp, find the path structure.
                # If all have degree 2, it's a cycle.
                # If some have degree 1, it's a path.
                
                # Find endpoints (degree 1)
                endpoints = [u for u in comp if len(h_adj[u]) == 1]
                
                if endpoints:
                    # It's a path
                    start_node = endpoints[0]
                    path = [start_node]
                    prev = -1
                    curr = start_node
                    while True:
                        next_node = -1
                        for nei in h_adj[curr]:
                            if nei != prev:
                                next_node = nei
                                break
                        if next_node == -1:
                            break
                        path.append(next_node)
                        prev, curr = curr, next_node
                    paths.append(path)
                else:
                    # It's a cycle
                    # Break it into a path by removing one edge
                    # Just traverse and stop before returning to start
                    start_node = comp[0]
                    path = [start_node]
                    prev = -1
                    curr = start_node
                    while True:
                        next_node = -1
                        for nei in h_adj[curr]:
                            if nei != prev and nei != start_node: # Don't close the cycle yet
                                next_node = nei
                                break
                        if next_node == -1:
                            # If we are stuck, it means we are at the last node before start
                            # But we need to include all nodes.
                            # In a cycle, every node has 2 neighbors.
                            # If we start at 0, go to 1... until we reach k.
                            # k's neighbors are k-1 and 0.
                            # We skip 0 (start) and k-1 (prev). So next_node is -1.
                            # We stop. The path is 0..k.
                            break
                        path.append(next_node)
                        prev, curr = curr, next_node
                    paths.append(path)

    # 3. Merge Paths
    # We have a list of paths. We want to find an ordering and orientation that allows concatenation.
    # If len(paths) == 1, return it.
    if len(paths) == 1:
        return paths[0]
        
    # Create nodes for DP: (path_index, orientation)
    # Orientation 0: forward, 1: reversed
    num_paths = len(paths)
    
    # Precompute start and end of each path in both orientations
    path_ends = []
    for p in paths:
        path_ends.append((p[0], p[-1])) # (start_fwd, end_fwd)
        
    # Compatibility check: Can we go from (i, o1) to (j, o2)?
    # End of (i, o1) must be connected to Start of (j, o2) in G.
    
    def get_end(i, o):
        if o == 0:
            return paths[i][-1]
        else:
            return paths[i][0]
            
    def get_start(i, o):
        if o == 0:
            return paths[i][0]
        else:
            return paths[i][-1]

    # DP state: (mask, last_path_idx, last_orientation)
    # mask: bitmask of used paths
    # last_path_idx: which path was last
    # last_orientation: orientation of the last path
    
    # Since num_paths can be up to 50 (if matching is poor), 2^50 is too big.
    # But in a random graph, the matching will cover most vertices in few long paths.
    # If num_paths > 20, we might timeout.
    # Heuristic: If num_paths is large, just return the longest path.
    
    if num_paths > 20:
        # Fallback to longest single path
        longest = max(paths, key=len)
        return longest
        
    # DP
    # State: (mask, last_idx, last_orient)
    # Value: length of the concatenated path so far (number of vertices)
    # We want to maximize the total vertices.
    
    # Actually, we just need to find if a Hamiltonian path exists in the meta-graph.
    # If it does, the total length is sum(len(p) for p in paths).
    # If not, we want the max subset.
    
    # Let's just try to find a valid ordering for ALL paths.
    # If found, return it.
    # If not, try to find the max subset?
    # Given time, let's just try to find a valid ordering for all.
    
    # Meta-Graph edges:
    # can_go[(i, o1)][(j, o2)] = True if end(i,o1) connected to start(j,o2)
    
    # We can use backtracking to find a permutation of paths and orientations.
    # Since num_paths <= 20, we can use DP over subsets.
    
    # dp[mask][last_idx][last_orient] = True if we can form a path covering mask, ending with last_idx in last_orient
    # To save memory, we can just store the parent pointers or reconstruct.
    # But we just need to return the path.
    
    # Let's use a simpler backtracking with pruning if num_paths is small.
    # If num_paths is 10, 10! is 3.6M, which is okay.
    # If num_paths is 20, 20! is huge.
    
    # Let's stick to DP for num_paths <= 15, else fallback.
    
    if num_paths <= 15:
        # dp[mask][last_idx][last_orient]
        # mask size 2^15 = 32768
        # last_idx 15
        # last_orient 2
        # Total states: 32768 * 15 * 2 = ~1M. Feasible.
        
        full_mask = (1 << num_paths) - 1
        dp = [[[False for _ in range(2)] for _ in range(num_paths)] for _ in range(1 << num_paths)]
        parent = [[[(-1, -1) for _ in range(2)] for _ in range(num_paths)] for _ in range(1 << num_paths)]
        
        # Initialize
        for i in range(num_paths):
            dp[1 << i][i][0] = True
            dp[1 << i][i][1] = True
            
        # Transition
        for mask in range(1, 1 << num_paths):
            for last_idx in range(num_paths):
                for last_orient in range(2):
                    if not dp[mask][last_idx][last_orient]:
                        continue
                    
                    # Try to add next path j with orientation o2
                    end_node = get_end(last_idx, last_orient)
                    
                    for j in range(num_paths):
                        if mask & (1 << j):
                            continue
                        for o2 in range(2):
                            start_node = get_start(j, o2)
                            if end_node in adj[start_node]:
                                new_mask = mask | (1 << j)
                                dp[new_mask][j][o2] = True
                                parent[new_mask][j][o2] = (last_idx, last_orient)
                                
        # Check if full_mask is reachable
        for last_idx in range(num_paths):
            for last_orient in range(2):
                if dp[full_mask][last_idx][last_orient]:
                    # Reconstruct
                    path_segments = []
                    curr_mask = full_mask
                    curr_idx = last_idx
                    curr_orient = last_orient
                    while curr_mask != 0:
                        path_segments.append((curr_idx, curr_orient))
                        if curr_mask == (1 << curr_idx):
                            break
                        prev_idx, prev_orient = parent[curr_mask][curr_idx][curr_orient]
                        curr_mask ^= (1 << curr_idx)
                        curr_idx = prev_idx
                        curr_orient = prev_orient
                    path_segments.reverse()
                    
                    result_path = []
                    for idx, orient in path_segments:
                        if orient == 0:
                            result_path.extend(paths[idx])
                        else:
                            result_path.extend(reversed(paths[idx]))
                    return result_path
        
    # Fallback: Return the longest path
    longest = max(paths, key=len)
    return longest

# EVOLVE-BLOCK-END
