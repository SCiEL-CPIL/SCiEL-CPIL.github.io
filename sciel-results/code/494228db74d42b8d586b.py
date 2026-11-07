"""Flow-based search for clique subdivisions using greedy set extension."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    
    def flow_oracle(branches):
        """
        Use max-flow to check if there exist vertex-disjoint paths between all pairs of branches.
        Returns list of paths or None.
        
        Construction:
        - Each original vertex v becomes two nodes: v_in and v_out, with edge v_in -> v_out capacity 1.
        - Each graph edge (u,v) becomes edges u_out -> v_in and v_out -> u_in with capacity 1.
        - For each pair (i,j) of branches, we need a path from branch_i to branch_j.
        - We model this as: super_source connects to each branch_in with capacity 1,
          each branch_out connects to super_sink with capacity 1.
        - But we need |C(k,2)| units of flow, one for each pair.
        - Actually, each branch needs to send (k-1) units out and receive (k-1) units in.
        
        Better approach: 
        - For each branch vertex b, split into b_in and b_out.
        - Edge b_in -> b_out has capacity (k-1) where k = len(branches).
        - For non-branch vertices v, edge v_in -> v_out has capacity 1.
        - Graph edges: u_out -> v_in and v_out -> u_in, capacity 1.
        - Super source S connects to each branch_in with capacity (k-1).
        - Each branch_out connects to super sink T with capacity (k-1).
        - Total flow needed: C(k,2) * 2 / 2 = C(k,2)... wait.
        
        Actually, each pair (i,j) needs one path from i to j and one from j to i? No, just one path.
        So we need C(k,2) paths total.
        
        Let me reconsider. We want to route C(k,2) units of flow.
        Each branch i must send (k-1) units (one to each other branch) and receive (k-1) units.
        
        So:
        - S -> branch_i_in: capacity (k-1) for each branch i
        - branch_i_out -> T: capacity (k-1) for each branch i
        - branch_i_in -> branch_i_out: capacity (k-1)
        - non-branch v_in -> v_out: capacity 1
        - u_out -> v_in: capacity 1 for each graph edge (u,v)
        
        Total flow = C(k,2). If max flow equals C(k,2), we can extract paths.
        """
        k = len(branches)
        if k < 2:
            return None
        
        num_pairs = k * (k - 1) // 2
        
        # Node mapping:
        # For each vertex v (0..n-1): v_in = 2*v, v_out = 2*v + 1
        # S = 2*n, T = 2*n + 1
        S = 2 * n
        T = 2 * n + 1
        total_nodes = 2 * n + 2
        
        # Build flow network using adjacency list for residual graph
        # We'll use a simple max-flow implementation (Edmonds-Karp or Dinic)
        # For small graphs, Edmonds-Karp is fine.
        
        # capacity[u][v] = residual capacity
        cap = [[0] * total_nodes for _ in range(total_nodes)]
        
        # Add edges: v_in -> v_out
        branch_set = set(branches)
        for v in range(n):
            vin = 2 * v
            vout = 2 * v + 1
            if v in branch_set:
                cap[vin][vout] = k - 1
            else:
                cap[vin][vout] = 1
        
        # Add graph edges: u_out -> v_in and v_out -> u_in
        for u, v in edges:
            uout = 2 * u + 1
            vin = 2 * v
            vout = 2 * v + 1
            uin = 2 * u
            cap[uout][vin] = 1
            cap[vout][uin] = 1
        
        # Add S and T connections
        for b in branches:
            bin_ = 2 * b
            bout = 2 * b + 1
            cap[S][bin_] = k - 1
            cap[bout][T] = k - 1
        
        # Max flow using Edmonds-Karp (BFS for augmenting paths)
        def max_flow():
            flow = 0
            while True:
                # BFS to find augmenting path
                parent = [-1] * total_nodes
                visited = [False] * total_nodes
                queue = deque([S])
                visited[S] = True
                found = False
                while queue:
                    u = queue.popleft()
                    if u == T:
                        found = True
                        break
                    for v in range(total_nodes):
                        if not visited[v] and cap[u][v] > 0:
                            visited[v] = True
                            parent[v] = u
                            queue.append(v)
                if not found:
                    break
                # Find bottleneck
                min_cap = float('inf')
                v = T
                while v != S:
                    u = parent[v]
                    min_cap = min(min_cap, cap[u][v])
                    v = u
                # Update residual
                v = T
                while v != S:
                    u = parent[v]
                    cap[u][v] -= min_cap
                    cap[v][u] += min_cap
                    v = u
                flow += min_cap
            return flow
        
        total_flow = max_flow()
        
        if total_flow < num_pairs:
            return None
        
        # Extract paths from the flow
        # For each pair of branches, we need to find the path.
        # The flow network has flow on edges. We need to trace paths.
        
        # For each branch pair (i, j), find a path from branch_i to branch_j in the flow.
        # We can do this by looking at which edges carry flow.
        
        # Build a directed graph of edges with positive flow
        # flow_edge[u][v] = cap_original[u][v] - cap_residual[u][v]
        # But we modified cap in-place. Let's track flow differently.
        
        # Actually, let's rebuild: we know the original capacities.
        # flow(u,v) = original_cap(u,v) - residual_cap(u,v)
        
        # Let me redo this more carefully.
        # I'll store the flow explicitly.
        
        # Let me restart the flow computation but track flows.
        
        # Actually, let me just extract paths by DFS in the residual graph where we follow edges with flow > 0.
        
        # For each branch b, look at outgoing edges from b_out that have flow.
        # The flow on edge (u_out, v_in) tells us how many paths go from u to v through that edge.
        
        # This is getting complex. Let me use a different extraction method.
        
        # For each pair (i, j) of branches, do a DFS from branch_i_in following edges with flow > 0 to reach branch_j_in.
        
        # First, compute flow on each edge:
        # flow(u, v) = original_cap(u,v) - cap(u,v)  [since cap is now residual]
        
        # But we need original capacities. Let me store them.
        
        # Let me redo the whole thing more carefully.
        
        return None  # Placeholder - need to implement path extraction
    
    # The flow oracle is complex. Let me use a simpler but effective approach:
    # Greedy set extension with BFS-based path finding, but with smart ordering.
    
    def find_disjoint_paths(branches):
        """
        Try to find vertex-disjoint paths for all pairs in branches.
        Uses BFS with multiple attempts and different orderings.
        Returns list of paths or None.
        """
        k = len(branches)
        if k < 2:
            return None
        
        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                pairs.append((i, j))
        
        num_pairs = len(pairs)
        
        # Try multiple strategies
        for strategy in range(10):
            trial_paths = {}
            trial_used = set(branches)
            success = True
            
            # Order pairs
            pair_indices = list(range(num_pairs))
            if strategy == 0:
                # Default order
                pass
            elif strategy == 1:
                # Random order
                random.shuffle(pair_indices)
            elif strategy == 2:
                # Sort by distance (shortest first)
                def pair_dist(idx):
                    i, j = pairs[idx]
                    return _shortest_dist(branches[i], branches[j], set(branches), adj)
                pair_indices.sort(key=pair_dist)
            elif strategy == 3:
                # Sort by distance (longest first)
                def pair_dist(idx):
                    i, j = pairs[idx]
                    return _shortest_dist(branches[i], branches[j], set(branches), adj)
                pair_indices.sort(key=pair_dist, reverse=True)
            else:
                # Other random orders
                random.shuffle(pair_indices)
            
            for idx in pair_indices:
                i, j = pairs[idx]
                b1 = branches[i]
                b2 = branches[j]
                
                path = _bfs_path(b1, b2, trial_used, adj)
                if path is None:
                    success = False
                    break
                
                for v in path[1:-1]:
                    if v in trial_used:
                        success = False
                        break
                    trial_used.add(v)
                
                if not success:
                    break
                
                trial_paths[(i, j)] = path
            
            if success:
                return [trial_paths[pairs[idx]] for idx in range(num_pairs)]
        
        return None
    
    def _bfs_path(start, end, forbidden, adj):
        """Find a simple path from start to end avoiding forbidden vertices (except start, end)."""
        if start == end:
            return [start]
        
        queue = deque()
        queue.append((start, [start]))
        visited = {start}
        
        while queue:
            node, path = queue.popleft()
            if node == end:
                return path
            for neighbor in adj[node]:
                if neighbor in visited:
                    continue
                if neighbor in forbidden and neighbor not in (start, end):
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def _shortest_dist(s, t, forbidden, adj):
        """Shortest path distance avoiding forbidden (except s, t). Returns inf if no path."""
        if s == t:
            return 0
        dist = {s: 0}
        queue = deque([s])
        while queue:
            node = queue.popleft()
            if node == t:
                return dist[node]
            for neighbor in adj[node]:
                if neighbor in dist:
                    continue
                if neighbor in forbidden and neighbor not in (s, t):
                    continue
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
        return float('inf')
    
    # Greedy set extension:
    # Start with a seed set, try to add vertices one by one.
    
    def greedy_extend(seed):
        """Try to extend a seed set by adding vertices one at a time."""
        current = list(seed)
        current_paths = find_disjoint_paths(current)
        if current_paths is None:
            return None
        
        remaining = [v for v in range(n) if v not in set(current)]
        # Sort remaining by degree descending
        remaining.sort(key=lambda v: degrees[v], reverse=True)
        
        for v in remaining:
            trial = current + [v]
            paths = find_disjoint_paths(trial)
            if paths is not None:
                current = trial
                current_paths = paths
        
        return current, current_paths
    
    # Generate seeds and try to extend
    best_result = None
    best_k = 1
    
    # Seeds: pairs of adjacent vertices, sorted by combined degree
    pairs_seed = []
    for u, v in edges:
        pairs_seed.append(([u, v], degrees[u] + degrees[v]))
    pairs_seed.sort(key=lambda x: x[1], reverse=True)
    
    # Also try single high-degree vertices as seeds (then extend)
    # Actually, we need at least 2 branches, so seeds should be at least size 2.
    
    # Try extending from top pairs
    start_time = time.time()
    timeout = 55
    
    for seed, _ in pairs_seed:
        if time.time() - start_time > timeout:
            break
        
        result = greedy_extend(seed)
        if result is not None:
            branches, paths = result
            k = len(branches)
            if k > best_k:
                best_k = k
                best_result = (branches, paths)
    
    # Also try random seeds
    for _ in range(500):
        if time.time() - start_time > timeout:
            break
        k_seed = random.randint(2, 4)
        seed = random.sample(range(n), k_seed)
        result = greedy_extend(seed)
        if result is not None:
            branches, paths = result
            k = len(branches)
            if k > best_k:
                best_k = k
                best_result = (branches, paths)
    
    if best_result is not None:
        branches, paths = best_result
        return {'branches': [int(b) for b in branches], 'paths': [[int(v) for v in p] for p in paths]}
    
    # Fallback
    u, v = edges[0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

# EVOLVE-BLOCK-END
