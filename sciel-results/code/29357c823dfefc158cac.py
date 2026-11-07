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
    
    def check_clique_subdivision(branches):
        """
        Check if 'branches' form a clique subdivision using max-flow.
        Returns (True, paths_list) if successful, (False, None) otherwise.
        """
        k = len(branches)
        if k < 2:
            return False, None
        
        num_pairs = k * (k - 1) // 2
        
        # Node indices in flow network:
        # v_in = 2*v, v_out = 2*v+1
        # S = 2*n, T = 2*n+1
        S = 2 * n
        T = 2 * n + 1
        total_nodes = 2 * n + 2
        
        # Capacities
        cap = [[0] * total_nodes for _ in range(total_nodes)]
        
        branch_set = set(branches)
        
        # 1. Vertex splitting edges
        for v in range(n):
            vin = 2 * v
            vout = 2 * v + 1
            if v in branch_set:
                cap[vin][vout] = k - 1
            else:
                cap[vin][vout] = 1
        
        # 2. Graph edges
        for u, v in edges:
            uout = 2 * u + 1
            vin = 2 * v
            vout = 2 * v + 1
            uin = 2 * u
            cap[uout][vin] = 1
            cap[vout][uin] = 1
        
        # 3. Source and Sink
        for b in branches:
            bin_ = 2 * b
            bout = 2 * b + 1
            cap[S][bin_] = k - 1
            cap[bout][T] = k - 1
        
        # Max Flow (Edmonds-Karp)
        def max_flow():
            flow = 0
            while True:
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
                min_cap = float('inf')
                v = T
                while v != S:
                    u = parent[v]
                    min_cap = min(min_cap, cap[u][v])
                    v = u
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
            return False, None
        
        # Extract paths
        # Rebuild original capacities to compute flow
        orig_cap = [[0] * total_nodes for _ in range(total_nodes)]
        for v in range(n):
            vin = 2 * v
            vout = 2 * v + 1
            if v in branch_set:
                orig_cap[vin][vout] = k - 1
            else:
                orig_cap[vin][vout] = 1
        
        for u, v in edges:
            uout = 2 * u + 1
            vin = 2 * v
            vout = 2 * v + 1
            uin = 2 * u
            orig_cap[uout][vin] = 1
            orig_cap[vout][uin] = 1
            
        for b in branches:
            bin_ = 2 * b
            bout = 2 * b + 1
            orig_cap[S][bin_] = k - 1
            orig_cap[bout][T] = k - 1
            
        # Compute flow matrix
        flow_mat = [[0] * total_nodes for _ in range(total_nodes)]
        for u in range(total_nodes):
            for v in range(total_nodes):
                if orig_cap[u][v] > 0:
                    flow_mat[u][v] = orig_cap[u][v] - cap[u][v]
        
        # Extract paths for each pair (i, j) with i < j
        paths_list = []
        
        for i in range(k):
            for j in range(i + 1, k):
                b_i = branches[i]
                b_j = branches[j]
                b_i_out = 2 * b_i + 1
                b_j_in = 2 * b_j
                
                # Find path from b_i_out to b_j_in in the flow graph
                visited = set()
                queue = deque([(b_i_out, [b_i])]) # (flow_node, path_orig)
                visited.add(b_i_out)
                found_path = None
                
                while queue:
                    node, path_orig = queue.popleft()
                    if node == b_j_in:
                        found_path = path_orig + [b_j]
                        break
                    # Explore neighbors in flow graph
                    for v in range(total_nodes):
                        if flow_mat[node][v] > 0 and v not in visited:
                            visited.add(v)
                            # Map flow node to original vertex
                            if v < 2 * n:
                                orig_v = v // 2
                                # Determine if we add orig_v to path
                                # If node is x_out and v is y_in, we add y.
                                # If node is x_in and v is x_out, we do not add x.
                                if node % 2 == 1: # node is out
                                    queue.append((v, path_orig + [orig_v]))
                                else: # node is in
                                    queue.append((v, path_orig))
                            
                if found_path:
                    paths_list.append(found_path)
                else:
                    return False, None
            
        return True, paths_list

    # Greedy extension
    best_k = 0
    best_result = None
    
    # Sort vertices by degree to prioritize high-degree vertices as branches
    degrees = [len(adj[i]) for i in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    start_time = time.time()
    timeout = 55
    
    # Try random subsets of increasing size
    # Or just try to extend from random pairs.
    
    # Let's try to find a clique of size k by random sampling of k vertices
    # and checking if they form a clique subdivision.
    
    # Since we want to MAXIMIZE k, we should try large k first?
    # No, if k is too large, it's unlikely.
    # Let's try k from 2 to 8.
    
    # Actually, a greedy extension from a good seed is more likely to find a local maximum.
    # Let's try multiple seeds.
    
    seeds = []
    # High degree pairs
    for u, v in edges:
        if degrees[u] + degrees[v] > 10:
            seeds.append([u, v])
            
    # Random pairs
    for _ in range(100):
        u, v = random.sample(range(n), 2)
        seeds.append([u, v])
        
    # Deduplicate
    unique_seeds = []
    seen = set()
    for s in seeds:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_seeds.append(s)
            
    for seed in unique_seeds:
        if time.time() - start_time > timeout:
            break
            
        current = list(seed)
        # Check if seed is valid
        valid, _ = check_clique_subdivision(current)
        if not valid:
            continue
            
        # Extend
        remaining = [v for v in sorted_verts if v not in set(current)]
        for v in remaining:
            if time.time() - start_time > timeout:
                break
            trial = current + [v]
            valid, paths = check_clique_subdivision(trial)
            if valid:
                current = trial
                if len(current) > best_k:
                    best_k = len(current)
                    best_result = (current, paths)
                    
    # Also try some random larger seeds
    for _ in range(200):
        if time.time() - start_time > timeout:
            break
        k_seed = random.randint(3, 5)
        seed = random.sample(range(n), k_seed)
        valid, paths = check_clique_subdivision(seed)
        if valid:
            current = list(seed)
            if len(current) > best_k:
                best_k = len(current)
                best_result = (current, paths)
            
            # Extend
            remaining = [v for v in sorted_verts if v not in set(current)]
            for v in remaining:
                if time.time() - start_time > timeout:
                    break
                trial = current + [v]
                valid, paths = check_clique_subdivision(trial)
                if valid:
                    current = trial
                    if len(current) > best_k:
                        best_k = len(current)
                        best_result = (current, paths)

    if best_result is not None:
        branches, paths = best_result
        # Format output
        out_paths = [ [int(v) for v in p] for p in paths ]
        
        return {'branches': [int(b) for b in branches], 'paths': out_paths}
    
    # Fallback: return a valid small clique
    # Find any edge
    u, v = edges[0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

# EVOLVE-BLOCK-END
