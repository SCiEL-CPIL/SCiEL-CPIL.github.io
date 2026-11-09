import time
import math
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin for 90s limit
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 1
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    nv = len(valid_nodes)
    
    if nv == 0:
        return [1]
        
    # Build conflict graph
    conflicts = [0] * nv
    for i_idx in range(nv):
        v = valid_nodes[i_idx]
        for i_j in range(i_idx + 1, nv):
            u = valid_nodes[i_j]
            if v + u in squares:
                conflicts[i_idx] |= (1 << i_j)
                conflicts[i_j] |= (1 << i_idx)
                
    def popcount(x):
        return bin(x).count('1')
        
    def mask_to_list(mask):
        res = []
        for i in range(nv):
            if mask & (1 << i):
                res.append(valid_nodes[i])
        return res
        
    # Check if conflict graph is bipartite using BFS
    def is_bipartite_and_get_partition():
        color = [-1] * nv
        for start in range(nv):
            if color[start] != -1:
                continue
            color[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                # Get neighbors from conflicts bitmask
                temp = conflicts[u]
                while temp:
                    r_bit = temp & (-temp)
                    v_idx = r_bit.bit_length() - 1
                    if color[v_idx] == -1:
                        color[v_idx] = color[u] ^ 1
                        queue.append(v_idx)
                    elif color[v_idx] == color[u]:
                        return None  # Not bipartite
                    temp ^= r_bit
        return color
        
    # Dinic's max flow algorithm
    def dinic_max_flow(n, source, sink, edges):
        # edges: list of (u, v, capacity)
        # Build adjacency list with forward and reverse edges
        adj = [[] for _ in range(n)]
        for u, v, cap in edges:
            adj[u].append([v, cap, len(adj[v])])
            adj[v].append([u, 0, len(adj[u]) - 1])
        
        flow = 0
        INF = float('inf')
        
        while True:
            # BFS to find level graph
            level = [-1] * n
            level[source] = 0
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for v, cap, rev in adj[u]:
                    if cap > 0 and level[v] == -1:
                        level[v] = level[u] + 1
                        queue.append(v)
                        
            if level[sink] == -1:
                break
                
            # DFS to find blocking flow
            ptr = [0] * n
            
            def dfs(u, f):
                if u == sink:
                    return f
                while ptr[u] < len(adj[u]):
                    v, cap, rev = adj[u][ptr[u]]
                    if cap > 0 and level[v] == level[u] + 1:
                        pushed = dfs(v, min(f, cap))
                        if pushed > 0:
                            adj[u][ptr[u]][1] -= pushed
                            adj[v][rev][1] += pushed
                            return pushed
                    ptr[u] += 1
                return 0
                
            while True:
                pushed = dfs(source, INF)
                if pushed == 0:
                    break
                flow += pushed
                
        return flow, adj
    
    # Try exact solution via max-flow if bipartite
    best_mask = 0
    best_count = 0
    
    color = is_bipartite_and_get_partition()
    if color is not None:
        # Graph is bipartite! Solve MIS exactly via min-cut
        # Left set: nodes with color 0, Right set: nodes with color 1
        left_nodes = [i for i in range(nv) if color[i] == 0]
        right_nodes = [i for i in range(nv) if color[i] == 1]
        
        # Create mapping: left nodes get indices 0..len(left)-1
        # right nodes get indices len(left)..len(left)+len(right)-1
        # source = len(left)+len(right), sink = len(left)+len(right)+1
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        n_total = n_left + n_right + 2
        source = n_left + n_right
        sink = n_left + n_right + 1
        
        left_idx = {node: i for i, node in enumerate(left_nodes)}
        right_idx = {node: n_left + i for i, node in enumerate(right_nodes)}
        
        # Build edges for max-flow
        # Source -> left nodes (capacity 1)
        # Left -> right (capacity INF for conflicts)
        # Right -> sink (capacity 1)
        edges = []
        for node in left_nodes:
            edges.append((source, left_idx[node], 1))
        for node in right_nodes:
            edges.append((right_idx[node], sink, 1))
        
        # Add conflict edges with infinite capacity
        INF_CAP = nv + 1
        for node in left_nodes:
            temp = conflicts[node]
            while temp:
                r_bit = temp & (-temp)
                v_idx = r_bit.bit_length() - 1
                if color[v_idx] == 1:  # v is in right set
                    edges.append((left_idx[node], right_idx[v_idx], INF_CAP))
                temp ^= r_bit
        
        # Compute max flow
        max_flow, _ = dinic_max_flow(n_total, source, sink, edges)
        
        # The minimum vertex cover size = max flow
        # Maximum independent set size = nv - min vertex cover
        mis_size = nv - max_flow
        
        # To find the actual MIS, we need to identify which nodes are in the minimum vertex cover
        # Rebuild and track which edges are saturated
        # For now, use the size and fall back to heuristic to find actual set
        # Or we can do a more careful reconstruction
        
        # Actually, let's just use the heuristic to find a set of that size
        # The max flow gives us the optimal size, so we know what to aim for
        
        # Use local search to try to find a set of size mis_size
        # Start with greedy
        order = sorted(range(nv), key=lambda i: popcount(conflicts[i]))
        mask = 0
        for idx in order:
            if (conflicts[idx] & mask) == 0:
                mask |= (1 << idx)
        
        # Local search to improve
        improved = True
        iters = 0
        target = mis_size
        while improved and iters < 5000 and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # Try adding each non-chosen node
            for i in range(nv):
                if mask & (1 << i):
                    continue
                if (conflicts[i] & mask) == 0:
                    mask |= (1 << i)
                    improved = True
                    
            # Try 1-for-1 swaps
            for i in range(nv):
                if mask & (1 << i):
                    continue
                conf_in_mask = conflicts[i] & mask
                if conf_in_mask == 0:
                    mask |= (1 << i)
                    improved = True
                    continue
                temp = conf_in_mask
                while temp:
                    r_bit = temp & (-temp)
                    r_idx = r_bit.bit_length() - 1
                    if (conflicts[i] & (mask ^ r_bit)) == 0:
                        mask = (mask & ~r_bit) | (1 << i)
                        improved = True
                        break
                    temp ^= r_bit
                        
            # Try 2-for-1 swaps
            if iters % 5 == 0:
                for i in range(nv):
                    if mask & (1 << i):
                        continue
                    conf_in_mask = conflicts[i] & mask
                    if conf_in_mask == 0:
                        mask |= (1 << i)
                        improved = True
                        continue
                    conf_list = []
                    temp = conf_in_mask
                    while temp:
                        r_bit = temp & (-temp)
                        r_idx = r_bit.bit_length() - 1
                        conf_list.append(r_idx)
                        temp ^= r_bit
                        
                    for a_idx in range(len(conf_list)):
                        for b_idx in range(a_idx + 1, len(conf_list)):
                            a = conf_list[a_idx]
                            b = conf_list[b_idx]
                            a_bit = 1 << a
                            b_bit = 1 << b
                            new_mask = mask & ~a_bit & ~b_bit
                            if (conflicts[i] & new_mask) == 0:
                                mask = new_mask | (1 << i)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                        
        c = popcount(mask)
        if c > best_count:
            best_count = c
            best_mask = mask
            
        # If we reached the optimal size, we're done
        if best_count >= target:
            result = mask_to_list(best_mask)
            res_set = set(result)
            for a in res_set:
                if 2 * a in squares:
                    res_set.remove(a)
                    continue
                for b in res_set:
                    if a <= b and (a + b) in squares:
                        res_set.remove(b)
                        break
            if not res_set:
                return [valid_nodes[0]]
            return list(res_set)
    
    # Fallback: heuristic approach
    if time.time() < deadline - 10:
        # 1. Greedy by degree (ascending)
        order = sorted(range(nv), key=lambda i: popcount(conflicts[i]))
        mask = 0
        for idx in order:
            if (conflicts[idx] & mask) == 0:
                mask |= (1 << idx)
        
        # Local search
        improved = True
        iters = 0
        while improved and iters < 2000 and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            for i in range(nv):
                if mask & (1 << i):
                    continue
                if (conflicts[i] & mask) == 0:
                    mask |= (1 << i)
                    improved = True
                    
            for i in range(nv):
                if mask & (1 << i):
                    continue
                conf_in_mask = conflicts[i] & mask
                if conf_in_mask == 0:
                    mask |= (1 << i)
                    improved = True
                    continue
                temp = conf_in_mask
                while temp:
                    r_bit = temp & (-temp)
                    r_idx = r_bit.bit_length() - 1
                    if (conflicts[i] & (mask ^ r_bit)) == 0:
                        mask = (mask & ~r_bit) | (1 << i)
                        improved = True
                        break
                    temp ^= r_bit
                        
            if iters % 10 == 0:
                for i in range(nv):
                    if mask & (1 << i):
                        continue
                    conf_in_mask = conflicts[i] & mask
                    if conf_in_mask == 0:
                        mask |= (1 << i)
                        improved = True
                        continue
                    conf_list = []
                    temp = conf_in_mask
                    while temp:
                        r_bit = temp & (-temp)
                        r_idx = r_bit.bit_length() - 1
                        conf_list.append(r_idx)
                        temp ^= r_bit
                        
                    for a_idx in range(len(conf_list)):
                        for b_idx in range(a_idx + 1, len(conf_list)):
                            a = conf_list[a_idx]
                            b = conf_list[b_idx]
                            a_bit = 1 << a
                            b_bit = 1 << b
                            new_mask = mask & ~a_bit & ~b_bit
                            if (conflicts[i] & new_mask) == 0:
                                mask = new_mask | (1 << i)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                        
        c = popcount(mask)
        if c > best_count:
            best_count = c
            best_mask = mask
            
        # 2. Random restarts
        import random
        random.seed(42)
        while time.time() < deadline - 3:
            order = list(range(nv))
            random.shuffle(order)
            mask = 0
            for idx in order:
                if (conflicts[idx] & mask) == 0:
                    mask |= (1 << idx)
            
            # Shorter local search
            improved = True
            iters = 0
            while improved and iters < 500 and time.time() < deadline - 2:
                improved = False
                iters += 1
                
                for i in range(nv):
                    if mask & (1 << i):
                        continue
                    if (conflicts[i] & mask) == 0:
                        mask |= (1 << i)
                        improved = True
                        
                for i in range(nv):
                    if mask & (1 << i):
                        continue
                    conf_in_mask = conflicts[i] & mask
                    if conf_in_mask == 0:
                        mask |= (1 << i)
                        improved = True
                        continue
                    temp = conf_in_mask
                    while temp:
                        r_bit = temp & (-temp)
                        r_idx = r_bit.bit_length() - 1
                        if (conflicts[i] & (mask ^ r_bit)) == 0:
                            mask = (mask & ~r_bit) | (1 << i)
                            improved = True
                            break
                        temp ^= r_bit
                            
            c = popcount(mask)
            if c > best_count:
                best_count = c
                best_mask = mask
    
    result = mask_to_list(best_mask)
    
    # Verification
    res_set = set(result)
    for a in list(res_set):
        if 2 * a in squares:
            res_set.remove(a)
            continue
        for b in list(res_set):
            if a <= b and (a + b) in squares:
                res_set.remove(b)
                break
                
    if not res_set:
        return [valid_nodes[0]]
        
    return list(res_set)
