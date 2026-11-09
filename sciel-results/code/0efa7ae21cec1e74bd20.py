import time
import math
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for safety
    
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
    
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    node_set = set(valid_nodes)
    nv = len(valid_nodes)
    
    # Build conflict graph using bitsets for speed
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

    # --- Exact MIS via Max Flow if Bipartite ---
    def is_bipartite_and_get_partition():
        color = [-1] * nv
        partition = [0] * nv
        for i in range(nv):
            if color[i] == -1:
                color[i] = 0
                partition[i] = 0
                q = deque([i])
                while q:
                    u = q.popleft()
                    mask = conflicts[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        mask ^= lsb
                        if color[v] == -1:
                            color[v] = 1 - color[u]
                            partition[v] = color[v]
                            q.append(v)
                        elif color[v] == color[u]:
                            return None, None
        return partition, color

    def solve_mis_flow():
        if time.time() > deadline - 5:
            return None
            
        partition, _ = is_bipartite_and_get_partition()
        if partition is None:
            return None
            
        source = nv
        sink = nv + 1
        n_nodes = nv + 2
        graph = [[] for _ in range(n_nodes)]
        
        def add_edge(u, v, cap):
            graph[u].append([v, cap, len(graph[v])])
            graph[v].append([u, 0, len(graph[u]) - 1])
            
        for i in range(nv):
            if partition[i] == 0:
                add_edge(source, i, 1)
            else:
                add_edge(i, sink, 1)
                
        INF_CAP = 10**9
        for i in range(nv):
            if partition[i] == 0:
                mask = conflicts[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    mask ^= lsb
                    if partition[j] == 1:
                        add_edge(i, j, INF_CAP)
                        
        level = [0] * n_nodes
        it = [0] * n_nodes
        
        def bfs_flow():
            for i in range(n_nodes):
                level[i] = -1
            level[source] = 0
            q = deque([source])
            while q:
                u = q.popleft()
                for v, cap, rev in graph[u]:
                    if cap > 0 and level[v] < 0:
                        level[v] = level[u] + 1
                        q.append(v)
            return level[sink] >= 0
            
        def dfs_flow(u, f):
            if u == sink:
                return f
            for i in range(it[u], len(graph[u])):
                it[u] = i
                v, cap, rev = graph[u][i]
                if cap > 0 and level[u] < level[v]:
                    d = dfs_flow(v, min(f, cap))
                    if d > 0:
                        graph[u][i][1] -= d
                        graph[v][rev][1] += d
                        return d
            return 0
            
        while bfs_flow():
            for i in range(n_nodes):
                it[i] = 0
            while True:
                f = dfs_flow(source, 10**9)
                if f == 0:
                    break
                    
        reachable = [False] * n_nodes
        q = deque([source])
        reachable[source] = True
        while q:
            u = q.popleft()
            for v, cap, rev in graph[u]:
                if cap > 0 and not reachable[v]:
                    reachable[v] = True
                    q.append(v)
                    
        result_nodes = []
        for i in range(nv):
            if partition[i] == 0:
                if reachable[i]:
                    result_nodes.append(valid_nodes[i])
            else:
                if not reachable[i]:
                    result_nodes.append(valid_nodes[i])
                    
        res_set = set(result_nodes)
        for a in res_set:
            if 2 * a in squares:
                return None
            for b in res_set:
                if a <= b and (a + b) in squares:
                    return None
                    
        return result_nodes

    exact_result = solve_mis_flow()
    if exact_result:
        return exact_result
        
    # --- Fallback: Simulated Annealing with Local Search Polish ---
    
    def greedy_init():
        # Greedy by degree (ascending)
        order = sorted(range(nv), key=lambda i: popcount(conflicts[i]))
        mask = 0
        for idx in order:
            if (conflicts[idx] & mask) == 0:
                mask |= (1 << idx)
        return mask

    def local_search_polish(mask, time_limit=2.0):
        """Exhaustive local search: try all single adds, removes, and pairwise swaps."""
        best_mask = mask
        best_size = popcount(mask)
        deadline_ls = time.time() + time_limit
        
        improved = True
        while improved and time.time() < deadline_ls:
            improved = False
            current_size = popcount(best_mask)
            
            # Try adding each node not in set
            for u in range(nv):
                if not (best_mask & (1 << u)):
                    if (conflicts[u] & best_mask) == 0:
                        new_mask = best_mask | (1 << u)
                        new_size = popcount(new_mask)
                        if new_size > best_size:
                            best_mask = new_mask
                            best_size = new_size
                            improved = True
            
            # Try removing each node in set (if it helps via subsequent adds)
            # For now, just try swaps: remove one, add one
            in_set_indices = []
            m = best_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                in_set_indices.append(idx)
                m ^= lsb
                
            for remove_idx in in_set_indices:
                if time.time() >= deadline_ls:
                    break
                mask_without = best_mask & ~(1 << remove_idx)
                # Try adding each node not in mask_without
                for add_idx in range(nv):
                    if not (mask_without & (1 << add_idx)) and add_idx != remove_idx:
                        if (conflicts[add_idx] & mask_without) == 0:
                            new_mask = mask_without | (1 << add_idx)
                            new_size = popcount(new_mask)
                            if new_size > best_size:
                                best_mask = new_mask
                                best_size = new_size
                                improved = True
        
        return best_mask

    def simulated_annealing(initial_mask, time_limit=5.0):
        mask = initial_mask
        best_mask = initial_mask
        best_size = popcount(mask)
        
        T = 2.0
        cooling = 0.9995
        min_T = 0.005
        
        nodes = list(range(nv))
        
        while T > min_T:
            if time.time() > deadline - 2:
                break
                
            # Random move
            u = random.choice(nodes)
            in_set = bool(mask & (1 << u))
            
            if in_set:
                # Remove u
                new_mask = mask & ~(1 << u)
                new_size = popcount(new_mask)
                current_size = popcount(mask)
                delta = new_size - current_size
                if delta >= 0 or random.random() < math.exp(delta / T):
                    mask = new_mask
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = new_mask
            else:
                # Try to add u
                conflict_mask = conflicts[u] & mask
                if conflict_mask == 0:
                    # Can add directly
                    new_mask = mask | (1 << u)
                    new_size = popcount(new_mask)
                    current_size = popcount(mask)
                    delta = new_size - current_size
                    if delta >= 0 or random.random() < math.exp(delta / T):
                        mask = new_mask
                        if new_size > best_size:
                            best_size = new_size
                            best_mask = new_mask
                else:
                    # Conflict exists. Try swap strategies.
                    pc = popcount(conflict_mask)
                    if pc == 1:
                        # Single conflict: swap
                        c_idx = conflict_mask.bit_length() - 1
                        new_mask = (mask & ~(1 << c_idx)) | (1 << u)
                        new_size = popcount(new_mask)
                        current_size = popcount(mask)
                        delta = new_size - current_size
                        if delta >= 0 or random.random() < math.exp(delta / T):
                            mask = new_mask
                            if new_size > best_size:
                                best_size = new_size
                                best_mask = new_mask
                    elif pc <= 4:
                        # Multi-conflict: remove all conflicts and add u
                        new_mask = (mask & ~conflict_mask) | (1 << u)
                        new_size = popcount(new_mask)
                        current_size = popcount(mask)
                        delta = new_size - current_size
                        if delta >= 0 or random.random() < math.exp(delta / T):
                            mask = new_mask
                            if new_size > best_size:
                                best_size = new_size
                                best_mask = new_mask
                    else:
                        # High conflict: try removing a random subset of conflicts
                        # Pick up to 3 random conflicting nodes to remove
                        cm = conflict_mask
                        to_remove = 0
                        count = 0
                        while cm and count < 3:
                            lsb = cm & -cm
                            if random.random() < 0.7:
                                to_remove |= lsb
                                count += 1
                            cm ^= lsb
                        if to_remove:
                            new_mask = (mask & ~to_remove) | (1 << u)
                            new_size = popcount(new_mask)
                            current_size = popcount(mask)
                            delta = new_size - current_size
                            if delta >= 0 or random.random() < math.exp(delta / T):
                                mask = new_mask
                                if new_size > best_size:
                                    best_size = new_size
                                    best_mask = new_mask
                            
            T *= cooling
            
        return best_mask

    best_mask = 0
    best_count = 0
    
    random.seed(42)
    
    # 1. Greedy Start + SA + Polish
    m = greedy_init()
    m = simulated_annealing(m)
    m = local_search_polish(m, time_limit=1.0)
    c = popcount(m)
    if c > best_count:
        best_count = c
        best_mask = m
        
    # 2. Multiple Random Starts with SA + Polish
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        # Random greedy with different orderings
        order = list(range(nv))
        random.shuffle(order)
        m = 0
        for idx in order:
            if (conflicts[idx] & m) == 0:
                m |= (1 << idx)
        m = simulated_annealing(m, time_limit=3.0)
        m = local_search_polish(m, time_limit=1.0)
        c = popcount(m)
        if c > best_count:
            best_count = c
            best_mask = m
            
    # 3. Final Polish on Best
    if time.time() < deadline - 3:
        m = local_search_polish(best_mask, time_limit=2.0)
        c = popcount(m)
        if c > best_count:
            best_count = c
            best_mask = m

    # Convert to list
    result = []
    for i in range(nv):
        if best_mask & (1 << i):
            result.append(valid_nodes[i])
            
    if not result:
        result = [valid_nodes[0]]
        
    return result
