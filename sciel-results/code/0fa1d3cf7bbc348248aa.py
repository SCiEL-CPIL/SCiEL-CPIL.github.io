import time
import math
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 256
    
    # 1. Filter nodes: 2*v must not be a perfect square
    valid_nodes = []
    for v in range(1, N + 1):
        s = 2 * v
        r = int(math.isqrt(s))
        if r * r != s:
            valid_nodes.append(v)
    
    if not valid_nodes:
        return [1]
        
    # Map values to indices for faster lookup
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    M = len(valid_nodes)
    
    # 2. Build adjacency list
    adj = [set() for _ in range(M)]
    
    for i in range(M):
        vi = valid_nodes[i]
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            s = vi + vj
            r = int(math.isqrt(s))
            if r * r == s:
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. Check if graph is bipartite via BFS 2-coloring
    color = [-1] * M
    is_bipartite = True
    for start in range(M):
        if color[start] == -1:
            color[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        is_bipartite = False
                        break
                if not is_bipartite:
                    break
        if not is_bipartite:
            break
    
    if is_bipartite:
        # Use max-flow to find exact Maximum Independent Set
        left_nodes = [i for i in range(M) if color[i] == 0]
        right_nodes = [i for i in range(M) if color[i] == 1]
        
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        total_flow_nodes = 2 + n_left + n_right
        source = 0
        sink = total_flow_nodes - 1
        
        left_flow_idx = {left_nodes[i]: i + 1 for i in range(n_left)}
        right_flow_idx = {right_nodes[i]: n_left + i + 1 for i in range(n_right)}
        
        flow_graph = [[] for _ in range(total_flow_nodes)]
        
        def add_edge(u, v, cap):
            flow_graph[u].append([v, cap, len(flow_graph[v])])
            flow_graph[v].append([u, 0, len(flow_graph[u]) - 1])
        
        for i in range(n_left):
            add_edge(source, i + 1, 1)
        
        for i in range(n_right):
            add_edge(n_left + i + 1, sink, 1)
        
        INF = float('inf')
        for i in left_nodes:
            for j in adj[i]:
                if color[j] == 1:
                    add_edge(left_flow_idx[i], right_flow_idx[j], INF)
        
        def dinic_max_flow():
            max_flow = 0
            while True:
                level = [-1] * total_flow_nodes
                level[source] = 0
                q = deque([source])
                while q:
                    u = q.popleft()
                    for e in flow_graph[u]:
                        v, cap, _ = e
                        if cap > 0 and level[v] == -1:
                            level[v] = level[u] + 1
                            q.append(v)
                
                if level[sink] == -1:
                    break
                
                ptr = [0] * total_flow_nodes
                
                def dfs(u, pushed):
                    if pushed == 0:
                        return 0
                    if u == sink:
                        return pushed
                    
                    while ptr[u] < len(flow_graph[u]):
                        e = flow_graph[u][ptr[u]]
                        v, cap, rev_idx = e
                        if level[v] == level[u] + 1 and cap > 0:
                            tr = dfs(v, min(pushed, cap))
                            if tr > 0:
                                e[1] -= tr
                                flow_graph[v][rev_idx][1] += tr
                                return tr
                        ptr[u] += 1
                    
                    return 0
                
                while True:
                    pushed = dfs(source, float('inf'))
                    if pushed == 0:
                        break
                    max_flow += pushed
            
            return max_flow
        
        min_vertex_cover_size = dinic_max_flow()
        
        visited = [False] * total_flow_nodes
        visited[source] = True
        q = deque([source])
        while q:
            u = q.popleft()
            for e in flow_graph[u]:
                v, cap, _ = e
                if cap > 0 and not visited[v]:
                    visited[v] = True
                    q.append(v)
        
        in_vertex_cover = set()
        for i in left_nodes:
            if not visited[left_flow_idx[i]]:
                in_vertex_cover.add(i)
        for i in right_nodes:
            if visited[right_flow_idx[i]]:
                in_vertex_cover.add(i)
        
        independent_set_indices = [i for i in range(M) if i not in in_vertex_cover]
        
        result = [valid_nodes[i] for i in independent_set_indices]
        
        if not result:
            result = [valid_nodes[0]]
        
        return result
    
    else:
        # Graph is not bipartite: use simulated annealing on the independent set
        # Build conflict bitmask for fast validity checks
        # M can be up to ~250, so use Python ints as bitsets
        
        # Precompute: for each node, which other nodes it conflicts with
        # adj[i] is already a set of conflicting node indices
        
        # Simulated annealing approach:
        # State: a set of selected nodes (independent set)
        # Moves: 
        #   1. Add a node that doesn't conflict with current set
        #   2. Remove a node from current set
        #   3. Swap: remove one node, add another (may temporarily violate, but we check)
        # Actually for independent set, we need to maintain validity.
        # Better moves:
        #   1. Add a valid node (increases size by 1) - always accept
        #   2. Remove a node (decreases size by 1) - accept with probability based on temperature
        #   3. Swap: remove node i, add node j where j doesn't conflict with (set - {i})
        #      This keeps size same but may enable future additions
        
        # Start with greedy minimum-degree heuristic
        remaining = set(range(M))
        current_set = set()
        
        while remaining and time.time() < deadline - 5:
            min_deg_node = None
            min_deg = float('inf')
            for u in remaining:
                deg = len(adj[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_deg_node = u
            if min_deg_node is None:
                break
            current_set.add(min_deg_node)
            remaining.discard(min_deg_node)
            for v in adj[min_deg_node]:
                remaining.discard(v)
        
        # Precompute for each node, the set of nodes it conflicts with
        # Already have adj[i]
        
        # Simulated annealing
        best_set = set(current_set)
        best_size = len(current_set)
        
        T_initial = 10.0
        T_final = 0.01
        cooling_rate = 0.999
        T = T_initial
        rng = random.Random(42)
        
        # Precompute: for each node not in current_set, check if it can be added
        # We'll track conflicts dynamically
        
        # Efficient move generation:
        # For "add" move: pick a node from (all nodes - current_set) that has no conflict with current_set
        # For "swap" move: pick i in current_set, j not in current_set, check if j conflicts with (current_set - {i})
        
        # To speed up, precompute conflict masks as Python integers
        # M <= ~250, so this is fine
        
        # Actually, let's use a different approach:
        # Maintain a list of "available" nodes (not in set, no conflict with set)
        # When we add a node, remove its neighbors from available
        # When we remove a node, add its non-neighbor neighbors back to available
        
        all_nodes = set(range(M))
        
        # available = nodes not in current_set that don't conflict with current_set
        available = set()
        for node in all_nodes - current_set:
            if not (adj[node] & current_set):
                available.add(node)
        
        # Simulated annealing loop
        sa_steps = 0
        max_sa_steps = 500000
        
        while T > T_final and sa_steps < max_sa_steps and time.time() < deadline - 3:
            sa_steps += 1
            current_size = len(current_set)
            
            # Choose move type
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try to add a node from available
                if available:
                    node_to_add = rng.choice(list(available))
                    # Adding always improves size, so accept
                    current_set.add(node_to_add)
                    # Update available: remove node_to_add and its neighbors
                    available.discard(node_to_add)
                    for neighbor in adj[node_to_add]:
                        available.discard(neighbor)
                    # Check if this is a new best
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                else:
                    # No available nodes, try a swap
                    if current_set:
                        node_to_remove = rng.choice(list(current_set))
                        # Find nodes that could replace it
                        # A node j can replace i if j doesn't conflict with (current_set - {i})
                        candidates = []
                        for j in all_nodes - current_set:
                            if not (adj[j] & (current_set - {node_to_remove})):
                                candidates.append(j)
                        if candidates:
                            node_to_add = rng.choice(candidates)
                            current_set.discard(node_to_remove)
                            current_set.add(node_to_add)
                            # Update available
                            available = set()
                            for node in all_nodes - current_set:
                                if not (adj[node] & current_set):
                                    available.add(node)
                        else:
                            # Can't swap, just remove
                            current_set.discard(node_to_remove)
                            available = set()
                            for node in all_nodes - current_set:
                                if not (adj[node] & current_set):
                                    available.add(node)
            elif move_type < 0.7:
                # Try a swap: remove one, add another (same size)
                if current_set:
                    node_to_remove = rng.choice(list(current_set))
                    # Find candidates that don't conflict with (current_set - {node_to_remove})
                    candidates = []
                    for j in all_nodes - current_set:
                        if not (adj[j] & (current_set - {node_to_remove})):
                            candidates.append(j)
                    if candidates:
                        node_to_add = rng.choice(candidates)
                        current_set.discard(node_to_remove)
                        current_set.add(node_to_add)
                        # Update available
                        available = set()
                        for node in all_nodes - current_set:
                            if not (adj[node] & current_set):
                                available.add(node)
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
            else:
                # Remove a node (to explore)
                if current_set:
                    node_to_remove = rng.choice(list(current_set))
                    current_set.discard(node_to_remove)
                    # Update available
                    available = set()
                    for node in all_nodes - current_set:
                        if not (adj[node] & current_set):
                            available.add(node)
            
            # Cool down
            T *= cooling_rate
        
        # Final greedy fill: add any node that doesn't conflict
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for node in all_nodes - best_set:
                if not (adj[node] & best_set):
                    best_set.add(node)
                    changed = True
        
        # Deterministic verification
        result_indices = list(best_set)
        # Verify no pair sums to a perfect square
        valid = True
        for i in range(len(result_indices)):
            a = valid_nodes[result_indices[i]]
            # Check diagonal: a + a
            s_diag = 2 * a
            r_diag = int(math.isqrt(s_diag))
            if r_diag * r_diag == s_diag:
                valid = False
                break
            for j in range(i + 1, len(result_indices)):
                b = valid_nodes[result_indices[j]]
                s = a + b
                r = int(math.isqrt(s))
                if r * r == s:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to a simple valid set
            result_indices = []
            for i in range(M):
                node = valid_nodes[i]
                # Check if node can be added
                can_add = True
                # Check diagonal
                s_diag = 2 * node
                r_diag = int(math.isqrt(s_diag))
                if r_diag * r_diag == s_diag:
                    continue
                for existing in result_indices:
                    s = node + valid_nodes[existing]
                    r = int(math.isqrt(s))
                    if r * r == s:
                        can_add = False
                        break
                if can_add:
                    result_indices.append(i)
        
        result = [valid_nodes[i] for i in result_indices]
        
        if not result:
            result = [valid_nodes[0]]
        
        return result
