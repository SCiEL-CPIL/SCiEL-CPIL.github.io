import math
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # 1. Precompute squares
    max_val = 256
    max_sum = 512
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)
    
    def is_square(n):
        return n in squares

    # 2. Identify valid nodes (2v is not a square)
    valid_nodes = []
    for v in range(1, max_val + 1):
        if not is_square(2 * v):
            valid_nodes.append(v)
    
    if not valid_nodes:
        return [1] # Fallback, though 1 is valid (2 is not square)

    # 3. Build conflict graph
    # conflict[u] = set of v such that u+v is square
    conflict = defaultdict(set)
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    n_nodes = len(valid_nodes)
    
    for i, u in enumerate(valid_nodes):
        for j, v in enumerate(valid_nodes):
            if i >= j:
                continue
            if is_square(u + v):
                conflict[u].add(v)
                conflict[v].add(u)
    
    # 4. Check if bipartite
    # Color: 0, 1
    color = {v: -1 for v in valid_nodes}
    is_bipartite = True
    partitions = [[], []]
    
    for start_node in valid_nodes:
        if color[start_node] != -1:
            continue
        color[start_node] = 0
        partitions[0].append(start_node)
        queue = deque([start_node])
        while queue:
            u = queue.popleft()
            for v in conflict[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    partitions[color[v]].append(v)
                    queue.append(v)
                elif color[v] == color[u]:
                    is_bipartite = False
                    break
            if not is_bipartite:
                break
        if not is_bipartite:
            break

    best_set = set()
    
    if is_bipartite:
        # Solve via Max Flow (Dinic's Algorithm)
        # Network:
        # Source S -> u (cap 1) for u in Part 0
        # v -> Sink T (cap 1) for v in Part 1
        # u -> v (cap INF) for edge (u,v) with u in Part 0, v in Part 1
        
        INF = 10**9
        # Node mapping for flow network
        # 0: S
        # 1..N: Valid Nodes
        # N+1: T
        
        S = 0
        T = n_nodes + 1
        total_nodes = n_nodes + 2
        
        # Adjacency list for Dinic
        # edge: [to, cap, rev_index]
        graph = [[] for _ in range(total_nodes)]
        
        def add_edge(u, v, cap):
            graph[u].append([v, cap, len(graph[v])])
            graph[v].append([u, 0, len(graph[u]) - 1])
            
        # Add source/sink edges
        for v in partitions[0]:
            idx = node_index[v] + 1
            add_edge(S, idx, 1)
        for v in partitions[1]:
            idx = node_index[v] + 1
            add_edge(idx, T, 1)
            
        # Add conflict edges
        for u in partitions[0]:
            idx_u = node_index[u] + 1
            for v in conflict[u]:
                if v in partitions[1]: # Should be true if bipartite and u in 0
                    idx_v = node_index[v] + 1
                    add_edge(idx_u, idx_v, INF)
                    
        # Dinic's Max Flow
        level = [-1] * total_nodes
        ptr = [0] * total_nodes
        
        def bfs():
            for i in range(total_nodes):
                level[i] = -1
            queue = deque([S])
            level[S] = 0
            while queue:
                u = queue.popleft()
                for edge in graph[u]:
                    v, cap, rev = edge
                    if cap > 0 and level[v] == -1:
                        level[v] = level[u] + 1
                        queue.append(v)
            return level[T] != -1
            
        def dfs(u, f):
            if u == T:
                return f
            while ptr[u] < len(graph[u]):
                edge = graph[u][ptr[u]]
                v, cap, rev = edge
                if cap > 0 and level[v] == level[u] + 1:
                    pushed = dfs(v, min(f, cap))
                    if pushed > 0:
                        edge[1] -= pushed
                        graph[v][rev][1] += pushed
                        return pushed
                ptr[u] += 1
            return 0
            
        max_flow = 0
        while bfs():
            for i in range(total_nodes):
                ptr[i] = 0
            while True:
                pushed = dfs(S, INF)
                if pushed == 0:
                    break
                max_flow += pushed
                
        # Recover MIS
        # Reachable nodes from S in residual graph
        reachable = [False] * total_nodes
        queue = deque([S])
        reachable[S] = True
        while queue:
            u = queue.popleft()
            for edge in graph[u]:
                v, cap, rev = edge
                if cap > 0 and not reachable[v]:
                    reachable[v] = True
                    queue.append(v)
                    
        mis_nodes = set()
        for v in partitions[0]:
            idx = node_index[v] + 1
            if reachable[idx]:
                mis_nodes.add(v)
        for v in partitions[1]:
            idx = node_index[v] + 1
            if not reachable[idx]:
                mis_nodes.add(v)
                
        best_set = mis_nodes
        
    else:
        # Fallback: Local Search
        # Greedy initialization
        current_set = set()
        # Sort by degree (conflict count) ascending to pick "safe" nodes first?
        # Or just random/greedy by value.
        nodes_sorted = sorted(valid_nodes, key=lambda x: len(conflict[x]))
        for v in nodes_sorted:
            # Check if v conflicts with any in current_set
            if not any(u in current_set for u in conflict[v]):
                current_set.add(v)
                
        best_set = current_set
        best_size = len(best_set)
        
        # Local Search: Swap moves
        # Try to add a node not in set by removing conflicting nodes in set
        # Or swap: remove u, add v if it increases size or maintains size with better future potential
        # Simple hill climbing: try adding any node, if conflict, remove all conflicts.
        # If len(new_set) > len(old_set), accept.
        
        # To make it faster, we can just run a few iterations of "try to add, resolve conflicts"
        # But since we have time, we can do more.
        
        # A better heuristic for MIS:
        # 1. Start with greedy.
        # 2. Iteratively try to add a node $x \notin S$.
        # 3. Let $C_x$ be conflicts of $x$ in $S$.
        # 4. If $|C_x| < 1$, add $x$ (gain 1).
        # 5. If $|C_x| = 1$, removing 1 to add 1 is neutral, but might open up more.
        # 6. If $|C_x| > 1$, likely bad.
        
        # Let's implement a simple swap-based local search.
        # Pick a node $u \in S$ and a node $v \notin S$.
        # Replace $u$ with $v$ if valid and potentially better?
        # Actually, just trying to add nodes is good.
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Shuffle to avoid local minima traps
            candidates = [v for v in valid_nodes if v not in best_set]
            random.shuffle(candidates)
            
            for v in candidates:
                conflicts_in_set = [u for u in conflict[v] if u in best_set]
                # If we remove all conflicts and add v, does it improve?
                # Gain = 1 - len(conflicts_in_set)
                if 1 - len(conflicts_in_set) > 0:
                    # Always beneficial
                    for u in conflicts_in_set:
                        best_set.remove(u)
                    best_set.add(v)
                    improved = True
                elif 1 - len(conflicts_in_set) == 0:
                    # Neutral, but might help later. Accept with some probability or always?
                    # Let's accept to explore.
                    for u in conflicts_in_set:
                        best_set.remove(u)
                    best_set.add(v)
                    improved = True
                    
    return list(best_set)
