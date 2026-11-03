import numpy as np
import time
import itertools
from collections import deque

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit

    # 1. Generate Type A vectors (one non-zero entry +/- 2)
    type_a_vectors = []
    for i in range(D):
        for sign in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = sign
            type_a_vectors.append(v)
    type_a_vectors = np.array(type_a_vectors)

    # 2. Generate Type B vectors (four non-zero entries +/- 1)
    type_b_vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            type_b_vectors.append(v)
    type_b_vectors = np.array(type_b_vectors)

    n_b = len(type_b_vectors)

    # 3. Compute incompatibility graph for Type B
    # Two vectors conflict if their dot product >= 3 (distance < max_norm)
    dot_products = type_b_vectors @ type_b_vectors.T
    adjacency = (dot_products >= 3)
    np.fill_diagonal(adjacency, False)

    # 4. Check if the conflict graph is bipartite using coordinate sum parity
    # For weight-4 vectors with entries in {+1,-1}, the sum of coordinates is even
    # Two vectors with same parity (both even sum or both odd sum) can have dot product >= 3
    # Actually, let's use a different bipartition: 
    # Partition by the sign pattern's "weight" or use a 2-coloring if possible
    
    # Let's try: partition by whether the first non-zero coordinate is +1 or -1
    # This doesn't guarantee bipartiteness, so let's do BFS-based 2-coloring
    
    # Build adjacency list
    adj_list = [np.where(adjacency[i])[0] for i in range(n_b)]
    
    # Try to 2-color the graph (check bipartiteness)
    color = np.full(n_b, -1, dtype=np.int8)
    is_bipartite = True
    for start in range(n_b):
        if color[start] != -1:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj_list[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    is_bipartite = False
                    break
        if not is_bipartite:
            break

    if is_bipartite:
        # Use max-flow to find maximum independent set in bipartite graph
        # Left set: color 0, Right set: color 1
        left_set = np.where(color == 0)[0]
        right_set = np.where(color == 1)[0]
        
        # Build flow network:
        # Source -> left nodes (capacity 1)
        # Left -> Right (capacity 1) if edge exists
        # Right -> Sink (capacity 1)
        
        n_left = len(left_set)
        n_right = len(right_set)
        source = 0
        sink = n_left + n_right + 1
        total_nodes = sink + 1
        
        # Map original indices to flow node indices
        left_map = np.full(n_b, -1, dtype=np.int64)
        for i, orig in enumerate(left_set):
            left_map[orig] = i + 1
        
        right_map = np.full(n_b, -1, dtype=np.int64)
        for i, orig in enumerate(right_set):
            right_map[orig] = n_left + i + 1
        
        # Build adjacency for flow network using dictionary of lists
        # Each edge: [to, capacity, reverse_edge_index]
        flow_graph = [[] for _ in range(total_nodes)]
        
        def add_edge(u, v, cap):
            flow_graph[u].append([v, cap, len(flow_graph[v])])
            flow_graph[v].append([u, 0, len(flow_graph[u]) - 1])
        
        # Add source to left
        for i in range(n_left):
            add_edge(source, i + 1, 1)
        
        # Add right to sink
        for i in range(n_right):
            add_edge(n_left + i + 1, sink, 1)
        
        # Add edges between left and right where conflict exists
        for u_orig in left_set:
            u_flow = left_map[u_orig]
            for v_orig in adj_list[u_orig]:
                if color[v_orig] == 1:
                    v_flow = right_map[v_orig]
                    add_edge(u_flow, v_flow, 1)
        
        # Run Ford-Fulkerson with BFS (Edmonds-Karp)
        def bfs_max_flow():
            max_flow_val = 0
            while True:
                # BFS to find augmenting path
                parent = [-1] * total_nodes
                parent_edge = [-1] * total_nodes
                visited = [False] * total_nodes
                visited[source] = True
                queue = deque([source])
                
                while queue and not visited[sink]:
                    u = queue.popleft()
                    for idx, (v, cap, rev_idx) in enumerate(flow_graph[u]):
                        if not visited[v] and cap > 0:
                            visited[v] = True
                            parent[v] = u
                            parent_edge[v] = idx
                            queue.append(v)
                
                if not visited[sink]:
                    break
                
                # Find bottleneck
                path_flow = float('inf')
                v = sink
                while v != source:
                    u = parent[v]
                    idx = parent_edge[v]
                    path_flow = min(path_flow, flow_graph[u][idx][1])
                    v = u
                
                # Update residual capacities
                v = sink
                while v != source:
                    u = parent[v]
                    idx = parent_edge[v]
                    flow_graph[u][idx][1] -= path_flow
                    rev_idx = flow_graph[u][idx][2]
                    flow_graph[v][rev_idx][1] += path_flow
                    v = u
                
                max_flow_val += path_flow
            
            return max_flow_val
        
        if time.time() > deadline - 10:
            # Fallback to greedy if running out of time
            best_mis_indices = []
            available = np.ones(n_b, dtype=bool)
            while np.any(available):
                if time.time() > deadline - 5:
                    break
                avail_indices = np.where(available)[0]
                if len(avail_indices) == 0:
                    break
                chosen = avail_indices[0]
                best_mis_indices.append(int(chosen))
                available[chosen] = False
                available[adjacency[chosen]] = False
        else:
            min_cut_size = bfs_max_flow()
            
            # Find vertices reachable from source in residual graph
            reachable = [False] * total_nodes
            reachable[source] = True
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for v, cap, rev_idx in flow_graph[u]:
                    if not reachable[v] and cap > 0:
                        reachable[v] = True
                        queue.append(v)
            
            # Maximum independent set in bipartite graph:
            # MIS = (Left \ reachable) ∪ (Right ∩ reachable)
            mis_original = []
            for i, orig in enumerate(left_set):
                if not reachable[i + 1]:
                    mis_original.append(int(orig))
            for i, orig in enumerate(right_set):
                if reachable[n_left + i + 1]:
                    mis_original.append(int(orig))
            
            best_mis_indices = mis_original
    else:
        # Graph is not bipartite, use greedy heuristic
        best_mis_indices = []
        available = np.ones(n_b, dtype=bool)
        degrees = np.sum(adjacency, axis=1).astype(np.int64)
        while np.any(available):
            if time.time() > deadline - 5:
                break
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            avail_degrees = degrees[avail_indices]
            min_deg_idx_in_avail = np.argmin(avail_degrees)
            chosen = avail_indices[min_deg_idx_in_avail]
            best_mis_indices.append(int(chosen))
            available[chosen] = False
            available[adjacency[chosen]] = False
            degrees[adjacency[chosen]] -= 1

    # Convert selected indices to vectors
    selected_b_vectors = type_b_vectors[best_mis_indices]

    # Combine with Type A
    all_vectors = np.vstack([type_a_vectors, selected_b_vectors])

    return all_vectors
