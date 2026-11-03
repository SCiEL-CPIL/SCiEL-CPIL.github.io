import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    D = 11
    
    # Type A: Vectors with one non-zero entry +/- 2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
        
    vectors_A = np.array(vectors_A, dtype=np.int64)
    
    # Type D: Vectors with four non-zero entries +/- 1
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)
    
    # Build conflict graph vectorized
    # Represent each support as a bitmask
    support_masks = np.zeros(N_S, dtype=np.uint64)
    for i, s in enumerate(supports):
        mask = 0
        for idx in s:
            mask |= (1 << idx)
        support_masks[i] = mask
    
    # Conflict if intersection size >= 3
    # Precompute popcount for 16-bit chunks
    popcount_16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.uint8)
    
    # Build conflict matrix
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        ands = support_masks[i] & support_masks
        low = (ands & 0xFFFF).astype(np.int64)
        high = ((ands >> 16) & 0xFFFF).astype(np.int64)
        inter_counts = popcount_16[low] + popcount_16[high]
        conflict[i, :] = inter_counts >= 3
        conflict[:, i] = inter_counts >= 3
    
    # Symmetrize and remove self-loops
    conflict = np.maximum(conflict, conflict.T)
    np.fill_diagonal(conflict, False)
    
    # Branch-and-bound for Maximum Independent Set
    # Using greedy coloring upper bound
    
    def greedy_coloring(vertices, adj):
        """
        Greedy coloring of the induced subgraph on 'vertices'.
        Returns a list of color classes (each a set of vertices with same color).
        """
        n = len(vertices)
        if n == 0:
            return []
        
        color_classes = []
        colored = np.zeros(n, dtype=bool)
        
        # Order by degree within the subgraph (descending)
        degrees = np.array([np.sum(adj[vertices[i], vertices]) for i in range(n)], dtype=np.int64)
        order = np.argsort(-degrees)
        
        for idx in order:
            if colored[idx]:
                continue
            # Find all uncolored vertices not adjacent to idx
            v = vertices[idx]
            non_adj = np.where(~colored & ~adj[v, vertices])[0]
            # Among non_adj, find those not adjacent to each other (greedy)
            color_class = [idx]
            colored[idx] = True
            for j in non_adj:
                if colored[j]:
                    continue
                # Check if j is non-adjacent to all in current color_class
                if not any(adj[vertices[j], vertices[k]] for k in color_class):
                    color_class.append(j)
                    colored[j] = True
            color_classes.append(color_class)
        
        return color_classes
    
    def branch_and_bound(vertices, adj, current_size, best_size, best_set, time_limit):
        """
        Branch-and-bound for MIS.
        vertices: array of vertex indices in the current subgraph
        adj: full adjacency matrix (bool)
        current_size: size of current independent set
        best_size: best size found so far
        best_set: best set found so far
        time_limit: absolute time limit
        """
        n = len(vertices)
        
        if n == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = list(current_set)
            return
        
        # Upper bound: current_size + greedy coloring bound
        color_classes = greedy_coloring(vertices, adj)
        upper_bound = current_size + len(color_classes)
        
        if upper_bound <= best_size[0]:
            return
        
        # Choose vertex with highest degree in subgraph
        degrees = np.array([np.sum(adj[vertices[i], vertices]) for i in range(n)], dtype=np.int64)
        v_idx = np.argmax(degrees)
        v = vertices[v_idx]
        
        # Branch 1: include v
        new_set = current_set + [v]
        # Remove v and its neighbors from vertices
        neighbors_mask = adj[v, vertices]
        remaining = vertices[~neighbors_mask]
        
        if time.time() < time_limit:
            branch_and_bound(remaining, adj, current_size + 1, best_size, best_set, time_limit)
        
        # Branch 2: exclude v
        remaining2 = np.delete(vertices, v_idx)
        
        if time.time() < time_limit:
            branch_and_bound(remaining2, adj, current_size, best_size, best_set, time_limit)
    
    # Use a simpler, more efficient branch-and-bound
    # with bitset representation for speed
    
    # Convert conflict to adjacency list
    adj_list = [np.where(conflict[i])[0] for i in range(N_S)]
    
    # Greedy initial solution
    in_set = np.zeros(N_S, dtype=bool)
    for v in range(N_S):
        if not np.any(conflict[v, in_set]):
            in_set[v] = True
    
    best_size = np.sum(in_set)
    best_set = np.where(in_set)[0].tolist()
    
    # Branch and bound with tighter bounds
    def bb_mis(vertices, adj_mat, current_set, current_size, best_size, best_set, time_limit):
        n = len(vertices)
        
        if n == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = list(current_set)
            return
        
        # Upper bound via greedy coloring
        color_classes = greedy_coloring(vertices, adj_mat)
        upper_bound = current_size + len(color_classes)
        
        if upper_bound <= best_size[0]:
            return
        
        if time.time() > time_limit:
            return
        
        # Pick vertex with max degree
        degrees = np.array([np.sum(adj_mat[vertices[i], vertices]) for i in range(n)], dtype=np.int64)
        v_idx = np.argmax(degrees)
        v = vertices[v_idx]
        
        # Branch: include v
        new_set = list(current_set) + [v]
        # Remaining vertices: those not adjacent to v and not v itself
        not_adj = np.where(~adj_mat[v, vertices])[0]
        remaining = vertices[not_adj]
        
        bb_mis(remaining, adj_mat, new_set, current_size + 1, best_size, best_set, time_limit)
        
        # Branch: exclude v
        remaining2 = np.delete(vertices, v_idx)
        bb_mis(remaining2, adj_mat, current_set, current_size, best_size, best_set, time_limit)
    
    # Run B&B with time limit
    time_limit = deadline - 30
    best_size_arr = [best_size]
    best_set_arr = [best_set]
    
    # Start from all vertices
    all_vertices = np.arange(N_S)
    bb_mis(all_vertices, conflict, [], 0, best_size_arr, best_set_arr, time_limit)
    
    final_support_set = best_set_arr[0]
    
    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in final_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
            
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)
    
    # Generate ALL Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            all_vectors_D.append(v)
            
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)
    
    # Greedy fill using vectorized dot product checks
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))
    
    # Precompute matrix for current set
    current_arr = np.array(current_set_vectors) if current_set_vectors else np.zeros((0, D))
    
    indices_all = np.arange(N_all_D)
    rng = np.random.RandomState(42)
    rng.shuffle(indices_all)
    
    for i in indices_all:
        if time.time() > deadline - 15:
            break
            
        v = all_vectors_D[i]
        
        if tuple(v) in current_set_vectors_set:
            continue
            
        # Check compatibility: dot(v, u) <= 2 for all u in set
        if len(current_set_vectors) > 0:
            dots = current_arr @ v
            if np.any(dots >= 3):
                continue
        
        current_set_vectors.append(v)
        current_set_vectors_set.add(tuple(v))
        current_arr = np.vstack([current_arr, v.reshape(1, -1)])
        
    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    
    # Try adding Type B vectors (two non-zero ±1, norm²=2)
    vectors_B_candidates = []
    for i in range(D):
        for j in range(i+1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    vectors_B_candidates.append(v)
    
    vectors_B_candidates = np.array(vectors_B_candidates, dtype=np.int64)
    N_B = len(vectors_B_candidates)
    
    added_B = []
    added_B_set = set()
    
    # Precompute dots of B candidates with A and D vectors
    dots_BA = vectors_B_candidates @ vectors_A.T  # (N_B, 22)
    dots_BD = vectors_B_candidates @ final_vectors_D.T  # (N_B, len(D))
    
    for i in range(N_B):
        if time.time() > deadline - 5:
            break
        
        vt = tuple(vectors_B_candidates[i])
        if vt in added_B_set:
            continue
        
        # Check against Type A: dot <= 1
        if np.any(dots_BA[i] >= 2):
            continue
        
        # Check against Type D: dot <= 1
        if np.any(dots_BD[i] >= 2):
            continue
        
        # Check against other Type B: dot = 0
        if added_B:
            added_B_arr = np.array(added_B)
            dots_with_added = added_B_arr @ vectors_B_candidates[i]
            if np.any(dots_with_added != 0):
                continue
        
        added_B.append(vectors_B_candidates[i])
        added_B_set.add(vt)
    
    if added_B:
        added_B = np.array(added_B, dtype=np.int64)
        final_vectors = np.vstack([vectors_A, final_vectors_D, added_B])
    else:
        final_vectors = np.vstack([vectors_A, final_vectors_D])
    
    return final_vectors
