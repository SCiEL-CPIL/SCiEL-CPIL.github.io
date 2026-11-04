import numpy as np
import itertools
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
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
    
    # Build conflict matrix: conflict if intersection size >= 3
    # Two supports conflict if they share 3 or more indices
    # We need to find maximum set of supports with pairwise intersection <= 2
    
    # For efficiency, use bit masks
    support_masks = []
    for s in supports:
        mask = 0
        for idx in s:
            mask |= (1 << idx)
        support_masks.append(mask)
    
    # Build conflict matrix
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i+1, N_S):
            inter = support_masks[i] & support_masks[j]
            if inter.bit_count() >= 3:
                conflict[i, j] = True
                conflict[j, i] = True
    
    # Now we need to find the maximum independent set in this conflict graph
    # This is equivalent to finding the maximum clique in the complement graph
    
    # Use a branch-and-bound approach with pivot-based coloring (Tomita algorithm)
    # But first, let's try a greedy approach to get a good initial solution
    
    # Greedy: pick supports that conflict with fewest already-picked supports
    in_set = set()
    remaining = set(range(N_S))
    
    while remaining:
        if time.time() > deadline - 30:
            break
        # Pick the vertex with minimum degree in the remaining subgraph
        best_v = None
        best_deg = float('inf')
        for v in remaining:
            deg = sum(1 for u in remaining if conflict[v, u])
            if deg < best_deg:
                best_deg = deg
                best_v = v
        in_set.add(best_v)
        remaining.remove(best_v)
        # Remove all neighbors of best_v from remaining
        to_remove = [u for u in remaining if conflict[best_v, u]]
        for u in to_remove:
            remaining.remove(u)
    
    best_size = len(in_set)
    best_set = list(in_set)
    
    # Now try to improve with local search / simulated annealing
    # or use a more sophisticated branch-and-bound
    
    # Let's implement a proper branch-and-bound for maximum independent set
    # using bitsets for efficiency
    
    # Convert conflict to adjacency lists
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(i+1, N_S):
            if conflict[i, j]:
                adj[i].add(j)
                adj[j].add(i)
    
    # Branch and bound with greedy coloring upper bound
    sys.setrecursionlimit(10000)
    
    def bb_mis(vertices, current_set, current_size, best_size, best_set, time_limit):
        n = len(vertices)
        if n == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = list(current_set)
            return
        
        if time.time() > time_limit:
            return
        
        # Upper bound: use a simple greedy coloring on the induced subgraph
        # Color vertices greedily; number of colors is an upper bound on MIS size
        color_classes = []
        colored = [False] * n
        v_list = list(vertices)
        
        # Order by degree (descending) for better coloring
        degs = [len([u for u in vertices if conflict[v_list[i], u]]) for i in range(n)]
        order = sorted(range(n), key=lambda i: -degs[i])
        
        for idx in order:
            if colored[idx]:
                continue
            # Find all uncolored vertices not adjacent to v_list[idx]
            v = v_list[idx]
            non_adj = [j for j in range(n) if not colored[j] and not conflict[v, v_list[j]]]
            color_class = [idx]
            colored[idx] = True
            for j in non_adj:
                if colored[j]:
                    continue
                # Check if j is non-adjacent to all in current color_class
                ok = True
                for k in color_class:
                    if conflict[v_list[j], v_list[k]]:
                        ok = False
                        break
                if ok:
                    color_class.append(j)
                    colored[j] = True
            color_classes.append(color_class)
        
        upper_bound = current_size + len(color_classes)
        if upper_bound <= best_size[0]:
            return
        
        # Pick vertex with highest degree in the subgraph
        v_idx = order[0]
        v = v_list[v_idx]
        
        # Branch 1: include v
        new_set = list(current_set) + [v]
        # Remaining: vertices not adjacent to v and not v itself
        remaining = [u for u in vertices if u != v and not conflict[v, u]]
        bb_mis(remaining, new_set, current_size + 1, best_size, best_set, time_limit)
        
        # Branch 2: exclude v
        remaining2 = [u for u in vertices if u != v]
        bb_mis(remaining2, current_set, current_size, best_size, best_set, time_limit)
    
    time_limit = deadline - 20
    best_size_arr = [best_size]
    best_set_arr = [best_set]
    
    all_vertices = list(range(N_S))
    bb_mis(all_vertices, [], 0, best_size_arr, best_set_arr, time_limit)
    
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
    
    # Generate ALL Type D vectors for greedy fill
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
            
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)
    
    # Greedy fill using vectorized dot product checks
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))
    
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
            
        # Check compatibility: for kissing number, we need |v - u| >= max_norm
        # max_norm is the maximum norm among all vectors
        # For Type D vectors, norm^2 = 4, so norm = 2
        # For Type A vectors, norm^2 = 4, so norm = 2
        # So max_norm = 2, and we need |v - u|^2 >= 4
        # |v - u|^2 = |v|^2 + |u|^2 - 2*v.u = 4 + 4 - 2*v.u = 8 - 2*v.u
        # We need 8 - 2*v.u >= 4, i.e., v.u <= 2
        
        if len(current_set_vectors) > 0:
            dots = current_arr @ v
            if np.any(dots > 2):
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
        
        # Check against Type A: dot <= 1 (since |v_B|^2=2, |v_A|^2=4, need 2+4-2*v.u >= 4 => v.u <= 1)
        if np.any(dots_BA[i] > 1):
            continue
        
        # Check against Type D: dot <= 1 (since |v_B|^2=2, |v_D|^2=4, need 2+4-2*v.u >= 4 => v.u <= 1)
        if np.any(dots_BD[i] > 1):
            continue
        
        # Check against other Type B: dot = 0 (since |v_B|^2=2, need 2+2-2*v.u >= 4 => v.u <= 0)
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
