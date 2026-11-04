import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Generate all 4-subsets of {0, ..., 10}
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330
    
    # Convert supports to bitmasks for fast intersection
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
    
    # Build conflict graph: conflict if intersection size is exactly 3
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Convert to list and boolean matrix for vectorized ops
    adj_list = [list(s) for s in adj]
    adj_bool = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in adj_list[i]:
            adj_bool[i, j] = True
    
    # Ensure symmetry
    adj_bool = adj_bool | adj_bool.T
    
    rng = np.random.RandomState(42)
    
    def greedy_min_degree():
        """Greedy min-degree heuristic for independent set."""
        remaining = set(range(N_S))
        s = set()
        
        while remaining:
            min_deg = float('inf')
            min_node = None
            for v in remaining:
                deg = sum(1 for nb in adj_list[v] if nb in remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_node = v
            s.add(min_node)
            to_remove = {min_node} | {nb for nb in adj_list[min_node] if nb in remaining}
            remaining -= to_remove
        
        return s
    
    def local_search(init_set, max_iter=300):
        """Local search with 1-add and 1-swap moves."""
        current_set = set(init_set)
        best_local = set(current_set)
        best_local_size = len(current_set)
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Compute blocked using vectorized ops
            in_set = np.zeros(N_S, dtype=bool)
            for v in current_set:
                in_set[v] = True
            
            if current_set:
                blocked = np.any(adj_bool[in_set], axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            # Candidates: not in set and not blocked
            candidates_mask = ~(in_set | blocked)
            cand_indices = np.where(candidates_mask)[0]
            
            if len(cand_indices) > 0:
                # Try to add a random candidate
                v_add = int(cand_indices[rng.randint(len(cand_indices))])
                current_set.add(v_add)
                if len(current_set) > best_local_size:
                    best_local_size = len(current_set)
                    best_local = set(current_set)
                continue
            
            # Stuck: try 1-swap
            if current_set:
                v_remove = list(current_set)[rng.randint(len(current_set))]
                current_set.remove(v_remove)
                
                # Recompute blocked
                in_set2 = np.zeros(N_S, dtype=bool)
                for v in current_set:
                    in_set2[v] = True
                
                if current_set:
                    blocked2 = np.any(adj_bool[in_set2], axis=0)
                else:
                    blocked2 = np.zeros(N_S, dtype=bool)
                
                candidates2_mask = ~(in_set2 | blocked2)
                cand_indices2 = np.where(candidates2_mask)[0]
                
                if len(cand_indices2) > 0:
                    v_add = int(cand_indices2[rng.randint(len(cand_indices2))])
                    current_set.add(v_add)
                    if len(current_set) > best_local_size:
                        best_local_size = len(current_set)
                        best_local = set(current_set)
                else:
                    current_set.add(v_remove)
        
        return best_local
    
    # Run multiple restarts
    best_size = 0
    best_set = set()
    
    num_restarts = 500
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        init_set = greedy_min_degree()
        final_set = local_search(init_set, max_iter=200)
        size = len(final_set)
        
        if size > best_size:
            best_size = size
            best_set = set(final_set)
    
    # Construct vectors from selected supports
    vectors = []
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if not vectors:
        # Fallback: use all 2-subsets
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
                    
    return np.array(vectors, dtype=np.int64)
