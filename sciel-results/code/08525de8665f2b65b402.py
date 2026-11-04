import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11
    
    # 1. Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330
    
    # Convert supports to bitmasks for fast intersection
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
        
    # 2. Build conflict graph using vectorized operations
    # Conflict if intersection size is 3
    # Create a matrix where adj_matrix[i,j] = 1 if supports[i] and supports[j] share exactly 3 indices
    # First, create a binary incidence matrix: shape (N_S, D)
    incidence = np.zeros((N_S, D), dtype=np.int8)
    for i, s in enumerate(supports):
        for idx in s:
            incidence[i, idx] = 1
    
    # Compute pairwise intersection sizes: inter[i,j] = dot(incidence[i], incidence[j])
    # This gives the number of common indices
    inter_matrix = incidence @ incidence.T  # shape (N_S, N_S), values 0-4
    
    # Conflict if intersection is exactly 3
    conflict = (inter_matrix == 3).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # adj_list[i] = list of j that conflict with i
    adj_list = [list(np.where(conflict[i] == 1)[0]) for i in range(N_S)]
    
    # 3. Find Max Independent Set with improved heuristics
    best_size = 0
    best_set = set()
    
    def greedy_min_degree_vectorized():
        """Greedy min-degree heuristic using vectorized degree computation."""
        remaining = np.ones(N_S, dtype=bool)
        s = []
        
        while remaining.any():
            # Compute degree within remaining for each remaining node
            # degree[i] = sum of conflict[i, j] for j in remaining
            degrees = (conflict @ remaining).astype(np.int32)
            # Only consider remaining nodes
            degrees[~remaining] = np.inf
            
            # Find node with minimum degree
            min_node = np.argmin(degrees)
            s.append(min_node)
            
            # Remove min_node and its neighbors from remaining
            to_remove = np.where(conflict[min_node] == 1)[0]
            remaining[to_remove] = False
            remaining[min_node] = False
        
        return set(s)
    
    def greedy_randomized(seed_val):
        """Randomized greedy: at each step, pick from top-k lowest degree nodes."""
        rng = np.random.RandomState(seed_val)
        remaining = np.ones(N_S, dtype=bool)
        s = []
        
        while remaining.any():
            degrees = (conflict @ remaining).astype(np.int32)
            degrees[~remaining] = np.inf
            
            # Get remaining nodes sorted by degree
            remaining_indices = np.where(remaining)[0]
            remaining_degrees = degrees[remaining_indices]
            
            # Pick from top-k (lowest degree) with some randomness
            k = min(5, len(remaining_indices))
            # Get the k smallest degrees
            order = np.argsort(remaining_degrees)
            top_k = order[:k]
            
            # Randomly pick from top_k
            chosen = top_k[rng.randint(len(top_k))]
            min_node = remaining_indices[chosen]
            s.append(min_node)
            
            # Remove min_node and its neighbors
            to_remove = np.where(conflict[min_node] == 1)[0]
            remaining[to_remove] = False
            remaining[min_node] = False
        
        return set(s)
    
    def local_search(init_set, max_iter=300, seed_val=0):
        """Improved local search with better swap heuristics."""
        rng = np.random.RandomState(seed_val)
        current_set = set(init_set)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
                
            # Compute blocked set (neighbors of current set members)
            blocked = set()
            for v in current_set:
                blocked.update(adj_list[v])
            
            # Try to add a random unblocked node
            candidates = [v for v in range(N_S) if v not in current_set and v not in blocked]
            
            if candidates:
                v_add = candidates[rng.randint(len(candidates))]
                current_set.add(v_add)
                continue
            
            # Stuck: try a 1-swap to escape local optimum
            if current_set:
                # Try multiple random removals and keep the best
                best_swap_size = len(current_set)
                best_swap_set = None
                
                for swap_attempt in range(3):
                    v_remove = list(current_set)[rng.randint(len(current_set))]
                    temp_set = set(current_set)
                    temp_set.remove(v_remove)
                    
                    # Recompute blocked
                    blocked2 = set()
                    for v in temp_set:
                        blocked2.update(adj_list[v])
                    
                    candidates2 = [v for v in range(N_S) if v not in temp_set and v not in blocked2]
                    
                    if candidates2:
                        v_add = candidates2[rng.randint(len(candidates2))]
                        new_set = set(temp_set)
                        new_set.add(v_add)
                        
                        # Try to greedily add more nodes after the swap
                        for _ in range(10):
                            blocked3 = set()
                            for v in new_set:
                                blocked3.update(adj_list[v])
                            cands3 = [v for v in range(N_S) if v not in new_set and v not in blocked3]
                            if not cands3:
                                break
                            new_set.add(cands3[rng.randint(len(cands3))])
                        
                        if len(new_set) > best_swap_size:
                            best_swap_size = len(new_set)
                            best_swap_set = new_set
                
                if best_swap_set is not None and len(best_swap_set) >= len(current_set):
                    current_set = best_swap_set
                # else: keep current_set (no improvement found)
        
        return current_set
    
    # Run multiple restarts with different strategies
    np.random.seed(42)
    random.seed(42)
    
    num_restarts = 400
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Alternate between different initialization strategies
        if r % 3 == 0:
            init_set = greedy_min_degree_vectorized()
        elif r % 3 == 1:
            init_set = greedy_randomized(r)
        else:
            # Random subset initialization
            subset_size = rng.randint(20, 50)
            init_set = set(rng.choice(N_S, size=min(subset_size, N_S), replace=False))
            # Filter to ensure independence
            filtered = []
            for v in sorted(init_set):
                if all(v2 not in adj_list[v] for v2 in filtered):
                    filtered.append(v)
            init_set = set(filtered)
        
        final_set = local_search(init_set, max_iter=150, seed_val=r)
        size = len(final_set)
        
        if size > best_size:
            best_size = size
            best_set = set(final_set)
    
    # 4. Construct vectors
    vectors = []
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors
