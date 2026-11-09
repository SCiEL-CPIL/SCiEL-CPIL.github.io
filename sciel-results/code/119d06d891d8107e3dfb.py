import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses Iterated Local Search (ILS) with a conflict matrix for speed.
    """
    m = 205
    deadline = time.time() + 580.0  # Safety margin before 630s kill

    # 1. Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = pow(x, 2, m)
        if val != 0:
            forbidden.add(val)
    
    # 2. Build conflict matrix
    # conflict[i, j] is True if i and j cannot be in the set together
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(i + 1, m):
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[i, j] = True
                conflict[j, i] = True

    # Helper to check if a candidate can be added to a set represented by a boolean array
    def can_add(candidate, S_bool):
        # Check if candidate conflicts with any element in S
        # We need to check if any True in S_bool is also True in conflict[candidate]
        # Note: conflict[candidate, candidate] is False, so if candidate is in S_bool, 
        # it won't block itself unless we handle it, but we only call this for candidates NOT in S_bool.
        if np.any(S_bool & conflict[candidate]):
            return False
        return True

    def greedy_construct(order, S_bool=None):
        """
        Greedy construction using a specific order.
        If S_bool is provided, it is used as the starting set (for repair).
        """
        if S_bool is None:
            S_bool = np.zeros(m, dtype=bool)
        else:
            S_bool = S_bool.copy()
        
        for i in order:
            if not S_bool[i]:
                if can_add(i, S_bool):
                    S_bool[i] = True
        return S_bool

    def local_search(S_bool):
        """
        Local search to improve the set.
        Moves:
        1. Add 1 element if possible.
        2. Remove 1, Add 2 elements (swap).
        """
        improved = True
        while improved:
            improved = False
            
            # Try Add 1
            for v in range(m):
                if not S_bool[v]:
                    if can_add(v, S_bool):
                        S_bool[v] = True
                        improved = True
                        break # Restart search after change
            
            if improved:
                continue # Restart loop to check for more adds or swaps

            # Try Remove 1, Add 2
            # Iterate over u in S to remove
            S_list = [i for i in range(m) if S_bool[i]]
            found_swap = False
            for u in S_list:
                # Tentatively remove u
                S_bool[u] = False
                
                # Find two elements v, z to add
                # Candidates are all v not in S
                # We need v and z to be compatible with S\{u} and each other
                
                valid_candidates = []
                for v in range(m):
                    if not S_bool[v]:
                        if can_add(v, S_bool):
                            valid_candidates.append(v)
                
                # Find pair v, z in valid_candidates with no conflict
                k = len(valid_candidates)
                pair_found = False
                for i_idx in range(k):
                    v = valid_candidates[i_idx]
                    for z_idx in range(i_idx + 1, k):
                        z = valid_candidates[z_idx]
                        if not conflict[v][z]:
                            # Add v and z
                            S_bool[v] = True
                            S_bool[z] = True
                            found_swap = True
                            pair_found = True
                            break
                    if pair_found:
                        break
                
                if found_swap:
                    improved = True
                    break # Restart local search with new set
                
                # If no swap found with this u, restore u
                S_bool[u] = True
            
        return S_bool

    best_size = 0
    best_bool = np.zeros(m, dtype=bool)
    
    # Initial random greedy
    order = list(range(m))
    random.shuffle(order)
    S_bool = greedy_construct(order)
    S_bool = local_search(S_bool)
    best_size = np.sum(S_bool)
    best_bool = S_bool.copy()

    # Iterated Local Search
    current_bool = best_bool.copy()
    
    while time.time() < deadline:
        # Perturb: remove 1 or 2 random elements
        num_remove = random.choice([1, 2])
        if num_remove == 1:
            S_list = [i for i in range(m) if current_bool[i]]
            if S_list:
                to_remove = random.choice(S_list)
                current_bool[to_remove] = False
        else:
            S_list = [i for i in range(m) if current_bool[i]]
            if len(S_list) >= 2:
                to_remove = random.sample(S_list, 2)
                for u in to_remove:
                    current_bool[u] = False
            elif len(S_list) == 1:
                current_bool[S_list[0]] = False
        
        # Repair: Greedy fill with random order
        order = list(range(m))
        random.shuffle(order)
        current_bool = greedy_construct(order, current_bool)
        
        # Local Search
        current_bool = local_search(current_bool)
        
        current_size = np.sum(current_bool)
        if current_size > best_size:
            best_size = current_size
            best_bool = current_bool.copy()

    # Return the best set found
    result_indices = np.where(best_bool)[0]
    return np.array(result_indices, dtype=np.int64)
