import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large subset of Z/205Z with no two elements differing by a nonzero square.
    Combines bitmask-based speed from inspiration 2 with the multi-strategy search
    and local search machinery from the current program.
    """
    m = 205
    deadline = time.time() + 580  # Safety margin before 630s kill

    # 1. Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_list = sorted(forbidden)

    # 2. Build conflict masks using bit operations for speed
    # conflict_mask[i] is an integer where bit j is 1 if (i-j) % m is forbidden
    # Since m=205, we need 205 bits. Python ints handle this natively.
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for f in forbidden_list:
            j1 = (i - f) % m
            j2 = (i + f) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        # Exclude self-conflict (though difference 0 is not in forbidden, 
        # it's good practice to ensure i doesn't conflict with itself in logic)
        # Actually, (i-i)=0 is not in forbidden, so bit i should be 0.
        # Let's verify: if f != 0, j1 != i and j2 != i. So bit i is 0.
        conflict_mask[i] = mask

    def get_neighbors_mask(S_mask):
        """
        Given a set S represented as a bitmask, return the bitmask of all elements
        that conflict with at least one element in S.
        """
        neighbors = 0
        # Iterate through set bits in S
        temp = S_mask
        while temp:
            # Get index of least significant set bit
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            neighbors |= conflict_mask[idx]
            temp ^= lsb
        return neighbors

    def greedy_init(order=None):
        """
        Greedy initialization. If order is None, uses random permutation.
        Returns (set_mask, set_list)
        """
        if order is None:
            order = list(range(m))
            random.shuffle(order)
        
        S_mask = 0
        S_list = []
        neighbors = 0
        
        for v in order:
            if not (neighbors & (1 << v)):
                S_mask |= (1 << v)
                S_list.append(v)
                neighbors |= conflict_mask[v]
        
        return S_mask, S_list

    def local_search(S_mask, S_list, time_limit=1.0):
        """
        Local search on the set.
        Returns (best_mask, best_list)
        """
        current_mask = S_mask
        current_list = S_list[:]
        current_size = len(current_list)
        best_mask = current_mask
        best_list = current_list[:]
        best_size = current_size
        start_time = time.time()
        
        kick_sizes = [2, 3, 4, 5]
        
        while time.time() - start_time < time_limit:
            # 1. Try to add any element that doesn't conflict
            neighbors = get_neighbors_mask(current_mask)
            
            # Find candidates: not in S, not in neighbors
            # We can iterate through 0..204
            added = False
            for v in range(m):
                if not (current_mask & (1 << v)) and not (neighbors & (1 << v)):
                    # Add v
                    current_mask |= (1 << v)
                    current_list.append(v)
                    current_size += 1
                    added = True
                    break
            
            if added:
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                    best_list = current_list[:]
                continue
            
            # 2. No immediate addition possible. Perform a kick.
            if current_size == 0:
                break
            
            k = random.choice(kick_sizes)
            if current_size < k:
                break
            
            # Remove k random elements
            indices = list(range(current_size))
            random.shuffle(indices)
            to_remove_indices = indices[:k]
            
            old_mask = current_mask
            old_list = current_list[:]
            old_size = current_size
            
            # Remove from mask and list
            # It's faster to rebuild or just clear bits
            for idx in to_remove_indices:
                v = current_list[idx]
                current_mask &= ~(1 << v)
            
            # Rebuild list for removed elements (simple filter)
            removed_set = set(current_list[idx] for idx in to_remove_indices)
            current_list = [v for v in current_list if v not in removed_set]
            current_size -= k
            
            # Re-greedy fill
            neighbors = get_neighbors_mask(current_mask)
            
            # We want to add as many as possible. 
            # To be more effective, we can try to add elements that "open up" space, 
            # but simple greedy fill is a good heuristic.
            # Let's try a few random orderings for the fill to escape plateaus?
            # Or just standard greedy.
            
            fill_order = list(range(m))
            random.shuffle(fill_order)
            
            for v in fill_order:
                if not (current_mask & (1 << v)) and not (neighbors & (1 << v)):
                    current_mask |= (1 << v)
                    current_list.append(v)
                    current_size += 1
                    neighbors |= conflict_mask[v]
            
            # Acceptance
            if current_size > old_size:
                # Strict improvement
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                    best_list = current_list[:]
            elif current_size == old_size:
                # Plateau: Accept with probability 0.5
                if random.random() < 0.5:
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                        best_list = current_list[:]
            else:
                # Degradation: Revert
                current_mask = old_mask
                current_list = old_list
                current_size = old_size

        return best_mask, best_list

    # Main Search Loop
    best_global_mask = 0
    best_global_list = []
    best_global_size = 0
    
    # 1. Deterministic greedy baseline
    S_det_mask, S_det_list = greedy_init(list(range(m)))
    best_global_mask = S_det_mask
    best_global_list = S_det_list
    best_global_size = len(S_det_list)
    
    # Run local search on the baseline to improve it
    S_ls_mask, S_ls_list = local_search(S_det_mask, S_det_list, time_limit=2.0)
    if len(S_ls_list) > best_global_size:
        best_global_mask = S_ls_mask
        best_global_list = S_ls_list
        best_global_size = len(S_ls_list)
        
    # 2. Multi-start random search
    # We allocate the remaining time to random restarts.
    # Each restart consists of: greedy_init + local_search
    # Time per restart is adaptive.
    
    restart_count = 0
    while time.time() < deadline - 5:
        # Generate random initial solution
        S_init_mask, S_init_list = greedy_init()
        
        # Allocate time for local search
        time_left = deadline - time.time()
        if time_left < 10:
            break
            
        # Give each restart a slice of time.
        # Start with 1s, increase if we are improving?
        # For now, fixed 1.5s per restart seems reasonable.
        ls_time = min(1.5, time_left - 5)
        
        S_best_mask, S_best_list = local_search(S_init_mask, S_init_list, time_limit=ls_time)
        size = len(S_best_list)
        
        if size > best_global_size:
            best_global_mask = S_best_mask
            best_global_list = S_best_list
            best_global_size = size
            
        restart_count += 1
        
        # Optional: If we hit a very high size, we can increase confidence
        # but for now, keep searching.

    # 3. Polish: Run local search on the best found set with more time
    # This is a "high-precision polish pass"
    time_left = deadline - time.time()
    if time_left > 10:
        S_polish_mask, S_polish_list = local_search(best_global_mask, best_global_list, time_limit=time_left - 5)
        if len(S_polish_list) > best_global_size:
            best_global_mask = S_polish_mask
            best_global_list = S_polish_list
            best_global_size = len(S_polish_list)

    # Return the best set found
    # Ensure sorted for consistent output? Not strictly required by task, but good practice.
    # The task asks for distinct residues in 0..204.
    result_indices = np.array(sorted(best_global_list), dtype=np.int64)
    return result_indices

# EVOLVE-BLOCK-END
