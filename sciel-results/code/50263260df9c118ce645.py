"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Fast incremental check using maintained set of triple sums
    def try_add(state, candidate):
        """
        state: (list_of_elements, set_of_triple_sums)
        candidate: int to add
        Returns: (new_list, new_sums) if valid, else None
        """
        lst, sums = state
        n = len(lst)
        new_sums = set()
        
        # Check collisions with existing sums and internal consistency of new sums
        # 1. Triples with one candidate: candidate + lst[i] + lst[j] for i <= j
        for i in range(n):
            li = lst[i]
            for j in range(i, n):
                lj = lst[j]
                s = candidate + li + lj
                if s in sums or s in new_sums:
                    return None
                new_sums.add(s)
        
        # 2. Triples with two candidates: 2*candidate + lst[i]
        for i in range(n):
            s = 2 * candidate + lst[i]
            if s in sums or s in new_sums:
                return None
            new_sums.add(s)
        
        # 3. Triple with three candidates: 3*candidate
        s = 3 * candidate
        if s in sums or s in new_sums:
            return None
        new_sums.add(s)
        
        # All checks passed, update state
        new_lst = lst + [candidate]
        new_sums_set = sums.union(new_sums)
        return (new_lst, new_sums_set)

    def init_state(lst):
        """Initialize the set of triple sums for a given list."""
        sums = set()
        n = len(lst)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums.add(lst[i] + lst[j] + lst[k])
        return (lst, sums)

    # Base set: Powers of 4 are a strong start
    base = [4**i for i in range(10) if 4**i <= 4096]
    base_set = set(base)
    
    best_set = list(base)
    best_len = len(base)
    
    # Candidate pool
    all_candidates = [x for x in range(1, 4097) if x not in base_set]
    
    # Randomized restarts
    num_restarts = 200 # Increased because check is much faster
    
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Start with base
        current_state = init_state(list(base))
        current_lst = current_state[0]
        
        # Shuffle candidates
        candidates = all_candidates[:]
        random.shuffle(candidates)
        
        for cand in candidates:
            if time.time() > deadline:
                break
            if len(current_lst) >= 80:
                break
            
            # Try to add candidate
            result = try_add(current_state, cand)
            if result is not None:
                current_state = result
                current_lst = result[0]
        
        if len(current_lst) > best_len:
            best_len = len(current_lst)
            best_set = list(current_lst)
            
            # If we reached the max possible or a very high number, we can stop early?
            # Theoretical max is lower than 80 for N=4096, but let's see.
            # B_3[1] max size for N=4096 is roughly sqrt(N) * constant?
            # Actually, for Sidon sets (B_2[1]), size is ~sqrt(N). For B_3[1], it's smaller.
            # sqrt(4096) = 64. So 80 might be impossible?
            # Wait, 4^0..4^6 is 7 elements.
            # If max is around 64-70, we should stop if we get close to that?
            # But we don't know the exact max. Let's just maximize.
            
    return best_set[:80]
# EVOLVE-BLOCK-END
