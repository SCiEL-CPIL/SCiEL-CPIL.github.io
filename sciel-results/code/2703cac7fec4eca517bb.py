import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety

    N = 512
    K = 32

    # Precompute relevant differences for each index
    # relevant_ds[x] is a set of d such that x is part of a 3-AP with difference d
    # i.e., x = a, x = a+d, or x = a+2d for some a.
    relevant_ds = [set() for _ in range(N)]
    for d in range(1, 256):
        for a in range(0, N - 2 * d):
            # Triplet: a, a+d, a+2d
            relevant_ds[a].add(d)
            relevant_ds[a + d].add(d)
            relevant_ds[a + 2 * d].add(d)
    
    # Convert to lists for faster iteration
    relevant_ds_list = [list(s) for s in relevant_ds]

    def get_score(m):
        """
        Calculate the number of distinct d > 0 such that 
        a, a+d, a+2d are in the set represented by bitmask m.
        """
        score = 0
        for d in range(1, 256):
            if (m & (m >> d) & (m >> (2 * d))) != 0:
                score += 1
        return score

    def get_sat_mask(m):
        """
        Returns a bitmask where bit d is set if d is satisfied.
        """
        sat_mask = 0
        for d in range(1, 256):
            if (m & (m >> d) & (m >> (2 * d))) != 0:
                sat_mask |= (1 << d)
        return sat_mask

    def get_set_bits(m):
        return [i for i in range(N) if (m >> i) & 1]

    def get_unset_bits(m):
        return [i for i in range(N) if not ((m >> i) & 1)]

    def init_random():
        indices = random.sample(range(N), K)
        mask = 0
        for i in indices:
            mask |= (1 << i)
        return mask

    def init_contiguous():
        mask = 0
        for i in range(K):
            mask |= (1 << i)
        return mask

    def init_two_blocks():
        mask = 0
        for i in range(16):
            mask |= (1 << i)
        for i in range(200, 216):
            mask |= (1 << i)
        return mask

    def init_sparse():
        step = N // K
        mask = 0
        for i in range(K):
            idx = i * step
            mask |= (1 << idx)
        return mask

    def init_arithmetic_progression():
        # Start with a dense arithmetic progression structure
        mask = 0
        start = random.randint(0, N - K)
        for i in range(K):
            mask |= (1 << (start + i))
        return mask

    def init_grid():
        # Product set structure: {i * 16 + j for i in 0..1, j in 0..15}
        # This covers small differences well.
        mask = 0
        # 2 blocks of 16
        for i in range(16):
            mask |= (1 << i)
            mask |= (1 << (i + 16 * 10)) # Shifted by 160
        return mask

    initializers = [init_random, init_contiguous, init_two_blocks, init_sparse, init_arithmetic_progression, init_grid]

    best_mask = 0
    best_score = -1

    num_restarts = 12

    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        init_func = initializers[r % len(initializers)]
        current_mask = init_func()
        
        current_score = get_score(current_mask)
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
        
        T = 15.0
        T_min = 0.05
        cooling_rate = 0.9995  # Slower cooling
        
        set_bits = get_set_bits(current_mask)
        unset_bits = get_unset_bits(current_mask)
        
        local_best_score = current_score
        local_best_mask = current_mask
        
        restart_start = time.time()
        sa_time_limit = (deadline - restart_start) * 0.6  # Leave more time for Polish
        
        while T > T_min and (time.time() - restart_start) < sa_time_limit:
            idx_i = random.randint(0, len(set_bits) - 1)
            idx_j = random.randint(0, len(unset_bits) - 1)
            
            i = set_bits[idx_i]
            j = unset_bits[idx_j]
            
            new_mask = current_mask ^ (1 << i) ^ (1 << j)
            new_score = get_score(new_mask)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_mask = new_mask
                current_score = new_score
                
                set_bits[idx_i] = j
                unset_bits[idx_j] = i
                
                if current_score > local_best_score:
                    local_best_score = current_score
                    local_best_mask = current_mask
                    
            T *= cooling_rate
            
        if local_best_score > best_score:
            best_score = local_best_score
            best_mask = local_best_mask
            
        # Polish with incremental score update
        polish_mask = local_best_mask
        polish_score = local_best_score
        polish_sat_mask = get_sat_mask(polish_mask)
        polish_set_bits = get_set_bits(polish_mask)
        polish_unset_bits = get_unset_bits(polish_mask)
        
        polish_deadline = deadline - 0.5
        improved = True
        polish_iter = 0
        max_polish_iter = 200 # Can run more iterations now that they are faster

        while improved and time.time() < polish_deadline and polish_iter < max_polish_iter:
            improved = False
            best_swap_score = polish_score
            best_i = -1
            best_j = -1
            best_new_sat_mask = polish_sat_mask
            
            for i in polish_set_bits:
                for j in polish_unset_bits:
                    # Calculate delta efficiently
                    # Affected d's are those in relevant_ds[i] U relevant_ds[j]
                    affected_ds = set(relevant_ds_list[i])
                    affected_ds.update(relevant_ds_list[j])
                    
                    new_mask_candidate = polish_mask ^ (1 << i) ^ (1 << j)
                    
                    score_delta = 0
                    new_sat_mask_local = polish_sat_mask
                    
                    for d in affected_ds:
                        was_sat = (polish_sat_mask >> d) & 1
                        is_sat = 1 if (new_mask_candidate & (new_mask_candidate >> d) & (new_mask_candidate >> (2 * d))) != 0 else 0
                        
                        if was_sat == 0 and is_sat == 1:
                            score_delta += 1
                            new_sat_mask_local |= (1 << d)
                        elif was_sat == 1 and is_sat == 0:
                            score_delta -= 1
                            new_sat_mask_local &= ~(1 << d)
                    
                    candidate_score = polish_score + score_delta
                    
                    if candidate_score > best_swap_score:
                        best_swap_score = candidate_score
                        best_i = i
                        best_j = j
                        best_new_sat_mask = new_sat_mask_local
                        improved = True
            
            if improved:
                polish_mask = polish_mask ^ (1 << best_i) ^ (1 << best_j)
                polish_score = best_swap_score
                polish_sat_mask = best_new_sat_mask
                
                polish_set_bits = get_set_bits(polish_mask)
                polish_unset_bits = get_unset_bits(polish_mask)
                
                if polish_score > best_score:
                    best_score = polish_score
                    best_mask = polish_mask
                polish_iter += 1

    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
