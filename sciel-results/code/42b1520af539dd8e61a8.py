import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 147.0  # 3s margin for safety

    N = 512
    K = 32

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
        mask = 0
        start = random.randint(0, N - K)
        for i in range(K):
            mask |= (1 << (start + i))
        return mask

    def init_grid_like():
        # Try a 2D grid-like structure: pick rows and columns
        # e.g., positions at (r*16 + c) for various r, c
        mask = 0
        rows = random.sample(range(32), 4)  # 4 rows of 8
        for r in rows:
            cols = random.sample(range(16), 8)
            for c in cols:
                pos = r * 16 + c
                if pos < N:
                    mask |= (1 << pos)
        return mask

    def init_mixed_blocks():
        # Two blocks of different sizes
        mask = 0
        n1 = random.randint(10, 22)
        n2 = K - n1
        start1 = random.randint(0, N - n1)
        for i in range(n1):
            mask |= (1 << (start1 + i))
        # Find a non-overlapping start for block 2
        start2 = random.randint(0, N - n2)
        while any(((mask >> (start2 + i)) & 1) for i in range(n2)):
            start2 = random.randint(0, N - n2)
        for i in range(n2):
            mask |= (1 << (start2 + i))
        return mask

    initializers = [init_random, init_contiguous, init_two_blocks, init_sparse, 
                    init_arithmetic_progression, init_grid_like, init_mixed_blocks]

    best_mask = 0
    best_score = -1

    num_restarts = 14

    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        init_func = initializers[r % len(initializers)]
        current_mask = init_func()
        
        current_score = get_score(current_mask)
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
        
        T = 12.0
        T_min = 0.05
        cooling_rate = 0.9998  # Very slow cooling for more exploration
        
        set_bits = get_set_bits(current_mask)
        unset_bits = get_unset_bits(current_mask)
        
        local_best_score = current_score
        local_best_mask = current_mask
        
        restart_start = time.time()
        sa_time_limit = min(8.0, (deadline - restart_start) * 0.55)  # Cap SA time, leave room for polish
        
        step_count = 0
        while T > T_min and (time.time() - restart_start) < sa_time_limit:
            # Mix of 1-swap and 2-swap moves
            if random.random() < 0.3:
                # 2-swap move: remove 2 elements, add 2 elements
                if len(set_bits) >= 2 and len(unset_bits) >= 2:
                    idx_i1 = random.randint(0, len(set_bits) - 1)
                    idx_i2 = random.randint(0, len(set_bits) - 1)
                    if idx_i1 == idx_i2:
                        idx_i2 = (idx_i2 + 1) % len(set_bits)
                    idx_j1 = random.randint(0, len(unset_bits) - 1)
                    idx_j2 = random.randint(0, len(unset_bits) - 1)
                    if idx_j1 == idx_j2:
                        idx_j2 = (idx_j2 + 1) % len(unset_bits)
                    
                    i1 = set_bits[idx_i1]
                    i2 = set_bits[idx_i2]
                    j1 = unset_bits[idx_j1]
                    j2 = unset_bits[idx_j2]
                    
                    new_mask = current_mask ^ (1 << i1) ^ (1 << i2) ^ (1 << j1) ^ (1 << j2)
                    new_score = get_score(new_mask)
                    
                    delta = new_score - current_score
                    
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        
                        # Update set_bits and unset_bits
                        set_bits[idx_i1] = j1
                        set_bits[idx_i2] = j2
                        unset_bits[idx_j1] = i1
                        unset_bits[idx_j2] = i2
                        
                        if current_score > local_best_score:
                            local_best_score = current_score
                            local_best_mask = current_mask
            else:
                # 1-swap move: remove 1 element, add 1 element
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
            step_count += 1
            
            # Periodically sort the lists to maintain consistency
            if step_count % 1000 == 0:
                set_bits = get_set_bits(current_mask)
                unset_bits = get_unset_bits(current_mask)
        
        if local_best_score > best_score:
            best_score = local_best_score
            best_mask = local_best_mask
            
        # Phase 1: 1-swap polish (best improvement)
        polish_mask = local_best_mask
        polish_score = local_best_score
        polish_set_bits = get_set_bits(polish_mask)
        polish_unset_bits = get_unset_bits(polish_mask)
        
        polish_deadline = deadline - 2.0
        improved = True
        polish_iter = 0
        max_polish_iter = 30
        
        while improved and time.time() < polish_deadline and polish_iter < max_polish_iter:
            improved = False
            best_swap_score = polish_score
            best_i = -1
            best_j = -1
            
            for i in polish_set_bits:
                for j in polish_unset_bits:
                    new_mask = polish_mask ^ (1 << i) ^ (1 << j)
                    new_score = get_score(new_mask)
                    if new_score > best_swap_score:
                        best_swap_score = new_score
                        best_i = i
                        best_j = j
                        improved = True
            
            if improved:
                polish_mask = polish_mask ^ (1 << best_i) ^ (1 << best_j)
                polish_score = best_swap_score
                
                polish_set_bits = get_set_bits(polish_mask)
                polish_unset_bits = get_unset_bits(polish_mask)
                
                if polish_score > best_score:
                    best_score = polish_score
                    best_mask = polish_mask
                polish_iter += 1

        # Phase 2: 2-swap polish (best improvement, limited)
        if time.time() < polish_deadline - 1.0:
            polish2_mask = polish_mask
            polish2_score = polish_score
            polish2_set_bits = get_set_bits(polish2_mask)
            polish2_unset_bits = get_unset_bits(polish2_mask)
            
            improved2 = True
            polish2_iter = 0
            max_polish2_iter = 10
            polish2_deadline = polish_deadline - 0.5
            
            while improved2 and time.time() < polish2_deadline and polish2_iter < max_polish2_iter:
                improved2 = False
                best_2swap_score = polish2_score
                best_i1 = -1
                best_i2 = -1
                best_j1 = -1
                best_j2 = -1
                
                # Sample a subset of 2-swap moves to keep it tractable
                n_samples = min(2000, len(polish2_set_bits) * (len(polish2_set_bits) - 1) // 2 * 20)
                
                for _ in range(n_samples):
                    idx_i1 = random.randint(0, len(polish2_set_bits) - 1)
                    idx_i2 = random.randint(0, len(polish2_set_bits) - 1)
                    if idx_i1 == idx_i2:
                        continue
                    idx_j1 = random.randint(0, len(polish2_unset_bits) - 1)
                    idx_j2 = random.randint(0, len(polish2_unset_bits) - 1)
                    if idx_j1 == idx_j2:
                        continue
                    
                    i1 = polish2_set_bits[idx_i1]
                    i2 = polish2_set_bits[idx_i2]
                    j1 = polish2_unset_bits[idx_j1]
                    j2 = polish2_unset_bits[idx_j2]
                    
                    new_mask = polish2_mask ^ (1 << i1) ^ (1 << i2) ^ (1 << j1) ^ (1 << j2)
                    new_score = get_score(new_mask)
                    
                    if new_score > best_2swap_score:
                        best_2swap_score = new_score
                        best_i1 = i1
                        best_i2 = i2
                        best_j1 = j1
                        best_j2 = j2
                        improved2 = True
                
                if improved2:
                    polish2_mask = polish2_mask ^ (1 << best_i1) ^ (1 << best_i2) ^ (1 << best_j1) ^ (1 << best_j2)
                    polish2_score = best_2swap_score
                    
                    polish2_set_bits = get_set_bits(polish2_mask)
                    polish2_unset_bits = get_unset_bits(polish2_mask)
                    
                    if polish2_score > best_score:
                        best_score = polish2_score
                        best_mask = polish2_mask
                    polish2_iter += 1

    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
