import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    N = 512
    K = 32

    def get_score(m):
        """
        Calculate the number of distinct d > 0 such that 
        a, a+d, a+2d are in the set represented by bitmask m.
        """
        score = 0
        # Max d is such that 2*d < N, so d < 256
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

    def init_sparse_ap():
        # Arithmetic progression with step s
        # 32 elements, max index 511
        # s * 31 <= 511 => s <= 16
        s = random.randint(10, 16)
        start = random.randint(0, N - s * (K - 1))
        mask = 0
        for i in range(K):
            mask |= (1 << (start + i * s))
        return mask

    def init_mixed():
        # Half contiguous, half sparse
        mask = 0
        # 16 contiguous
        start1 = random.randint(0, N - 16)
        for i in range(16):
            mask |= (1 << (start1 + i))
        # 16 sparse
        s = random.randint(10, 20)
        start2 = random.randint(0, N - s * 15)
        for i in range(16):
            idx = start2 + i * s
            if idx < N and not ((mask >> idx) & 1):
                mask |= (1 << idx)
            else:
                # Find a free spot
                for j in range(1, 10):
                    new_idx = idx + j
                    if new_idx < N and not ((mask >> new_idx) & 1):
                        mask |= (1 << new_idx)
                        break
        return mask

    initializers = [init_random, init_contiguous, init_two_blocks, init_sparse_ap, init_mixed]

    best_mask = 0
    best_score = -1

    num_restarts = 12

    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        init_func = initializers[r % len(initializers)]
        current_mask = init_func()
        
        current_score = get_score(current_mask)
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
        
        # SA Phase
        T = 15.0
        T_min = 0.1
        cooling_rate = 0.9995
        
        set_bits = get_set_bits(current_mask)
        unset_bits = get_unset_bits(current_mask)
        
        local_best_score = current_score
        local_best_mask = current_mask
        
        restart_start = time.time()
        sa_time_limit = min(10.0, (deadline - restart_start) * 0.6)
        
        step_count = 0
        while T > T_min and (time.time() - restart_start) < sa_time_limit:
            # Mix of 1-swap and 2-swap moves
            # 2-swaps are more powerful for escaping local optima
            if random.random() < 0.4:
                # 2-swap move
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
                    
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        
                        # Update lists
                        set_bits[idx_i1] = j1
                        set_bits[idx_i2] = j2
                        unset_bits[idx_j1] = i1
                        unset_bits[idx_j2] = i2
                        
                        if current_score > local_best_score:
                            local_best_score = current_score
                            local_best_mask = current_mask
            else:
                # 1-swap move
                idx_i = random.randint(0, len(set_bits) - 1)
                idx_j = random.randint(0, len(unset_bits) - 1)
                
                i = set_bits[idx_i]
                j = unset_bits[idx_j]
                
                new_mask = current_mask ^ (1 << i) ^ (1 << j)
                new_score = get_score(new_mask)
                
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    
                    set_bits[idx_i] = j
                    unset_bits[idx_j] = i
                    
                    if current_score > local_best_score:
                        local_best_score = current_score
                        local_best_mask = current_mask
                        
            T *= cooling_rate
            step_count += 1
            
            # Periodically sort/rebuild lists to avoid drift
            if step_count % 500 == 0:
                set_bits = get_set_bits(current_mask)
                unset_bits = get_unset_bits(current_mask)
        
        if local_best_score > best_score:
            best_score = local_best_score
            best_mask = local_best_mask
            
        # Polish Phase
        polish_mask = local_best_mask
        polish_score = local_best_score
        polish_set_bits = get_set_bits(polish_mask)
        polish_unset_bits = get_unset_bits(polish_mask)
        
        polish_deadline = deadline - 3.0
        
        # Phase 1: Greedy 1-swap
        improved = True
        polish_iter = 0
        max_polish_iter = 50
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
            
            if best_i != -1:
                polish_mask = polish_mask ^ (1 << best_i) ^ (1 << best_j)
                polish_score = best_swap_score
                
                polish_set_bits.remove(best_i)
                polish_set_bits.append(best_j)
                polish_unset_bits.remove(best_j)
                polish_unset_bits.append(best_i)
                
                if polish_score > local_best_score:
                    local_best_score = polish_score
                    local_best_mask = polish_mask
                if polish_score > best_score:
                    best_score = polish_score
                    best_mask = polish_mask
                
                improved = True
                polish_iter += 1
        
        # Phase 2: Random 2-swap hill climbing
        # Try to find any improving 2-swap
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < polish_deadline and no_improve_count < max_no_improve:
            # Try a random 2-swap
            if len(polish_set_bits) < 2 or len(polish_unset_bits) < 2:
                break
                
            idx_i1 = random.randint(0, len(polish_set_bits) - 1)
            idx_i2 = random.randint(0, len(polish_set_bits) - 1)
            if idx_i1 == idx_i2:
                no_improve_count += 1
                continue
            idx_j1 = random.randint(0, len(polish_unset_bits) - 1)
            idx_j2 = random.randint(0, len(polish_unset_bits) - 1)
            if idx_j1 == idx_j2:
                no_improve_count += 1
                continue
                
            i1 = polish_set_bits[idx_i1]
            i2 = polish_set_bits[idx_i2]
            j1 = polish_unset_bits[idx_j1]
            j2 = polish_unset_bits[idx_j2]
            
            new_mask = polish_mask ^ (1 << i1) ^ (1 << i2) ^ (1 << j1) ^ (1 << j2)
            new_score = get_score(new_mask)
            
            if new_score > polish_score:
                polish_mask = new_mask
                polish_score = new_score
                
                # Update lists
                polish_set_bits[idx_i1] = j1
                polish_set_bits[idx_i2] = j2
                polish_unset_bits[idx_j1] = i1
                polish_unset_bits[idx_j2] = i2
                
                if polish_score > local_best_score:
                    local_best_score = polish_score
                    local_best_mask = polish_mask
                if polish_score > best_score:
                    best_score = polish_score
                    best_mask = polish_mask
                    
                no_improve_count = 0
            else:
                no_improve_count += 1

    # Convert best_mask to list of integers
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    # Ensure exactly 32 distinct integers
    if len(result) != K:
        result = list(range(1, K + 1))
        
    return result
