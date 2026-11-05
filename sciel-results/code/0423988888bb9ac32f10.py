import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety
    
    N = 512
    K = 32
    
    def get_score(m):
        """
        Calculate the number of distinct d > 0 such that 
        a, a+d, a+2d are in the set represented by bitmask m.
        """
        score = 0
        # d can range from 1 to 255 (since 2*255 = 510 < 512)
        for d in range(1, 256):
            # Check if there exists a such that bits a, a+d, a+2d are set
            # m & (m >> d) & (m >> 2*d) will have bit k set if 
            # m has bits k, k+d, k+2d set.
            if (m & (m >> d) & (m >> (2 * d))) != 0:
                score += 1
        return score

    def get_set_bits(m):
        return [i for i in range(N) if (m >> i) & 1]
    
    def get_unset_bits(m):
        return [i for i in range(N) if not ((m >> i) & 1)]

    # Initializations: Create a diverse set of starting points
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
        # Two blocks of 16
        for i in range(16):
            mask |= (1 << i)
        for i in range(200, 216):
            mask |= (1 << i)
        return mask

    def init_sparse():
        # Spread out elements
        step = N // K
        mask = 0
        for i in range(K):
            idx = i * step
            mask |= (1 << idx)
        return mask

    initializers = [init_random, init_contiguous, init_two_blocks, init_sparse, init_random]
    
    best_mask = 0
    best_score = -1
    
    # Number of restarts
    # We will run 10 restarts. Each gets ~11 seconds.
    num_restarts = 10
    
    for r in range(num_restarts):
        # Check time
        if time.time() > deadline:
            break
            
        # Select initializer
        init_func = initializers[r % len(initializers)]
        current_mask = init_func()
        
        # For random init, ensure it's not the exact same as previous if possible, 
        # but random.sample is usually distinct enough.
        
        current_score = get_score(current_mask)
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
        
        # Simulated Annealing
        T = 10.0
        T_min = 0.05
        cooling_rate = 0.999
        
        set_bits = get_set_bits(current_mask)
        unset_bits = get_unset_bits(current_mask)
        
        local_best_score = current_score
        local_best_mask = current_mask
        
        restart_start = time.time()
        # Allocate time for SA: 80% of the restart budget
        sa_time_limit = (deadline - restart_start) * 0.8
        
        while T > T_min and (time.time() - restart_start) < sa_time_limit:
            idx_i = random.randint(0, len(set_bits) - 1)
            idx_j = random.randint(0, len(unset_bits) - 1)
            
            i = set_bits[idx_i]
            j = unset_bits[idx_j]
            
            # Swap
            new_mask = current_mask ^ (1 << i) ^ (1 << j)
            new_score = get_score(new_mask)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_mask = new_mask
                current_score = new_score
                
                # Update lists
                set_bits[idx_i] = j
                unset_bits[idx_j] = i
                
                if current_score > local_best_score:
                    local_best_score = current_score
                    local_best_mask = current_mask
                    
            T *= cooling_rate
            
        # Update global best with local best
        if local_best_score > best_score:
            best_score = local_best_score
            best_mask = local_best_mask
            
        # Greedy Polish Phase
        # Try to improve the local best by making the single best swap available.
        # We do this for a few iterations or until no improvement is found.
        polish_mask = local_best_mask
        polish_score = local_best_score
        polish_set_bits = get_set_bits(polish_mask)
        polish_unset_bits = get_unset_bits(polish_mask)
        
        # Limit polish time to remaining time
        polish_deadline = deadline - 0.5
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            best_swap_score = polish_score
            best_i = -1
            best_j = -1
            
            # Iterate through all possible swaps
            # 32 * 480 = 15360 swaps.
            # To speed up, we can sample a subset if time is tight, 
            # but 15360 * 256 ops is ~4M ops, which is fast in Python if optimized? 
            # Actually, get_score is 256 iterations of bitwise ops. 
            # 15360 * 256 = ~4 million bitwise ops. 
            # In Python, this might take ~0.1-0.2s.
            # So we can do a full pass.
            
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
                # Apply the best swap
                polish_mask = polish_mask ^ (1 << best_i) ^ (1 << best_j)
                polish_score = best_swap_score
                
                # Update lists
                # Remove best_i from set, add best_j
                # Remove best_j from unset, add best_i
                polish_set_bits = get_set_bits(polish_mask)
                polish_unset_bits = get_unset_bits(polish_mask)
                
                if polish_score > best_score:
                    best_score = polish_score
                    best_mask = polish_mask

    # Convert best_mask to list of integers
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
