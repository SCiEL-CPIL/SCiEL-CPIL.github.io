import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    N = 512
    K = 32
    
    def get_score(m):
        score = 0
        for d in range(1, 256):
            if (m & (m >> d) & (m >> (2 * d))) != 0:
                score += 1
        return score
    
    best_mask = 0
    best_score = 0
    
    # Multiple random restarts with simulated annealing
    # Each restart gets a fraction of the time
    num_restarts = 12
    time_per_restart = 110.0 / num_restarts
    
    for restart in range(num_restarts):
        # Initialize with random set
        indices = random.sample(range(N), K)
        mask = 0
        for i in indices:
            mask |= (1 << i)
        
        current_score = get_score(mask)
        restart_best_mask = mask
        restart_best_score = current_score
        
        # Simulated annealing parameters
        T = 50.0
        T_min = 0.1
        cooling_rate = 0.9995
        
        set_bits = [i for i in range(N) if (mask >> i) & 1]
        unset_bits = [i for i in range(N) if not ((mask >> i) & 1)]
        
        # Track time for this restart
        restart_start = time.time()
        
        while T > T_min and (time.time() - restart_start) < time_per_restart:
            # Pick random i from set_bits, j from unset_bits
            idx_i = random.randint(0, len(set_bits) - 1)
            idx_j = random.randint(0, len(unset_bits) - 1)
            
            i = set_bits[idx_i]
            j = unset_bits[idx_j]
            
            # Swap
            new_mask = mask ^ (1 << i) ^ (1 << j)
            new_score = get_score(new_mask)
            
            delta = new_score - current_score
            
            if delta >= 0:
                # Accept improving or equal moves
                mask = new_mask
                current_score = new_score
                set_bits[idx_i] = j
                unset_bits[idx_j] = i
                
                if current_score > restart_best_score:
                    restart_best_score = current_score
                    restart_best_mask = mask
                    
            else:
                # Accept worsening moves with probability exp(delta / T)
                if random.random() < (2.718281828 ** (delta / T)):
                    mask = new_mask
                    current_score = new_score
                    set_bits[idx_i] = j
                    unset_bits[idx_j] = i
            
            T *= cooling_rate
        
        # Update global best
        if restart_best_score > best_score:
            best_score = restart_best_score
            best_mask = restart_best_mask
    
    # Convert best_mask to list of integers
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
