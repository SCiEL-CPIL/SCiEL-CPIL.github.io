import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Bitset representation
    # x in 1..512 maps to bit x-1
    # S is an integer where bit i is 1 if (i+1) is in the set
    
    def get_bitset(S):
        b = 0
        for x in S:
            b |= (1 << (x - 1))
        return b

    def score_bitset(b):
        count = 0
        # d can be 1..255
        # If d is too large, S >> 2d will be 0
        for d in range(1, 256):
            if (b & (b >> d) & (b >> 2 * d)):
                count += 1
        return count

    def random_set():
        return random.sample(range(1, 513), 32)
    
    # Initialize with multiple random restarts
    best_set = None
    best_bit = None
    best_score = -1
    
    # Use more restarts with faster SA
    num_restarts = 8
    time_per_restart = (deadline - start_time - 15.0) / num_restarts  # Reserve 15s for final polish
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 15:  # Stop if close to deadline
            break
        
        # Time budget for this restart
        restart_start = time.time()
        restart_end = restart_start + time_per_restart
        
        current_set = random_set()
        current_bit = get_bitset(current_set)
        current_score = score_bitset(current_bit)
        
        # Simulated Annealing with faster cooling
        T = 20.0
        T_min = 0.01
        cooling_rate = 0.99995
        stall_count = 0
        stall_threshold = 1000
        no_improve_count = 0
        no_improve_threshold = 5000
        
        while time.time() < restart_end and T > T_min:
            # Generate neighbor: swap one element
            idx = random.randint(0, 31)
            old_val = current_set[idx]
            
            # Try a few random values to increase chance of valid move
            for attempt in range(3):
                new_val = random.randint(1, 512)
                if new_val == old_val or new_val in current_set:
                    continue
                
                # Update bitset
                new_bit = current_bit
                new_bit &= ~(1 << (old_val - 1))
                new_bit |= (1 << (new_val - 1))
                
                new_score = score_bitset(new_bit)
                delta = new_score - current_score
                
                # Acceptance criterion
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    # Accept
                    current_set[idx] = new_val
                    current_bit = new_bit
                    current_score = new_score
                    stall_count = 0
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                        best_bit = current_bit
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        
                    break
                else:
                    stall_count += 1
                    no_improve_count += 1
            else:
                # All attempts failed, skip
                stall_count += 1
                no_improve_count += 1
            
            # Increase cooling when stalled
            if stall_count > stall_threshold:
                T *= 0.9
                stall_count = 0
            
            # Force restart if no improvement for a long time
            if no_improve_count > no_improve_threshold:
                break
                
            T *= cooling_rate
        
        # Local search polish for this restart
        if time.time() < deadline - 5:
            improved = True
            polish_deadline = min(restart_end + 2.0, deadline - 5)
            while improved and time.time() < polish_deadline:
                improved = False
                for i in range(32):
                    if time.time() >= polish_deadline:
                        break
                    old_val = current_set[i]
                    best_new_val = None
                    best_new_score = current_score
                    for new_val in range(1, 513):
                        if new_val == old_val or new_val in current_set:
                            continue
                        new_bit = current_bit
                        new_bit &= ~(1 << (old_val - 1))
                        new_bit |= (1 << (new_val - 1))
                        new_score = score_bitset(new_bit)
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_new_val = new_val
                    if best_new_val is not None:
                        current_set[i] = best_new_val
                        current_bit = get_bitset(current_set)
                        current_score = best_new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current_set)
                            best_bit = current_bit
    
    # Final global polish on best solution found
    if best_set is not None and time.time() < deadline - 2:
        final_set = list(best_set)
        final_bit = best_bit
        final_score = best_score
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(32):
                if time.time() >= deadline - 2:
                    break
                old_val = final_set[i]
                best_new_val = None
                best_new_score = final_score
                for new_val in range(1, 513):
                    if new_val == old_val or new_val in final_set:
                        continue
                    new_bit = final_bit
                    new_bit &= ~(1 << (old_val - 1))
                    new_bit |= (1 << (new_val - 1))
                    new_score = score_bitset(new_bit)
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                if best_new_val is not None:
                    final_set[i] = best_new_val
                    final_bit = get_bitset(final_set)
                    final_score = best_new_score
                    improved = True
        
        if final_score > best_score:
            best_set = final_set
            best_bit = final_bit
            best_score = final_score
    
    if best_set is None:
        best_set = list(range(1, 33))
        
    return best_set
