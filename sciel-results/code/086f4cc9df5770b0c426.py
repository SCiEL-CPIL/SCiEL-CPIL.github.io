import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
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
    
    # Initialize
    current_set = random_set()
    current_bit = get_bitset(current_set)
    current_score = score_bitset(current_bit)
    
    best_set = current_set
    best_bit = current_bit
    best_score = current_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.99999
    
    while time.time() < deadline and T > T_min:
        # Generate neighbor: swap one element
        idx = random.randint(0, 31)
        old_val = current_set[idx]
        new_val = random.randint(1, 512)
        
        if new_val in current_set:
            continue
            
        # Update bitset
        new_bit = current_bit
        new_bit &= ~(1 << (old_val - 1))
        new_bit |= (1 << (new_val - 1))
        
        # Check if it's a valid set (distinctness is maintained by logic above, 
        # but we should ensure we didn't accidentally create a duplicate in the list logic if we were modifying list directly.
        # Here we are just checking bitset validity implicitly by construction.
        # However, we need to update the list for tracking.
        
        new_score = score_bitset(new_bit)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            # Accept
            current_set[idx] = new_val
            current_bit = new_bit
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
                best_bit = current_bit
                
        T *= cooling_rate
        
    return best_set
