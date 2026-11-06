import time
import random
import math
import bisect

def _score(S_list, S_bits):
    """
    S_list must be sorted.
    S_bits is a bitset integer where bit i is 1 if i is in S.
    Returns count of distinct d > 0 such that a, a+d, a+2d in S.
    """
    count = 0
    for d in range(1, 256):
        if (S_bits & (S_bits >> d) & (S_bits >> (2 * d))) != 0:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    N = 32
    MAXV = 512
    
    best_score = -1
    best_list = []
    
    # Run multiple restarts if time permits
    while time.time() < deadline:
        # Initialize random set
        current_list = random.sample(range(1, MAXV + 1), N)
        current_list.sort()
        current_bits = 0
        for x in current_list:
            current_bits |= (1 << x)
        
        current_score = _score(current_list, current_bits)
        
        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
        
        # Simulated Annealing
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        
        # Local time limit for this restart
        local_deadline = start_time + min(5.0, deadline - start_time)
        
        while time.time() < local_deadline:
            # Pick x to remove
            x_idx = random.randint(0, N - 1)
            x = current_list[x_idx]
            
            # Pick y to add
            while True:
                y = random.randint(1, MAXV)
                if not ((current_bits >> y) & 1):
                    break
            
            # Update bits
            new_bits = current_bits ^ (1 << x) ^ (1 << y)
            
            # Update list
            # Remove x
            # Since list is sorted, we can find x quickly? 
            # We have x_idx, so we can just replace and sort? 
            # Or use bisect.
            # Since N is small, replacing and sorting is fine, but 
            # removing and inserting is O(N).
            
            # Remove x from list
            # current_list[x_idx] is x
            # Shift elements? No, just create new list or use slice.
            # Slicing is fast in C.
            new_list = current_list[:x_idx] + current_list[x_idx+1:]
            # Insert y in sorted order
            ins_idx = bisect.bisect_left(new_list, y)
            new_list.insert(ins_idx, y)
            
            new_score = _score(new_list, new_bits)
            
            delta = new_score - current_score
            
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                current_list = new_list
                current_bits = new_bits
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min

    return best_list
