import time
import random

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    N = 512
    K = 32
    
    def set_to_bits(s):
        bits = 0
        for x in s:
            bits |= (1 << (x - 1))
        return bits
            
    def bits_to_set(bits):
        return [i + 1 for i in range(N) if (bits >> i) & 1]

    def score_bits(bits):
        count = 0
        for d in range(1, 256):
            if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
                count += 1
        return count

    # 1. Greedy Initialization
    S = set()
    current_bits = 0
    # We can speed up greedy by noting that score is monotonic? No, adding an element can reduce the number of distinct d's?
    # No, adding an element can only add new APs or keep existing ones. It cannot remove APs.
    # So score is non-decreasing with set inclusion.
    # Thus, greedy is safe.
    
    for _ in range(K):
        best_x = -1
        best_score = -1
        # Optimization: We can precompute candidate scores? No, S changes.
        for x in range(1, N + 1):
            if x in S:
                continue
            new_bits = current_bits | (1 << (x - 1))
            sc = score_bits(new_bits)
            if sc > best_score:
                best_score = sc
                best_x = x
        
        if best_x != -1:
            S.add(best_x)
            current_bits |= (1 << (best_x - 1))
            
    current_score = best_score
    best_score = current_score
    best_bits = current_bits
    current_set = S
    
    # 2. Simulated Annealing Polish
    if time.time() - start_time > time_limit - 5:
        # Not enough time for SA, return greedy result
        return bits_to_set(best_bits)
        
    T_start = 5.0
    T_end = 0.01
    # Estimate remaining time
    remaining_time = time_limit - (time.time() - start_time)
    
    # We don't know exact iterations, so we use time-based cooling in the loop
    # Or just a fixed number of iterations if we can estimate speed.
    # Let's just loop until time limit.
    
    # To make SA effective, we need to track current state in a way that allows fast swaps.
    # current_set is a set.
    
    while time.time() - start_time < time_limit:
        # Pick random x in S, y not in S
        # Using list conversion is O(K). K=32.
        x = random.choice(list(current_set))
        y = random.randint(1, N)
        
        if y in current_set:
            continue
            
        # Compute new bits
        new_bits = current_bits
        new_bits &= ~(1 << (x - 1))
        new_bits |= (1 << (y - 1))
        
        new_score = score_bits(new_bits)
        delta = new_score - current_score
        
        # Simple Hill Climbing with occasional random restarts or SA
        # Given the time, SA is better.
        # Calculate T based on time elapsed in SA phase?
        # Let's just use a simple geometric cooling over a fixed number of steps, 
        # but since we don't know steps, we'll use a very slow cooling or just hill climbing with random kicks.
        
        # Actually, for this small problem, pure hill climbing often gets stuck.
        # Let's use a simple acceptance criterion: accept if delta > 0.
        # If no improvement for a while, do a random restart?
        # Or use SA with T decaying based on wall clock time.
        
        # Let's calculate T from time
        t_elapsed = time.time() - start_time
        # Normalize t_elapsed to 0..1 over the total time_limit
        progress = t_elapsed / time_limit
        # T = T_start * (T_end/T_start)^progress
        # This ensures T reaches T_end at time_limit
        T = T_start * pow(T_end / T_start, progress)
        
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            # Accept
            current_bits = new_bits
            current_set.remove(x)
            current_set.add(y)
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_bits = current_bits

    return bits_to_set(best_bits)
