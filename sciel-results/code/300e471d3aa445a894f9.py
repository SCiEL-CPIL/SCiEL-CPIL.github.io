"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    def greedy_random():
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        chosen = []
        sum_counts = [0] * 2001  # Max sum 1998
        
        for x in candidates:
            valid = True
            # Check conflicts
            # We need to check sums x+y for all y in chosen, and 2*x
            # If any of these sums already has count 2, adding x would make it 3.
            
            # Optimization: Check 2*x first
            if sum_counts[2 * x] == 2:
                valid = False
            
            if valid:
                for y in chosen:
                    s = x + y
                    if sum_counts[s] == 2:
                        valid = False
                        break
            
            if valid:
                # Update counts
                for y in chosen:
                    sum_counts[x + y] += 1
                sum_counts[2 * x] += 1
                chosen.append(x)
        
        return chosen

    # Run restarts
    # We can also try different strategies:
    # 1. Random shuffle
    # 2. Sort by random priority
    # Random shuffle is equivalent to random priority.
    
    # To diversify, we can bias the shuffle?
    # e.g., random.shuffle is uniform.
    
    # Let's just run many uniform random restarts.
    
    # Initial run
    res = greedy_random()
    best_size = len(res)
    best_set = res
    
    while time.time() < deadline:
        res = greedy_random()
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
            # If we find a very large set, we might want to continue to see if we can beat it.
            # But if we are close to the deadline, we should stop.
            
    return best_set
# EVOLVE-BLOCK-END
