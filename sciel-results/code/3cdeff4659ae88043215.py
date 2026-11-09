import time
import random
from itertools import combinations

def is_sidon_4(subset):
    # subset is a tuple of 4 distinct integers
    # Check if all unordered pair sums (including diagonals) are unique
    sums = set()
    for i in range(4):
        for j in range(i, 4):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def count_sidon_4(a_set):
    count = 0
    for comb in combinations(a_set, 4):
        if is_sidon_4(comb):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin
    
    best_set = list(range(12))
    best_count = count_sidon_4(best_set)
    
    # Target: Find a set with 0 Sidon-4 subsets.
    # If found, max Sidon size is likely 3.
    
    # Strategy:
    # 1. Try multiple random initializations in small ranges.
    # 2. For each, run local search to minimize Sidon-4 count.
    # 3. If count becomes 0, return immediately.
    
    # Ranges to try
    ranges = [20, 25, 30, 40, 50]
    
    # Number of restarts per range
    # We have ~58 seconds.
    # count_sidon_4 is fast (C(12,4) = 495 checks).
    # Local search step: move one element, recount.
    
    total_restarts = 0
    max_restarts = 5000 # Upper bound, will be limited by time
    
    while time.time() < deadline:
        total_restarts += 1
        
        # Pick a range, biased towards smaller
        R = random.choice(ranges)
        
        # Generate initial set
        s = random.sample(range(R + 1), 12)
        s.sort()
        
        c = count_sidon_4(s)
        if c == 0:
            return s
            
        if c < best_count:
            best_count = c
            best_set = s[:]
            
        # Local search
        # Try to reduce c by moving one element
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Pick a random index
            idx = random.randint(0, 11)
            old_val = s[idx]
            
            # Try a few new values
            # Bias towards values that might cause collisions
            # Just try random values in the range
            for _ in range(20):
                new_val = random.randint(0, R)
                if new_val == old_val or new_val in s:
                    continue
                
                s[idx] = new_val
                s.sort()
                new_c = count_sidon_4(s)
                
                if new_c < c:
                    c = new_c
                    improved = True
                    if c == 0:
                        return s
                    if c < best_count:
                        best_count = c
                        best_set = s[:]
                    break
                else:
                    # Revert
                    s[idx] = old_val
                    s.sort()
            
            # If no improvement in this batch, try swapping two elements?
            # Or just accept and try next restart if stuck.
            # To avoid getting stuck in local minima, we can do a few random jumps.
            if not improved:
                # Try a random jump: move 2 elements randomly
                if random.random() < 0.1:
                    idx1 = random.randint(0, 11)
                    idx2 = random.randint(0, 11)
                    if idx1 != idx2:
                        old1 = s[idx1]
                        old2 = s[idx2]
                        new1 = random.randint(0, R)
                        new2 = random.randint(0, R)
                        if new1 not in s and new2 not in s and new1 != new2:
                            s[idx1] = new1
                            s[idx2] = new2
                            s.sort()
                            new_c = count_sidon_4(s)
                            if new_c < c:
                                c = new_c
                                if c < best_count:
                                    best_count = c
                                    best_set = s[:]
                            else:
                                # Revert
                                s[idx1] = old1
                                s[idx2] = old2
                                s.sort()

    return best_set
