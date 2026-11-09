import random
import time
from itertools import combinations

def is_sidon_4(s):
    # s is a list of 4 distinct integers
    # Check if all pairwise sums (i<=j) are distinct
    sums = set()
    for i in range(4):
        for j in range(i, 4):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def count_sidon_4_subsets(s):
    # s is a list of 12 integers
    count = 0
    for subset in combinations(s, 4):
        if is_sidon_4(subset):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_set = None
    best_count = float('inf')
    
    # Try different ranges
    # Smaller ranges favor collisions
    ranges = [20, 25, 30, 40, 50, 128]
    
    for R in ranges:
        if time.time() > deadline:
            break
            
        # Random search in this range
        # Estimate number of tries based on time
        # We can just loop until deadline
        
        while time.time() < deadline:
            # Random sample
            s = random.sample(range(R + 1), 12)
            s.sort()
            
            c = count_sidon_4_subsets(s)
            
            if c == 0:
                return s
            
            if c < best_count:
                best_count = c
                best_set = s[:]
                
            # Local search: try to break a Sidon-4
            # Find one Sidon-4 subset
            target_subset = None
            for subset in combinations(s, 4):
                if is_sidon_4(subset):
                    target_subset = subset
                    break
            
            if target_subset:
                # Try to move one element of the target subset to break it
                # Or move any element
                # Pick a random element from the whole set to move
                idx_to_move = random.randint(0, 11)
                old_val = s[idx_to_move]
                
                # Try to move it to a new value
                # Prefer values that cause collisions with existing sums
                # Or just random in a small neighborhood
                moved = False
                for _ in range(10): # Try a few moves
                    new_val = random.randint(0, R)
                    if new_val not in s:
                        s[idx_to_move] = new_val
                        s.sort()
                        new_c = count_sidon_4_subsets(s)
                        if new_c < c:
                            c = new_c
                            moved = True
                            if c == 0:
                                return s
                            if c < best_count:
                                best_count = c
                                best_set = s[:]
                            break
                        else:
                            # Revert
                            s[idx_to_move] = old_val
                            s.sort()
                
                if not moved:
                    # Revert if no improvement
                    s[idx_to_move] = old_val
                    s.sort()
                    
    return best_set if best_set else list(range(12))
