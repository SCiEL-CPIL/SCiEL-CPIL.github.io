import itertools
import time
import random

def is_sidon(S):
    sums = set()
    for i, x in enumerate(S):
        for j, y in enumerate(S):
            if i <= j:
                s = x + y
                if s in sums:
                    return False
                sums.add(s)
    return True

def has_sidon_subset_of_size(S, k):
    for subset in itertools.combinations(S, k):
        if is_sidon(subset):
            return True
    return False

def max_sidon_size(S):
    n = len(S)
    # We only care about distinguishing 4 vs 5 vs >5
    # But let's get exact.
    # Start from 5 down to 1?
    # If we find size 5, we can stop? No, we need max.
    # But for optimization, if we find 5, we know it's not <5.
    # If we want to minimize, we prefer 4 over 5.
    
    # Let's check 5 first.
    if has_sidon_subset_of_size(S, 5):
        # It could be 6, 7, etc.
        # But for most random sets, 5 is likely the max or close.
        # Let's check 6, 7... up to n
        for k in range(6, n + 1):
            if has_sidon_subset_of_size(S, k):
                return k
        return 5
    else:
        # Max is <= 4
        if has_sidon_subset_of_size(S, 4):
            return 4
        if has_sidon_subset_of_size(S, 3):
            return 3
        return 2 # Any 2 distinct ints form a Sidon set?
        # x1+x1, x1+x2, x2+x2.
        # 2x1, x1+x2, 2x2.
        # If x1 != x2, these are distinct?
        # 2x1 = x1+x2 => x1=x2 (False)
        # 2x1 = 2x2 => x1=x2 (False)
        # x1+x2 = 2x2 => x1=x2 (False)
        # So yes, any 2 distinct integers form a Sidon set.
        # So min is 2.

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best_set = list(range(12))
    best_size = max_sidon_size(best_set)
    
    # If best_size is 4, we might try to find 3.
    # But 3 is very hard. 4 is the target.
    
    # Random search with local improvement
    # We will use a simple hill climbing / random restart
    
    for restart in range(500):
        if time.time() > deadline:
            break
            
        # Generate random set
        curr = set()
        while len(curr) < 12:
            curr.add(random.randint(0, 128))
        curr = list(curr)
        
        curr_size = max_sidon_size(curr)
        
        if curr_size < best_size:
            best_size = curr_size
            best_set = curr[:]
            if best_size <= 3:
                return best_set
        
        # Local search
        # Try to reduce size by changing one element
        for step in range(50):
            if time.time() > deadline:
                break
            i = random.randint(0, 11)
            old_val = curr[i]
            # Try a few new values
            for _ in range(5):
                new_val = random.randint(0, 128)
                if new_val in curr:
                    continue
                curr[i] = new_val
                new_size = max_sidon_size(curr)
                if new_size < curr_size:
                    curr_size = new_size
                    if curr_size < best_size:
                        best_size = curr_size
                        best_set = curr[:]
                        if best_size <= 3:
                            return best_set
                    break
                else:
                    curr[i] = old_val
                    
    return best_set
