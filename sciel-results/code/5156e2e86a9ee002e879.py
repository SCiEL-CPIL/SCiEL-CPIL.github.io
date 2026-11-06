import time
import random
from itertools import combinations

def is_sidon(subset):
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def has_sidon_k(A, k):
    # Check if any k-element subset is Sidon
    for comb in combinations(A, k):
        if is_sidon(list(comb)):
            return True
    return False

def max_sidon_size(A):
    # Check from high to low to find the maximum size
    # Max possible is 12, but we only care about distinguishing 3, 4, 5...
    # If no 4-set is Sidon, max is 3.
    # If no 5-set is Sidon but 4 is, max is 4.
    # etc.
    
    # Optimization: If we just want to minimize, and 3 is the target,
    # we can check 4 first.
    if not has_sidon_k(A, 4):
        return 3
    
    if not has_sidon_k(A, 5):
        return 4
        
    if not has_sidon_k(A, 6):
        return 5
        
    if not has_sidon_k(A, 7):
        return 6
        
    # Fallback for larger sizes, though unlikely to be optimal
    # For the purpose of local search, distinguishing 3, 4, 5 is key.
    # If it's >= 5, we treat it as "bad" and try to reduce.
    # We can just return a higher number to indicate "worse".
    return 7 # Placeholder for >6, effectively "high cost"

def construct():
    start_time = time.time()
    deadline = start_time + 55 # Safety margin
    
    # Initial set: dense interval to maximize collisions
    # A dense interval like [0, 11] has many sum collisions.
    # Example: 0+3 = 1+2. So {0,1,2,3} is NOT Sidon.
    # In fact, any 4 consecutive integers contain a collision?
    # {0,1,2,3}: sums: 0,1,2,3,4,5,6. Wait.
    # 0+0=0, 0+1=1, 0+2=2, 0+3=3
    # 1+1=2 (collision with 0+2!)
    # So {0,1,2,3} is not Sidon.
    # It turns out that [0, n] often has small max Sidon size.
    
    A = list(range(12))
    best_A = A[:]
    best_score = max_sidon_size(best_A)
    
    # Local Search
    while time.time() < deadline:
        idx = random.randint(0, 11)
        old_val = A[idx]
        
        # Generate a new value
        # To maintain density, maybe pick from a local range?
        # Or just random in [0, 128].
        # Random is safer to escape local minima.
        new_val = random.randint(0, 128)
        
        if new_val in A:
            continue
            
        A[idx] = new_val
        
        current_score = max_sidon_size(A)
        
        if current_score < best_score:
            best_score = current_score
            best_A = A[:]
            if best_score == 3:
                break # Optimal found
        else:
            # Revert if worse
            A[idx] = old_val
            
    return best_A
