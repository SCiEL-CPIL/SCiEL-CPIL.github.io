import time
from fractions import Fraction

# Global variables for the search
N = 256
best_sum = Fraction(0, 1)
best_set = set()
start_time = 0
time_limit = 55.0 # Leave margin for 60s limit

# Precompute suffix sums of reciprocals
suffix_sum = [0] * (N + 2)
for i in range(N, 0, -1):
    suffix_sum[i] = suffix_sum[i + 1] + Fraction(1, i)

def is_conflict(x, S_mask):
    """
    Check if adding x to set represented by S_mask creates a 3-term arithmetic progression.
    S_mask is an integer where bit i is 1 if i is in the set.
    """
    # Case 1: x is the middle term. u, x, w => 2x = u + w
    # u, w must be in S.
    m = S_mask
    while m:
        lsb = m & -m
        u = lsb.bit_length() - 1
        w = 2 * x - u
        if 1 <= w <= N:
            if (S_mask >> w) & 1:
                return True
        m ^= lsb
        
    # Case 2: x is the first term. x, v, w => 2v = x + w => w = 2v - x
    # v, w must be in S.
    m = S_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        w = 2 * v - x
        if 1 <= w <= N:
            if (S_mask >> w) & 1:
                return True
        m ^= lsb
        
    # Case 3: x is the last term. u, v, x => 2v = u + x => u = 2v - x
    # u, v must be in S.
    m = S_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        u = 2 * v - x
        if 1 <= u <= N:
            if (S_mask >> u) & 1:
                return True
        m ^= lsb
            
    return False

def search(x, S_mask, current_sum):
    global best_sum, best_set, start_time, time_limit
    
    # Time check
    if time.time() - start_time > time_limit:
        return
        
    # If we are at the end
    if x > N:
        if current_sum > best_sum:
            best_sum = current_sum
            # Reconstruct set from mask
            best_set = set()
            m = S_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                best_set.add(idx)
                m ^= lsb
        return

    # Calculate upper bound for pruning
    if current_sum + suffix_sum[x] <= best_sum:
        return

    # Option 1: Include x
    if not is_conflict(x, S_mask):
        new_mask = S_mask | (1 << x)
        new_sum = current_sum + Fraction(1, x)
        search(x + 1, new_mask, new_sum)
        
    # Option 2: Exclude x
    search(x + 1, S_mask, current_sum)

def construct():
    global start_time, best_sum, best_set
    start_time = time.time()
    best_sum = Fraction(0, 1)
    best_set = set()
    
    # Initial call
    search(1, 0, Fraction(0, 1))
    
    # Ensure non-empty
    if not best_set:
        return [1]
        
    return list(best_set)
