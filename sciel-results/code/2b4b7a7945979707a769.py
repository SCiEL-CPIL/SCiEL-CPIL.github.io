import time
import random

START_TIME = time.time()
DEADLINE = 55.0

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def check_validity(sums, n):
    """
    Check if the bitsets in 'sums' represent a valid cardinality-separated set.
    sums[k] is an integer where bit s is 1 if sum s is achievable with k elements.
    """
    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            if sums[i] & sums[j]:
                return False
    return True

def construct():
    best_set = list(range(226, 257)) # n=31, valid AP
    best_size = 31
    
    # Try to find n=32
    # We need 32 distinct ints in 1..256.
    # Minimal range 31 (consecutive) failed.
    # Try range 32 (one gap).
    # Start with base A.
    # Iterate A from 1 to 256-32.
    
    for A in range(1, 256 - 32 + 1):
        if is_done():
            break
        # Generate all combinations of 32 elements in [A, A+32]?
        # That's C(33, 32) = 33 possibilities (which element to skip).
        # Plus we can shift the window.
        
        # Actually, just try to add one element to a valid 31-set?
        # Or start with a candidate 32-set.
        
        # Let's try skipping one element in [A, A+32]
        for skip in range(33):
            if is_done():
                break
            # Candidate set
            cand = [i for i in range(A, A+33) if i != A + skip]
            if len(cand) != 32:
                continue
            
            # Check validity
            # Use bitset DP for speed
            max_sum = sum(cand)
            # Bitset for sums
            # sums[k] is a bitset of achievable sums with k elements
            sums = [0] * (33)
            sums[0] = 1 # sum 0
            
            for x in cand:
                # Update sums in reverse to avoid using x multiple times in same step?
                # No, we are building subsets.
                # Standard knapsack style update.
                for k in range(31, -1, -1):
                    if sums[k]:
                        sums[k+1] |= (sums[k] << x)
            
            if check_validity(sums, 32):
                return cand
                
    # If no 32 found, return best 31
    return best_set

# EVOLVE-BLOCK-END
