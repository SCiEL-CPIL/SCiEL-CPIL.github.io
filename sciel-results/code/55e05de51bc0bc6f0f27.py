import time
from fractions import Fraction

# Global variables for the search
N = 256
best_sum = Fraction(0, 1)
best_set = set()
start_time = 0
time_limit = 55.0 # Leave margin for 60s limit

def is_conflict(x, S):
    """
    Check if adding x to set S creates a 3-term arithmetic progression.
    S is a set of integers.
    """
    # Case 1: x is the middle term. u, x, w => 2x = u + w
    # u, w must be in S.
    # Iterate through pairs in S? Or iterate through one and check if 2x - u is in S?
    for u in S:
        w = 2 * x - u
        if w in S and w != u: # w != u ensures non-constant progression
            return True
            
    # Case 2: x is the first term. x, v, w => 2v = x + w => w = 2v - x
    # v, w must be in S.
    for v in S:
        w = 2 * v - x
        if w in S and w != v:
            return True
            
    # Case 3: x is the last term. u, v, x => 2v = u + x => u = 2v - x
    # u, v must be in S.
    for v in S:
        u = 2 * v - x
        if u in S and u != v:
            return True
            
    return False

def search(x, current_set, current_sum):
    global best_sum, best_set, start_time, time_limit
    
    # Time check
    if time.time() - start_time > time_limit:
        return
        
    # Pruning: Upper bound
    # The maximum additional sum we can get is sum of 1/i for all i from x to N
    # that are not in current_set.
    # This is a loose bound but cheap to calculate if we precompute suffix sums.
    # Let's precompute suffix_sum[i] = sum(1/k for k in range(i, N+1))
    # But we must exclude elements already in current_set.
    # A tighter bound: sum(1/k for k in range(x, N+1) if k not in current_set)
    # This is O(N-x) per node. N=256, so max 256 ops. Acceptable.
    
    # If we are at the end
    if x > N:
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = set(current_set)
        return

    # Calculate upper bound for pruning
    # Optimization: Precompute suffix sums of reciprocals for the full range [1, N]
    # suffix_sum[i] = 1/i + 1/(i+1) + ... + 1/N
    # The max possible sum from x onwards is suffix_sum[x].
    # If current_sum + suffix_sum[x] <= best_sum, we can prune.
    # This ignores that some numbers in [x, N] are already in current_set (impossible as we add in order? No, we add in order x, x+1... so current_set only contains < x).
    # Wait, if we iterate x from 1 to N, and decide include/exclude, then current_set only contains numbers < x.
    # So all numbers in [x, N] are NOT in current_set.
    # Therefore, the upper bound is exactly current_sum + suffix_sum[x].
    
    if current_sum + suffix_sum[x] <= best_sum:
        return

    # Option 1: Include x
    if not is_conflict(x, current_set):
        new_set = set(current_set)
        new_set.add(x)
        new_sum = current_sum + Fraction(1, x)
        search(x + 1, new_set, new_sum)
        
    # Option 2: Exclude x
    search(x + 1, current_set, current_sum)

# Precompute suffix sums of reciprocals
suffix_sum = [0] * (N + 2)
for i in range(N, 0, -1):
    suffix_sum[i] = suffix_sum[i + 1] + Fraction(1, i)

def construct():
    global start_time, best_sum, best_set
    start_time = time.time()
    best_sum = Fraction(0, 1)
    best_set = set()
    
    # Initial call
    search(1, set(), Fraction(0, 1))
    
    # Ensure non-empty (task requires nonempty subset)
    # The search will always find at least one number (e.g., 1 is always valid initially)
    # If best_set is empty, add 1 as fallback
    if not best_set:
        return [1]
        
    return list(best_set)

# EVOLVE-BLOCK-START
def construct():
    global start_time, best_sum, best_set
    start_time = time.time()
    best_sum = Fraction(0, 1)
    best_set = set()
    
    # Initial call
    search(1, set(), Fraction(0, 1))
    
    # Ensure non-empty
    if not best_set:
        return [1]
        
    return list(best_set)
# EVOLVE-BLOCK-END
