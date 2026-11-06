"""Optimized search for sum-product extremal sets."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def _evaluate(A):
    """Compute max(|A+A|, |A*A|) for set A (list of distinct positive ints)."""
    A = list(A)
    n = len(A)
    # Sumset A+A with equal summands allowed
    sums = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
    # Product set A*A with equal factors allowed
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _evaluate_fast(A):
    """Faster evaluation using numpy where possible, but sets are fine for n=20."""
    A = np.array(A, dtype=np.int64)
    n = len(A)
    # Sumset
    i_idx, j_idx = np.triu_indices(n)
    sums = A[i_idx] + A[j_idx]
    n_sums = len(np.unique(sums))
    # Product set
    prods = A[i_idx] * A[j_idx]
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

def _generate_geometric(a, r_num, r_den, n=20, max_val=10000):
    """Generate a set from a geometric progression a*(r_num/r_den)^k, rounded to ints.
    Returns None if not all distinct and within bounds."""
    A = []
    for k in range(n):
        val = a * (r_num ** k) // (r_den ** k)
        # Use exact rational arithmetic to avoid float issues
        # Actually, let's use Fraction or just careful integer math
        # val = a * r_num^k / r_den^k
        # We need this to be integer. Let's just generate and round.
        A.append(val)
    A = sorted(set(A))
    if len(A) < n:
        return None
    if A[-1] > max_val:
        return None
    if A[0] < 1:
        return None
    # Need exactly n distinct. If we have more, take first n? No, the structure is fixed.
    # If we have fewer, invalid.
    if len(A) != n:
        # Try to fill in? No, this is just a generator.
        # If len(A) < n, return None.
        # If len(A) > n, it's not a pure GP anymore due to collisions.
        # The task requires 20 distinct. If collisions happen, we have <20.
        return None
    return A

def _local_search(A, time_limit, rng):
    """Hill climbing / simulated annealing style local search."""
    A = list(A)
    current_score = _evaluate(A)
    best_A = list(A)
    best_score = current_score
    start_time = time.time()
    
    # Neighborhood: swap an element with another, or perturb
    # We can replace an element with a random valid integer, or swap two.
    # Better: perturb one element by +/- delta, ensuring distinctness and bounds.
    
    while time.time() - start_time < time_limit:
        i = rng.randint(0, len(A) - 1)
        # Choose a perturbation
        # Option 1: replace A[i] with a random value
        # Option 2: A[i] += delta
        # Let's try replacing with a value near A[i] or random.
        # Random value in [1, 10000]
        new_val = rng.randint(1, 10000)
        if new_val in A:
            continue
        
        old_val = A[i]
        A[i] = new_val
        new_score = _evaluate(A)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = list(A)
        else:
            # Simulated annealing acceptance?
            # For now, just reject if worse (hill climbing)
            # To escape local minima, occasionally accept worse.
            # Let's add a simple SA mechanism
            T = max(0.1, 1.0 * (1 - (time.time() - start_time) / time_limit))
            if rng.random() < np.exp(-(new_score - current_score) / T):
                current_score = new_score
            else:
                A[i] = old_val
        
        # Check deadline
        if time.time() - start_time > time_limit:
            break
            
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    rng = random.Random(42)
    
    best_A = None
    best_score = float('inf')
    
    # Strategy 1: Grid search over rational geometric progressions
    # A = { a * (p/q)^k rounded to int }
    # We need 20 distinct integers.
    # Small p, q.
    # p > q for increasing.
    # a * (p/q)^19 <= 10000
    
    # Let's try specific structures known to have small sum/product sets.
    # e.g. powers of 2: {1, 2, 4, ..., 2^19} -> product set is small (powers of 2), sum set is large (all sums of distinct powers, which is just binary representation, so 2^20 - 1 sums? No, sumset of powers of 2 is all integers from 2^0+2^0=2 to 2^20-1. Size is 2^20-2. That's huge.
    # We want BOTH sum and product sets to be small.
    # This is the sum-product problem.
    # Small sets with small sum and product sets are rare.
    # Arithmetic progressions have small sum sets (size 3n-1) but large product sets.
    # Geometric progressions have small product sets (size ~n^2/2) but large sum sets.
    # We need a balance.
    
    # Let's start with an arithmetic progression: {1, 2, ..., 20}
    # Sumset size: 39. Productset size: ?
    # 1*1=1, ..., 20*20=400.
    # Many collisions? 1*20=20, 2*10=20, 4*5=20.
    # Let's evaluate {1..20}.
    A_init = list(range(1, 21))
    score_init = _evaluate(A_init)
    if score_init < best_score:
        best_score = score_init
        best_A = list(A_init)
        
    # Try some other initializations
    # Geometric with small ratio
    # a=1, r=2: {1,2,4,...,2^19} -> 2^19 = 524288 > 10000. Too big.
    # a=1, r=1.2? Not integer.
    # a=1, p=6, q=5: 1, 1.2, 1.44...
    # Let's just run local search from a few diverse starts.
    
    starts = [
        list(range(1, 21)),
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 100, 200, 300, 400, 500, 600], # mix
        [1, 10, 100, 1000, 2, 20, 200, 2000, 3, 30, 300, 3000, 4, 40, 400, 4000, 5, 50, 500, 5000], # grid
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    ]
    
    # Remove invalid starts
    valid_starts = []
    for s in starts:
        s = sorted(set(s))
        if len(s) == 20 and s[0] >= 1 and s[-1] <= 10000:
            valid_starts.append(s)
        else:
            # Try to fix
            if len(s) < 20:
                # Fill in
                existing = set(s)
                for i in range(1, 10001):
                    if i not in existing:
                        s.append(i)
                        existing.add(i)
                        if len(s) == 20:
                            break
                s = sorted(s)
            if len(s) == 20 and s[-1] <= 10000 and s[0] >= 1:
                valid_starts.append(s)
                
    if not valid_starts:
        valid_starts = [list(range(1, 21))]
        
    # Run local search from each start
    # Budget: 80 seconds total.
    # Split time among starts? Or just run one until deadline?
    # Let's run multiple short searches.
    
    num_starts = len(valid_starts)
    time_per_start = 70.0 / num_starts  # Leave 10s for safety/eval
    
    for s in valid_starts:
        if time.time() > deadline - 5:
            break
        A, score = _local_search(s, time_per_start, rng)
        if score < best_score:
            best_score = score
            best_A = A
            
    # Final polish: run one more local search from the best found
    if time.time() < deadline - 10 and best_A is not None:
        A, score = _local_search(best_A, 10.0, rng)
        if score < best_score:
            best_score = score
            best_A = A
            
    # Ensure best_A is valid
    if best_A is None:
        best_A = list(range(1, 21))
        
    # Final check
    best_A = sorted(set(best_A))
    if len(best_A) != 20:
        # Fallback
        best_A = list(range(1, 21))
        
    return best_A
# EVOLVE-BLOCK-END
