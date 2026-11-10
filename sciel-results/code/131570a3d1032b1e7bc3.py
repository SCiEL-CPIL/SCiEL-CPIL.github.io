import time
import math

N = 5000
TIME_LIMIT = 330

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def construct():
    start_time = time.time()
    
    # Branching on first 5 primes: 2, 3, 5, 7, 11
    # Choices: for each prime p, choose a residue r to remove.
    # We remove ALL x in S such that x % p == r.
    
    best_size = 0
    best_set = []
    
    # Primes to branch on
    branch_primes = [2, 3, 5, 7, 11]
    # Primes to process greedily afterwards
    # We need to process primes p such that p <= current_size
    # Since current_size will be < N, we can just iterate through all primes 
    # and stop if p > current_size.
    
    # To avoid re-computing full set for each of 2310 branches from scratch 
    # (which is 2310 * 5000 ops = 11.5M ops, very fast), we can just do it.
    
    # However, we can be smarter.
    # Start with full set [0, N]
    # For each combination of removals for branch_primes:
    #   S = [0..N]
    #   for p, r in zip(branch_primes, choice):
    #       S = [x for x in S if x % p != r]
    #   # Now apply greedy for remaining primes
    #   for p in PRIMES:
    #       if p <= len(S):
    #           # Check if all residues covered
    #           residues = {}
    #           for x in S:
    #               residues[x % p] = residues.get(x % p, 0) + 1
    #           if len(residues) == p:
    #               # Find min count residue
    #               min_res = min(residues, key=residues.get)
    #               S = [x for x in S if x % p != min_res]
    #       else:
    #           break
    #   if len(S) > best_size:
    #       best_size = len(S)
    #       best_set = S
    #       if time.time() - start_time > TIME_LIMIT - 10:
    #           break

    # This is O(2310 * (5000 + 5000 + ...))
    # The list comprehensions are efficient in Python.
    
    # Let's optimize the inner loop.
    # Instead of list comprehensions which create new lists, 
    # we can use a boolean mask?
    # List of 5000 ints is small.
    
    # We need to iterate over all combinations of residues for branch_primes.
    # itertools.product
    
    import itertools
    
    # We also need to consider that we might not need to branch on ALL 5.
    # Maybe branching on 2,3,5 is enough?
    # The prompt says branching on first 5 improved score.
    
    # Let's add a safety check for time.
    
    # One optimization: 
    # The order of greedy primes matters?
    # Usually smaller primes first.
    
    # Let's run it.
    
    # Note: The set elements must be distinct integers in [0, N].
    # The input is [0, N], so they are distinct.
    
    # We return a list.
    
    # Potential issue: 
    # If we remove classes for 2,3,5,7,11, the size drops significantly.
    # 5000 * (1/2) * (2/3) * (4/5) * (6/7) * (10/11)
    # = 5000 * 0.5 * 0.66 * 0.8 * 0.857 * 0.909
    # = 5000 * 0.256 = 1280.
    # Then we process p=13.
    # 13 <= 1280.
    # We check residues mod 13.
    # We remove smallest class.
    # Size drops by ~1/13.
    # This continues.
    
    # The "greedy" part after branching is deterministic.
    
    # Is there any other freedom?
    # Maybe we can try different orders of greedy primes?
    # Unlikely to help much.
    
    # Let's implement the branching.
    
    # To speed up, we can pre-compute the effect of branching?
    # No, 2310 is small enough.
    
    # Let's write the loop.
    
    # We need to be careful with the time limit.
    # If 2310 iterations take too long, we reduce branching depth.
    # 2310 * (cost of one greedy run).
    # Cost of one greedy run:
    #   Branching part: 5 list comprehensions on ~5000 elements. ~25k ops.
    #   Greedy part: ~50 primes. Each involves counting (5000 ops) and filtering (5000 ops).
    #   50 * 10k = 500k ops.
    #   Total per branch ~ 500k ops.
    #   2310 * 500k = 1.1 billion ops.
    #   Python does ~10-50 million ops/sec.
    #   1.1e9 / 1e7 = 110 seconds.
    #   This is close to the 330s limit.
    #   We need to optimize or reduce branching.
    
    # Optimization:
    # 1. Use numpy?
    #    Numpy array of 5000 ints.
    #    Filtering is fast.
    # 2. Reduce branching to first 4 primes (210 combos).
    #    210 * 500k = 105 million ops. ~10-20 seconds. Safe.
    # 3. Use a boolean mask and index array to avoid creating new lists?
    
    # Let's try Numpy.
    
    import numpy as np
    
    # Initial array
    base_arr = np.arange(N + 1)
    
    # We will process branches.
    # To manage time, we will try first 5 primes, but break if time is low.
    
    # We can also randomize the order of branches or just stop.
    
    # Let's use a hybrid: 
    # 1. Run greedy on full set (1 run) to get a baseline.
    # 2. Run branching on first 4 primes (210 runs).
    # 3. If time permits, run branching on first 5 primes (2310 runs).
    
    # Actually, the branching *is* the search.
    # The "greedy on full set" is just one specific branch (the one that picks min class for 2,3,5,7,11).
    # Wait, the greedy algorithm *picks* the min class.
    # So "greedy on full set" corresponds to one specific combination of choices for 2,3,5,7,11.
    # The branching explores *other* choices.
    
    # So we should definitely do the branching.
    
    # Let's optimize the inner loop with Numpy.
    
    def run_greedy_from_mask(mask, start_prime_idx):
        # mask is a boolean numpy array of size N+1
        # start_prime_idx is index in PRIMES to start from
        # Returns the final mask and size
        current_mask = mask.copy()
        current_size = np.sum(current_mask)
        
        for i in range(start_prime_idx, len(PRIMES)):
            p = PRIMES[i]
            if p > current_size:
                break
            
            # Count residues
            # Get active indices
            # Instead of getting indices, we can use modulo on the whole array?
            # No, we only care about active ones.
            
            # Efficient counting:
            # residues = current_indices % p
            # np.bincount(residues, minlength=p)
            
            active_indices = np.nonzero(current_mask)[0]
            if len(active_indices) == 0:
                break
            
            residues = active_indices % p
            counts = np.bincount(residues, minlength=p)
            
            # Check if any count is 0
            if np.min(counts) > 0:
                # All residues covered
                # Find min count residue
                min_res = np.argmin(counts)
                # Remove this residue
                # current_mask[active_indices[residues == min_res]] = False
                # Or:
                current_mask[active_indices[residues == min_res]] = False
                current_size = np.sum(current_mask)
            else:
                # Already missed a residue, do nothing
                pass
                
        return current_mask, current_size

    # Branching
    # Primes: 2, 3, 5, 7, 11
    # Indices in PRIMES:
    # 2 is index 0
    # 3 is index 1
    # 5 is index 2
    # 7 is index 3
    # 11 is index 4
    
    # We will iterate over product of range(p) for p in [2,3,5,7,11]
    
    # To save time, we can pre-compute the masks for the first few primes?
    # No, the combinations are independent.
    
    # Let's just loop.
    
    # We need to be careful with the time.
    # We will start with a small branching set and expand if fast.
    
    # Strategy:
    # 1. Try all 210 combinations for first 4 primes.
    # 2. If time allows, try all 2310 for first 5.
    
    # Actually, let's just try all 2310 and monitor time.
    # If it's too slow, we can reduce to 210.
    
    # Let's implement the 2310 loop with a time check.
    
    best_size = 0
    best_mask = None
    
    # Pre-compute base mask
    base_mask = np.ones(N + 1, dtype=bool)
    
    # Primes for branching
    branch_primes = [2, 3, 5, 7, 11]
    branch_indices = [PRIMES.index(p) for p in branch_primes]
    next_prime_idx = branch_indices[-1] + 1
    
    # To speed up, we can pre-compute the effect of removing a residue for each prime?
    # For p=2, removing 0 leaves odds, removing 1 leaves evens.
    # We can pre-compute these masks.
    # For p=3, removing 0, 1, 2.
    # But the combinations are sequential.
    # Mask_1 = Base & ~Residue_2_r1
    # Mask_2 = Mask_1 & ~Residue_3_r2
    # This is hard to pre-compute for all combinations without storing 2310 masks.
    # Storing 2310 masks of 5001 bools is 2310 * 5KB = 11MB. Acceptable.
    # But computing them is the cost.
    
    # Let's just do the filtering in the loop.
    
    # Optimization: 
    # Instead of np.nonzero every time, we can maintain a list of active indices?
    # Numpy filtering is fast.
    
    # Let's run.
    
    # We'll use itertools.product
    
    # To avoid repeating the same work, we can structure it recursively?
    # No, simple loop is fine.
    
    # One more thing: 
    # The "greedy" part after branching starts from p=13.
    # Is it possible that a different order of primes in the greedy phase helps?
    # Probably not.
    
    # Let's execute.
    
    # We need to handle the case where the set becomes empty or small.
    
    # Time guard
    iterations = 0
    max_iterations = 3000 # Safety cap
    
    for combo in itertools.product(*[range(p) for p in branch_primes]):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        # Apply branching
        mask = base_mask.copy()
        
        for p, r in zip(branch_primes, combo):
            # Remove x where x % p == r
            # mask = mask & ~(np.arange(N+1) % p == r)
            # This creates a temp array.
            # Better:
            mask[np.arange(N+1)[np.arange(N+1) % p == r]] = False
            # Even better: pre-compute residue masks for small primes?
            # Let's pre-compute for 2,3,5,7,11.
            
        # Apply greedy
        final_mask, size = run_greedy_from_mask(mask, next_prime_idx)
        
        if size > best_size:
            best_size = size
            best_mask = final_mask
        
        iterations += 1
        if iterations > max_iterations:
            break

    if best_mask is None:
        # Fallback
        return list(range(N + 1))
        
    return list(np.nonzero(best_mask)[0])

# Pre-computing residue masks for speed
# This will make the inner loop much faster.
