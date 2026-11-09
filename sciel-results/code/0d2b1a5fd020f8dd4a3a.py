"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

def _score(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L is covered by at least 2 congruences."""
    n = len(residues)
    coverage = np.zeros(max_check + 1, dtype=np.int8)
    for i, (r, p) in enumerate(zip(residues, primes)):
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_check:
            xs = np.arange(start, max_check + 1, p)
            coverage[xs] += 1
    
    L = 0
    for x in range(1, max_check + 1):
        if coverage[x] < 2:
            L = x - 1
            break
    else:
        L = max_check
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit (or 150s if that's the hard kill, but 120s is the task contract)
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Initial solution: random or simple heuristic
    # We will use a greedy approach to build a good initial state or improve it.
    
    # Strategy:
    # 1. Start with random residues.
    # 2. Compute coverage.
    # 3. Find the first gap (x where coverage[x] < 2).
    # 4. If no gap up to some limit, we are done or can try to extend.
    # 5. To fix the gap, we need to add coverage to x.
    #    We can change the residue of some prime p to be x % p.
    #    Which prime to change?
    #    Ideally, we want to change a prime that is "less critical" or where changing the residue
    #    doesn't break coverage elsewhere.
    #    A heuristic: choose the prime p that currently covers the fewest numbers in [1, current_L] 
    #    or simply the largest prime (largest primes have sparser coverage, so changing them 
    #    might have less global impact? Or maybe smaller primes are better because they 
    #    cover more numbers, so we should be careful not to break them).
    #    
    #    Actually, a better heuristic:
    #    We want to maximize L. 
    #    Let's try to fix the first gap x0.
    #    For each prime p, if we set r_p = x0 % p, does it help?
    #    It adds 1 to coverage[x0].
    #    But it removes 1 from coverage[old_x] where old_x was the previous hit for this prime in [1, x0] or nearby?
    #    No, it removes the coverage of the *previous* residue's hits.
    #    
    #    Let's use a local search that is smarter:
    #    1. Calculate coverage.
    #    2. Find first x where coverage[x] < 2.
    #    3. If none, return.
    #    4. For each prime p (sorted by size, maybe?), calculate the cost of changing r_p to x % p.
    #       Cost = (Loss of coverage elsewhere) - (Gain at x).
    #       Gain at x is 1 if coverage[x] becomes 2 (or more).
    #       Loss is the number of y in [1, L] where coverage[y] drops from >=2 to <2.
    #       This is expensive to calculate exactly for each p.
    #       
    #    Alternative:
    #    Just try changing the residue of a random prime to cover the gap.
    #    If it improves L, keep it.
    #    This is basically what the seed does, but we can make it faster and more directed.
    
    # Let's implement a directed hill-climbing:
    # 1. Start with random residues.
    # 2. Loop:
    #    a. Compute coverage and current L.
    #    b. If time is up, break.
    #    c. If L is very large (e.g., > 5000), try to push further.
    #    d. Find the first gap x0 in [1, L+100] where coverage[x0] < 2.
    #       Actually, we want to extend L. So look for the first x > L where coverage[x] < 2?
    #       No, L is defined as the largest such that 1..L are covered. So L+1 is the first gap.
    #       So we always look at x0 = L + 1.
    #    e. We need to increase coverage[L+1].
    #    f. Try changing the residue of a prime p to (L+1) % p.
    #       Which p?
    #       We can try all p and pick the one that results in the best new L.
    #       To speed this up, we can approximate the new L.
    #       Changing r_p affects coverage at positions:
    #       Old hits: r_p + k*p
    #       New hits: (L+1)%p + k*p
    #       The change in coverage is +1 at new hits, -1 at old hits.
    #       We can simulate this change on the coverage array, compute new L, and if it's better, keep it.
    #       If not, revert.
    
    residues = [random.randint(0, p - 1) for p in primes]
    
    # Precompute indices for speed?
    # primes is a list of 168 ints.
    
    def _update_coverage(coverage, idx, old_r, new_r, p, max_check):
        """Update coverage array in-place for a residue change. Returns list of changes to revert if needed."""
        # Remove old
        if old_r == 0:
            start_old = p
        else:
            start_old = old_r
        if start_old <= max_check:
            xs_old = np.arange(start_old, max_check + 1, p)
            coverage[xs_old] -= 1
        
        # Add new
        if new_r == 0:
            start_new = p
        else:
            start_new = new_r
        if start_new <= max_check:
            xs_new = np.arange(start_new, max_check + 1, p)
            coverage[xs_new] += 1

    def _compute_L_from_coverage(coverage, max_check):
        # Find first index where coverage < 2
        # We can use np.argmax on (coverage < 2) but we need to handle the case where all are >= 2
        # Actually, we just need the first x where coverage[x] < 2.
        # Since we are looking for the first gap, we can iterate? 
        # Numpy is faster:
        # mask = coverage < 2
        # idx = np.argmax(mask)
        # if mask[idx]: return idx (but careful with 1-based vs 0-based)
        # coverage[0] is unused.
        
        # Let's just iterate from 1. It's fast enough if L is small.
        # But if L is large, it's slow.
        # However, we only call this after a change.
        # Let's use a binary search? No, it's not monotonic.
        # Let's use np.where
        # But we only care about the FIRST one.
        
        # Optimization: Keep a variable `current_L`. 
        # When we change coverage, L can only increase or decrease slightly?
        # Not necessarily.
        
        # Let's stick to a fast Python loop for L calculation, but break early.
        L = 0
        for x in range(1, max_check + 1):
            if coverage[x] < 2:
                return x - 1
        return max_check

    max_check = 100000
    coverage = np.zeros(max_check + 1, dtype=np.int8)
    for i, (r, p) in enumerate(zip(residues, primes)):
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_check:
            xs = np.arange(start, max_check + 1, p)
            coverage[xs] += 1
            
    current_L = _compute_L_from_coverage(coverage, max_check)
    best_L = current_L
    best_residues = list(residues)
    
    # To speed up L calculation, we can maintain a sorted set of "gaps" (indices where coverage < 2)?
    # That's complex.
    
    # Let's try the directed hill climbing.
    # At each step, we look at x0 = current_L + 1.
    # We try to change one residue to cover x0.
    # We try each prime p in order of increasing p? Or random?
    # Let's try random primes to avoid bias.
    
    # We need to be careful with time.
    # Each step involves:
    # 1. Trying to change a residue (O(max_check/p) ops for coverage update).
    # 2. Calculating new L (O(L) ops in worst case, but usually breaks early).
    # 3. Reverting if not improved.
    
    # If p is small, coverage update is expensive.
    # If p is large, coverage update is cheap.
    # So we should prioritize trying large primes first?
    # Or maybe small primes are more important for coverage?
    
    # Let's try:
    # For each step:
    #   x0 = current_L + 1
    #   Shuffle prime indices
    #   For each idx in shuffled:
    #       p = primes[idx]
    #       new_r = x0 % p
    #       if new_r == residues[idx]: continue
    #       Update coverage
    #       new_L = _compute_L_from_coverage(coverage, max_check)
    #       if new_L > current_L:
    #           current_L = new_L
    #           best_L = new_L
    #           best_residues = list(residues)
    #           break # Restart search for next gap
    #       else:
    #           Revert coverage
    #           residues[idx] = old_r # (already reverted in coverage, need to revert residue value)
    #   If no prime improved, we are stuck.
    #   Try a random restart or perturb?
    
    # This might be slow if we try many primes.
    # Let's limit the number of primes tried per step.
    
    # Also, _compute_L_from_coverage is the bottleneck if L is large.
    # We can optimize it by only checking up to current_L + some buffer?
    # No, because changing a residue might break coverage at a small x, reducing L.
    # But if we only accept changes that increase L, then L only increases.
    # So we can assume L is non-decreasing?
    # Yes, if we only accept improvements.
    # So we can start checking from 1, but we know 1..current_L are covered.
    # We only need to check from current_L+1 upwards?
    # Wait, if we change a residue, we might break coverage at x < current_L.
    # So we MUST check from 1.
    # However, we can optimize the check:
    # We know coverage[1..current_L] >= 2 before the change.
    # The change only affects O(max_check/p) points.
    # We can check if any of those points are < 2?
    # This is complex.
    
    # Let's just use the full check but with a fast numpy implementation?
    # np.argmax(coverage < 2) is fast.
    
    def _fast_L(coverage):
        # coverage[0] is 0.
        # We want first x >= 1 where coverage[x] < 2.
        # mask = coverage < 2
        # idx = np.argmax(mask)
        # if mask[idx]:
        #    return idx if idx > 0 else 0 # If idx=0, then x=1 is gap? No, coverage[0] is unused.
        #    Actually, if coverage[0] < 2, argmax returns 0. But we don't care about 0.
        #    We should set coverage[0] = 2 to ignore it.
        
        # Let's set coverage[0] = 2 permanently.
        
        mask = coverage < 2
        idx = np.argmax(mask)
        if mask[idx]:
            return int(idx) # If idx=0, L=0. If idx=1, L=0. If idx=2, L=1.
            # Wait. If coverage[1] < 2, idx=1, L=0. Correct.
            # If coverage[1]>=2, coverage[2]<2, idx=2, L=1. Correct.
        else:
            return len(coverage) - 1
            
    coverage[0] = 2 # Ignore index 0
    
    current_L = _fast_L(coverage)
    best_L = current_L
    best_residues = list(residues)
    
    # Optimization: Precompute prime indices sorted by size?
    # Large primes are cheaper to update.
    
    while time.time() < deadline - 2:
        x0 = current_L + 1
        if x0 > max_check:
            break
            
        # Try to improve
        improved = False
        # Try a subset of primes, maybe random order
        indices = list(range(n_primes))
        random.shuffle(indices)
        
        # To save time, maybe only try first 20?
        for idx in indices[:20]:
            p = primes[idx]
            new_r = x0 % p
            old_r = residues[idx]
            
            if new_r == old_r:
                continue
            
            # Update coverage
            if old_r == 0:
                start_old = p
            else:
                start_old = old_r
            if start_old <= max_check:
                xs_old = np.arange(start_old, max_check + 1, p)
                coverage[xs_old] -= 1
                
            if new_r == 0:
                start_new = p
            else:
                start_new = new_r
            if start_new <= max_check:
                xs_new = np.arange(start_new, max_check + 1, p)
                coverage[xs_new] += 1
                
            new_L = _fast_L(coverage)
            
            if new_L > current_L:
                # Accept
                current_L = new_L
                best_L = new_L
                best_residues = list(residues)
                residues[idx] = new_r
                improved = True
                break
            else:
                # Revert
                if old_r == 0:
                    start_old = p
                else:
                    start_old = old_r
                if start_old <= max_check:
                    xs_old = np.arange(start_old, max_check + 1, p)
                    coverage[xs_old] += 1
                    
                if new_r == 0:
                    start_new = p
                else:
                    start_new = new_r
                if start_new <= max_check:
                    xs_new = np.arange(start_new, max_check + 1, p)
                    coverage[xs_new] -= 1
                
                # residues[idx] remains old_r
            
        if not improved:
            # Stuck. Try a random perturbation?
            # Or just break?
            # Let's try a few random restarts if we have time.
            # Or just break and return best.
            # Since we are doing hill climbing, getting stuck is common.
            # Let's try to "jump" by changing a random residue and seeing if it helps later?
            # That's risky.
            
            # Let's try to change a random residue to a random value and see if L increases.
            # This is basically simulated annealing without temperature.
            
            # If we are stuck, maybe we should restart from a different random state?
            # But we have limited time.
            
            # Let's just break for now.
            break
            
    return best_residues
# EVOLVE-BLOCK-END
