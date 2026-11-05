import time
import random
import numpy as np

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)
# We only need primes up to N, but the constraint is p <= len(A).
# Max len(A) is N+1. So all primes up to N are potentially relevant.

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_set = []
    
    # Precompute residues for all numbers 0..N for all primes?
    # That's 5000 * 669 ints. ~3.3M ints. ~26MB. Acceptable.
    # But we can compute on the fly or precompute a list of lists.
    # residues[x][i] = x % PRIMES[i]
    # This might be slow to build. Let's just use % operator, it's fast in Python/NumPy.
    
    def repair(points):
        # points is a set of ints
        # We maintain counts for each prime p
        # counts[p][r] = number of elements in points with x % p == r
        
        # Optimization: Only track primes p <= len(points)
        # But len(points) changes.
        
        # Let's use a dictionary or list of arrays.
        # Since p goes up to 5000, and we have ~669 primes.
        # We can have an array of dicts or arrays.
        
        # For speed, let's precompute the list of primes we care about.
        # Initially, we care about primes <= len(points).
        
        # We will iterate until no changes.
        # To speed up, we can check conflicts in a specific order.
        # Small primes are more likely to be fully covered.
        
        # Let's store points as a list for fast removal?
        # Removal from a list is O(N).
        # Using a set is O(1) for removal, but iterating is slow?
        # Let's use a list and a set for membership if needed, or just list.
        
        # The seed uses list comprehension for removal.
        
        # To make it faster, let's precompute for each prime p, the residues of all points.
        # This is expensive if done every iteration.
        
        # Alternative: 
        # 1. Compute counts for all primes p <= len(points).
        # 2. Find a prime p with all counts > 0.
        # 3. If found, remove all x in points where x % p == r_min.
        # 4. Repeat.
        
        # The expensive part is computing counts.
        # len(points) can be up to 5000.
        # Number of primes <= 5000 is 669.
        # 5000 * 669 = 3.3M operations per check.
        # If we do 100 checks, that's 330M ops. Might be too slow in Python.
        
        # We need to optimize the count update.
        # Instead of recounting from scratch, we can update counts when we remove elements.
        
        # Let's try this:
        # Initialize counts for all primes p <= len(points).
        # Loop:
        #   Find p with conflict.
        #   If none, return points.
        #   If conflict, determine r_min.
        #   Identify elements to remove: R = {x in points | x % p == r_min}
        #   Remove R from points.
        #   Update counts for all primes q <= len(points) (new length) by decrementing counts[q][x % q] for each x in R.
        #   Note: len(points) decreases, so we might need to stop tracking large primes?
        #   Actually, if p > len(points), the condition "p <= len(points)" fails, so we don't care if it's covered.
        #   But we still have counts for it. We just don't check it.
        
        # This incremental update is much faster.
        
        # Initialization:
        # points_set = set(points)
        # points_list = list(points)
        
        # counts = {}
        # for p in PRIMES:
        #    if p > len(points): break # Only initialize up to current len?
        #    # But len changes. 
        #    # Let's initialize for all primes up to N.
        #    counts[p] = [0] * p
        
        # for x in points:
        #    for p in PRIMES:
        #       if p > N: break # Not needed
        #       counts[p][x % p] += 1
        #       # Optimization: stop if p > len(points)? No, x%q is needed for update later.
        #       # But we only need to *check* p <= len(points).
        #       # However, we need counts for *all* primes to update correctly when removing?
        #       # Yes, because a removed x affects counts for all primes.
        
        # This initialization is O(|points| * #primes). 5000 * 669 = 3.3M. OK.
        
        # Let's implement this.
        
        p_set = set(points)
        p_list = list(points)
        current_len = len(p_list)
        
        # Initialize counts for all primes
        # To save memory/time, only store counts for primes <= N
        # We can use a dict: prime -> list of counts
        counts = {}
        for p in PRIMES:
            counts[p] = [0] * p
            
        for x in p_list:
            for p in PRIMES:
                counts[p][x % p] += 1
                
        # Main loop
        while True:
            if time.time() > deadline:
                return p_list # Return current state as best effort
            
            conflict_prime = None
            min_residue = None
            min_count = float('inf')
            
            # Check primes p <= current_len
            # We should check small primes first as they are more likely to be full?
            # Or just iterate.
            for p in PRIMES:
                if p > current_len:
                    break
                # Check if all residues are covered
                # If min(counts[p]) > 0, then all are covered.
                # We want to find *a* prime with conflict.
                # And among residues, the one with min count.
                # Actually, the strategy is:
                # 1. Find a prime p with conflict.
                # 2. For that p, remove the least populated residue class.
                
                # To be efficient, we can just pick the first prime with conflict?
                # Or the one that requires the fewest removals?
                # The seed picks the first prime in the loop (smallest p).
                # Let's stick to smallest p, but maybe we can check if removing a class for a larger p is "cheaper"?
                # Cost is min_count.
                # Let's find the prime p and residue r that minimizes the number of elements to remove?
                # i.e., min over all (p, r) such that p has conflict of counts[p][r].
                # This requires checking all p and all r.
                # 669 primes * avg 25 residues = ~16k checks. Very fast.
                
                min_in_p = min(counts[p])
                if min_in_p > 0:
                    # Conflict exists for p
                    # Find the residue with min count
                    # min_in_p is the min count.
                    # We want to remove min_in_p elements.
                    # If min_in_p < min_count, update best.
                    if min_in_p < min_count:
                        min_count = min_in_p
                        conflict_prime = p
                        # Find which residue has this min count
                        # There might be multiple. Pick first one.
                        for r in range(p):
                            if counts[p][r] == min_in_p:
                                min_residue = r
                                break
            
            if conflict_prime is None:
                # No conflicts
                return p_list
                
            # Remove all x in p_list such that x % conflict_prime == min_residue
            p_to_remove = conflict_prime
            r_to_remove = min_residue
            
            # Identify elements to remove
            to_remove = [x for x in p_list if x % p_to_remove == r_to_remove]
            
            if not to_remove:
                # Should not happen if counts were correct
                break
                
            # Remove them
            remove_set = set(to_remove)
            p_list = [x for x in p_list if x not in remove_set]
            current_len = len(p_list)
            
            # Update counts
            for x in to_remove:
                for p in PRIMES:
                    counts[p][x % p] -= 1
                    
            # Note: We don't need to shrink the counts dict, just ignore p > current_len in check.
            
    # Generate multiple starting sets
    # 1. Seed set
    seed_points = [1] + [p for p in PRIMES if p * p > N]
    res1 = repair(seed_points)
    if len(res1) > len(best_set):
        best_set = res1
        
    # 2. Random sets
    # Try different sizes
    # The size of the result from seed is likely the baseline.
    # Let's try sizes around that.
    # But we don't know it yet.
    # Let's try a few random starts of size 100, 200, 300, 500, 1000.
    
    random.seed(42)
    for size in [50, 100, 200, 300, 500, 1000, 2000, 5000]:
        if time.time() > deadline:
            break
        # Random sample
        # Using np.random.choice is fast
        # But we need distinct ints in [0, N]
        pts = np.random.choice(N + 1, size=size, replace=False)
        pts = [int(x) for x in pts]
        res = repair(pts)
        if len(res) > len(best_set):
            best_set = res
            
    # 3. Structured starts?
    # e.g. multiples of 2, 3, etc?
    # These will likely fail immediately for p=2 or 3.
    
    # Let's try a greedy construction from scratch?
    # Start with empty set, add elements one by one?
    # No, admissibility is a global constraint on the set.
    
    # Let's try more random restarts with varying sizes if time permits.
    for i in range(10):
        if time.time() > deadline:
            break
        size = random.randint(100, 2000)
        pts = np.random.choice(N + 1, size=size, replace=False)
        pts = [int(x) for x in pts]
        res = repair(pts)
        if len(res) > len(best_set):
            best_set = res
            
    return best_set
