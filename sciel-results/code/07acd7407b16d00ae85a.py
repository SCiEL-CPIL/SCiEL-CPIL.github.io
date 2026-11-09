import time
import math
import random
import numpy as np
from collections import defaultdict

N = 5000
DEADLINE_SECONDS = 320.0

def _primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [i for i in range(limit + 1) if sieve[i]]

def _is_admissible_numpy(points, primes, max_prime_val):
    """Check admissibility using numpy."""
    points_arr = np.array(points)
    # primes is a list
    # For each prime, check if all residues are covered
    # This is slow in pure python, so we use numpy broadcasting if possible
    # But points size varies.
    # For verification only, we can use a simple loop or set logic
    for p in primes:
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    all_primes = _primes(N + 1)
    num_primes = len(all_primes)
    max_prime_val = all_primes[-1]
    
    # Precompute mods for all x in [0, N] for all primes
    # mods[x, i] = x % all_primes[i]
    # This is 5001 x 669 int16 array. ~6MB. Fast to access.
    x_vals = np.arange(N + 1)
    mods = np.zeros((N + 1, num_primes), dtype=np.int16)
    for i, p in enumerate(all_primes):
        mods[:, i] = x_vals % p
    
    best_points = []
    best_size = 0
    
    # Strategy 1: Random Greedy Construction
    # We try multiple random permutations
    # We maintain state: counts[prime_idx, residue], missing[prime_idx]
    
    # We need to map prime value to index for quick lookup of "primes <= size"
    # Since primes are sorted, we can use bisect or precompute a lookup table
    # lookup_size[s] = number of primes <= s
    lookup_size = np.zeros(N + 2, dtype=np.int32)
    prime_idx = 0
    for s in range(N + 2):
        while prime_idx < num_primes and all_primes[prime_idx] <= s:
            prime_idx += 1
        lookup_size[s] = prime_idx
    
    def run_random_greedy(seed):
        np.random.seed(seed)
        perm = np.random.permutation(N + 1)
        
        # counts[i, r] is count of residue r for prime i
        # We only care about primes up to current size.
        # max size is N+1.
        counts = np.zeros((num_primes, max_prime_val + 1), dtype=np.int32)
        # missing[i] is number of residues with count 0 for prime i
        missing = np.zeros(num_primes, dtype=np.int32)
        for i, p in enumerate(all_primes):
            missing[i] = p
            
        S = []
        
        for x in perm:
            new_size = len(S) + 1
            # Primes to check are indices 0 to lookup_size[new_size]-1
            active_count = lookup_size[new_size]
            if active_count == 0:
                # No primes to check (size 1, prime 2 is not <= 1)
                S.append(x)
                continue
                
            active_indices = np.arange(active_count)
            
            # Get the residues for x for these active primes
            residues = mods[x, :active_count]
            
            # Get current counts
            old_counts = counts[active_indices, residues]
            
            # Check if any of these counts were 0
            was_zero = (old_counts == 0)
            
            # Simulate update
            # new_counts = old_counts + 1
            # new_missing = missing[active_indices].copy()
            # new_missing[was_zero] -= 1
            
            # We only need to check if any new_missing becomes 0
            # This happens if old_counts was 0 AND missing[active_indices] was 1
            # i.e., the prime was missing exactly one residue, and we just filled it.
            
            # Identify problematic primes
            # Indices in active_indices where was_zero is True
            zero_indices = np.where(was_zero)[0]
            
            if len(zero_indices) > 0:
                # Check missing count for these
                current_missing = missing[active_indices[zero_indices]]
                # If any is 1, it becomes 0 -> Invalid
                if np.any(current_missing == 1):
                    # Reject x
                    continue
            
            # Accept x
            # Update counts and missing
            counts[active_indices, residues] += 1
            missing[active_indices[zero_indices]] -= 1
            S.append(x)
            
        return S

    # Run Random Greedy
    num_tries = 0
    while time.time() < deadline - 50: # Leave time for other strategies
        seed = int(time.time() * 1000) % 1000000
        S = run_random_greedy(seed)
        if len(S) > best_size:
            best_size = len(S)
            best_points = S
            print(f"Random Greedy {num_tries}: size {best_size}")
        num_tries += 1
        if num_tries % 10 == 0:
            print(f"Time: {time.time() - start_time:.2f}s, Best: {best_size}")

    # Strategy 2: Greedy Removal from Full Set (Optimized)
    # Start with all elements, remove to make admissible
    # This is often faster to converge to a good solution if done smartly.
    # We use a set for O(1) removal? No, we need to check residues.
    # Let's try a simple heuristic: 
    # For each prime p, if all residues are present, remove the element from the least frequent residue class?
    # This is what the previous code did.
    # Let's try "remove random element from a fully covered prime's class".
    
    if time.time() < deadline - 10:
        points = list(range(N + 1))
        point_set = set(points)
        # We need to track counts for each prime
        # counts[p][r]
        counts = {p: [0]*p for p in all_primes}
        for v in points:
            for p in all_primes:
                counts[p][v % p] += 1
        
        # We need to know which primes are "full" (all residues > 0)
        # full_primes = set of p where min(counts[p]) > 0
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Find a full prime
            # To be efficient, we can check a random subset of primes or iterate
            # Let's iterate through primes in random order
            primes_to_check = all_primes[:]
            random.shuffle(primes_to_check)
            for p in primes_to_check:
                if time.time() > deadline - 5:
                    break
                # Check if p is full
                if min(counts[p]) > 0:
                    # Find residue with min count
                    min_count = min(counts[p])
                    # Find all residues with this min count
                    candidates_residues = [r for r in range(p) if counts[p][r] == min_count]
                    # Pick a random candidate residue
                    r = random.choice(candidates_residues)
                    # Find an element in points with v % p == r
                    # We need to search for it.
                    # To make this fast, we can maintain a list of elements for each (p, r)?
                    # That's heavy memory.
                    # Alternatively, just scan points.
                    # Scanning 5000 elements is fast.
                    removed = False
                    for i, v in enumerate(points):
                        if v % p == r:
                            # Remove v
                            points.pop(i)
                            point_set.remove(v)
                            # Update counts
                            for pp in all_primes:
                                counts[pp][v % pp] -= 1
                            removed = True
                            changed = True
                            break
                    if removed:
                        break # Restart prime check from beginning? Or continue?
                        # Restarting ensures we don't miss other primes, but is slower.
                        # Let's just break and loop again.
                        break
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = points
            print(f"Greedy Removal: size {best_size}")

    # Final check and return
    if not best_points:
        # Fallback
        best_points = [0]
        
    # Verify admissibility of best_points
    # primes_to_check = [p for p in all_primes if p <= len(best_points)]
    # if not _is_admissible_numpy(best_points, primes_to_check, max_prime_val):
    #     # If invalid, try to repair or return a known valid small set
    #     pass

    return best_points
