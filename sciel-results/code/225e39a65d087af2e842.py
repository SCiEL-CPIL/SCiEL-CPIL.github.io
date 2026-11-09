import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    n_min = 1000
    n_max = 1000000
    
    # We want to minimize the number of distinct primes.
    # Let's try k = 1, 2, 3, ...
    # For each subset of primes of size k, check if there is a valid n.
    
    # Precompute modular inverses or just use simple logic.
    # For a subset S, let M = lcm(S).
    # For each j in 1..24, n must satisfy:
    #   exists p in S such that (n + j) % p == 0
    #   => n % p == -j % p
    # So n must be in the union of residue classes for each j.
    # Actually, the condition is on n itself.
    # For a fixed j, the set of valid residues for n modulo M is:
    #   U_j = { r in [0, M) | exists p in S such that r % p == (-j) % p }
    # We need n such that n % M is in INTERSECTION_{j=1..24} U_j.
    
    # Let's implement a helper to check a subset.
    
    def check_subset(S):
        # S is a tuple of primes
        # Calculate M = lcm(S)
        M = 1
        for p in S:
            M = M * p // np.gcd(M, p)
            
        # If M is too large, we might run out of memory or time.
        # Max M for 9 primes is huge, but we start with small k.
        # For k=3, max M = 19*23*17 = 7429. Very small.
        # For k=4, max M = 23*19*17*13 = 96577. Still small.
        # For k=5, max M = 23*19*17*13*11 = 1,062,347. ~1M size.
        # For k=6, M can be ~10M.
        
        if M > 10_000_000:
            return None
            
        # valid_residues: boolean array of size M
        # Initialize with True for all residues? No, we intersect.
        # Start with all True for the first j, then intersect with others.
        
        # For each j, compute the set of valid residues modulo M.
        # A residue r is valid for j if there exists p in S such that r % p == (-j) % p.
        
        # We can compute a bitmask for each j, or just a boolean array.
        # Since M is small for small k, boolean array is fine.
        
        # Optimization: Precompute for each p in S and each residue r % p, 
        # but M is small enough to just loop.
        
        # Let's represent the set of valid residues for the current intersection as a set or array.
        # Using a boolean array `possible` of size M.
        
        # Start with all True
        possible = np.ones(M, dtype=bool)
        
        for j in range(1, 25):
            # For this j, a residue r is allowed if:
            # any( (r % p) == (-j % p) for p in S )
            
            # We can update `possible` by checking each p.
            # new_possible = OR_{p in S} (r % p == -j % p)
            
            # To avoid creating many temporary arrays, we can do:
            # temp = np.zeros(M, dtype=bool)
            # for p in S:
            #    temp |= (np.arange(M) % p == -j % p)
            # possible &= temp
            
            # But np.arange(M) is expensive if done repeatedly.
            # Let's precompute arange_M = np.arange(M)
            
            # Actually, we can just check if the intersection becomes empty early.
            
            temp = np.zeros(M, dtype=bool)
            for p in S:
                # Residue condition: r % p == (-j) % p
                # This is periodic with period p.
                # We can use slicing or modulo.
                # np.arange(M) % p creates an array of size M.
                # For M=1M, this is 1M ops. 24 * 5 * 1M = 120M ops. Acceptable.
                
                # Optimization:
                # The condition r % p == c is true for indices i where i % p == c.
                # These are c, c+p, c+2p...
                # We can set temp[c::p] = True
                c = (-j) % p
                temp[c::p] = True
                
            possible &= temp
            
            if not np.any(possible):
                return None
        
        # If we reach here, there are valid residues.
        # Find the smallest n in [n_min, n_max] such that n % M is in possible.
        
        # possible indices:
        valid_indices = np.where(possible)[0]
        
        if len(valid_indices) == 0:
            return None
            
        # For each valid index r, we need n = k*M + r in [n_min, n_max].
        # Smallest n for a given r:
        # n_min <= k*M + r <= n_max
        # k >= (n_min - r) / M
        # k <= (n_max - r) / M
        
        # We want the global minimum n.
        # Iterate through valid_indices? There might be many.
        # Better: Find the minimum n directly.
        
        # The candidates are of the form k*M + r.
        # The smallest n >= n_min is determined by the smallest k for each r, then taking min over r.
        
        # k_r = ceil((n_min - r) / M)
        # n_r = k_r * M + r
        # We want min(n_r) subject to n_r <= n_max.
        
        # Since M is constant, n_r is roughly linear in r (with jumps).
        # Actually, n_r = M * ceil((n_min - r) / M) + r.
        
        # Let's just check the first few valid residues?
        # Or simply:
        # The set of valid n is the union of arithmetic progressions.
        # We can find the minimum by checking the first valid residue >= n_min % M?
        # No, because the "step" is M.
        
        # Simple approach:
        # For each r in valid_indices:
        #   k = ceil((n_min - r) / M)
        #   n_candidate = k * M + r
        #   if n_candidate <= n_max:
        #       update best_n
        #   else:
        #       break (if valid_indices is sorted, larger r might still yield smaller n? No.)
        #       Actually, if k is fixed, larger r gives larger n.
        #       But k can decrease as r increases?
        #       ceil((n_min - r) / M) decreases as r increases.
        #       Example: n_min=100, M=10.
        #       r=1: ceil(99/10)=10, n=101
        #       r=9: ceil(91/10)=10, n=109
        #       r=10 (invalid, r<M)
        #       r=0: ceil(100/10)=10, n=100
        #       
        #       Let's just iterate through valid_indices. 
        #       The number of valid residues is at most M.
        #       For small k (subset size), M is small.
        
        best_n_local = None
        for r in valid_indices:
            # Calculate smallest k such that k*M + r >= n_min
            if r >= n_min:
                k = 0
            else:
                # k*M >= n_min - r
                # k >= (n_min - r) / M
                k = (n_min - r + M - 1) // M
                
            n_candidate = k * M + r
            
            if n_candidate <= n_max:
                if best_n_local is None or n_candidate < best_n_local:
                    best_n_local = n_candidate
            else:
                # If n_candidate > n_max, could a larger r yield a smaller n?
                # n_candidate = k*M + r.
                # If we increase r, k might stay same (n increases) or decrease (n decreases).
                # k decreases by 1 when r crosses a multiple of M boundary relative to n_min.
                # Specifically, k(r) = ceil((n_min - r)/M).
                # As r increases, n_min - r decreases, so k(r) is non-increasing.
                # n(r) = k(r)*M + r.
                # If k stays same, n increases.
                # If k drops by 1, n drops by M-1.
                # So n(r) is not monotonic.
                # We must check all r.
                # But if M is small, valid_indices is small.
                # If M is large (e.g., 1M), valid_indices could be large.
                # However, for k=3, M ~ 7k.
                # For k=4, M ~ 96k.
                # For k=5, M ~ 1M.
                # Iterating 1M items is fast in Python if simple, but might be slow.
                # Let's optimize:
                # The minimum n will be close to n_min.
                # We can just find the first valid r such that the resulting n is minimal.
                # Actually, we can just check r in the range [n_min % M, M) first?
                # No, r is in [0, M).
                
                # Let's just break if we've checked enough?
                # No, we need the exact minimum.
                
                # Optimization:
                # The function n(r) has a sawtooth shape.
                # The minimum will be at some r.
                # We can iterate r from 0 to M-1.
                # If M is 1M, 1M iterations is ~0.1s in Python.
                # 90s budget is plenty.
                pass
        
        return best_n_local

    # Iterate over subsets of primes
    # We want to minimize the size of the subset.
    # Try size 1, then 2, etc.
    
    for k in range(1, len(primes) + 1):
        if time.time() > deadline:
            break
            
        # Generate all combinations of size k
        for S in combinations(primes, k):
            if time.time() > deadline:
                break
                
            n_candidate = check_subset(S)
            if n_candidate is not None:
                # We found a valid n with k distinct primes.
                # Since we iterate k in increasing order, this is the global minimum
                # for the number of distinct primes.
                # Note: There might be multiple n for this k. We found one.
                # The task asks to "MINIMIZE the number of distinct extracted factors".
                # It does not ask to minimize n.
                # So returning any valid n with the minimum k is sufficient.
                return int(n_candidate)
        
        # If no subset of size k works, try k+1.
        
    # Fallback: If no subset found (should not happen as k=9 always works),
    # fall back to brute force or return a default.
    # Actually, k=9 (all primes) means every number n+i is divisible by some prime < 24.
    # This is always true for any n? No.
    # If n+i has a prime factor >= 24, it is not "extracted" into the small prime set?
    # Wait, the task says: "extract its complete factor composed of primes STRICTLY BELOW 24".
    # "MINIMIZE the number of distinct extracted factors".
    # This implies we look at the set of primes p < 24 that divide n+i.
    # If n+i is prime > 24, the extracted factor is 1 (empty set).
    # So for a subset S to "cover" n, it means:
    # For each j, the set of primes p < 24 dividing n+j is a subset of S.
    # This is different from "n+j is divisible by at least one p in S".
    # 
    # RE-READING TASK:
    # "For each n+i, extract its complete factor composed of primes STRICTLY BELOW 24."
    # "MINIMIZE the number of distinct extracted factors."
    #
    # Let E(n) be the set of all primes p < 24 such that p divides some n+i (i=1..24).
    # We want to minimize |E(n)|.
    #
    # My previous logic assumed we need to *cover* the divisibility with a small set S,
    # meaning all small prime factors of n+i must be in S.
    # This is equivalent to: For all j, and for all p < 24, if p | (n+j), then p in S.
    # This implies that for any p NOT in S, p must NOT divide any n+j.
    # So, if we choose S, we are forcing that no prime outside S divides any n+j.
    # This means n+j must be composed ONLY of primes in S (and primes >= 24).
    #
    # So the condition for a subset S to be valid for n is:
    # For all j in 1..24, and for all p in P_small \ S, (n+j) % p != 0.
    #
    # This is a MUCH stronger constraint.
    # It means n must avoid certain residue classes modulo p for each p not in S.
    #
    # Let's re-evaluate.
    # If we pick S = {2}, then for all j, n+j must not be divisible by 3, 5, 7, ..., 23.
    # This means n+j is of the form 2^a * k, where k has no small prime factors.
    #
    # This changes the check_subset logic.
    #
    # Revised check_subset(S):
    # Let P_bad = P_small \ S.
    # For each p in P_bad, n must NOT be in the residue classes -1, -2, ..., -24 modulo p.
    # i.e., for each j, n % p != -j % p.
    # So for each p in P_bad, the set of forbidden residues is F_p = { -1 % p, ..., -24 % p }.
    # n must avoid F_p for all p in P_bad.
    #
    # This is a "sieve" problem.
    # We want to find n in [1000, 10^6] such that for all p in P_bad, n % p not in F_p.
    
    def check_subset_v2(S):
        P_small = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        S_set = set(S)
        P_bad = [p for p in P_small if p not in S_set]
        
        if not P_bad:
            # If S is all primes, any n works.
            return n_min
            
        # We need to find n such that for all p in P_bad, n % p not in Forbidden[p]
        # Forbidden[p] = { -j % p for j in 1..24 }
        
        # We can use a sieve on the range [n_min, n_max]?
        # Range is 1M.
        # For each p in P_bad, mark multiples of p that are in the forbidden residues.
        # Actually, we just need to find ONE n.
        
        # Let's use a boolean array `valid` of size (n_max - n_min + 1).
        # Initialize to True.
        # For each p in P_bad:
        #   For each j in 1..24:
        #     Residue r = -j % p
        #     Mark all n in range where n % p == r as invalid.
        
        # This is O(|P_bad| * 24 * (Range / p)).
        # For p=2, Range/2 = 500k. 24 * 500k = 1.2M ops.
        # For p=3, Range/3 = 333k.
        # Total ops is roughly Sum_{p in P_bad} 24 * Range / p.
        # If S is small, P_bad is large.
        # If S is large, P_bad is small.
        # We iterate S by size k.
        # If k=1, P_bad has 8 primes.
        # If k=8, P_bad has 1 prime.
        
        # This approach is feasible.
        
        size = n_max - n_min + 1
        valid = np.ones(size, dtype=bool)
        
        # Optimization: If we use the "forbidden residues" logic,
        # we can just check candidates?
        # No, sieving is better.
        
        for p in P_bad:
            # For each j in 1..24, forbid n such that n % p == -j % p
            # This is equivalent to: n is NOT in the set of numbers coprime to p? No.
            # It forbids specific residues.
            
            # The forbidden residues are a subset of {0, ..., p-1}.
            # Let's compute the set of forbidden residues for this p.
            forbidden_residues = set()
            for j in range(1, 25):
                forbidden_residues.add(-j % p)
            
            # Now, for each residue r in forbidden_residues,
            # mark n in [n_min, n_max] such that n % p == r as invalid.
            
            # We can do this by iterating through the range.
            # But NumPy is faster.
            
            # Create array of indices 0..size-1
            # n = n_min + idx
            # We want to mark idx where (n_min + idx) % p == r
            
            # (n_min + idx) % p == r
            # idx % p == (r - n_min) % p
            
            target_residues_idx = set()
            for r in forbidden_residues:
                target_residues_idx.add((r - n_min) % p)
            
            # Now mark all idx where idx % p is in target_residues_idx
            # idx % p takes values 0..p-1.
            
            # We can create a mask for idx % p.
            # idx = np.arange(size)
            # This is 1M array.
            # mod_p = idx % p
            # mask = np.isin(mod_p, list(target_residues_idx))
            # valid &= ~mask
            
            # To avoid creating large temporary arrays for each p,
            # we can loop over the residues in target_residues_idx.
            
            # For each tr in target_residues_idx:
            #   valid[tr::p] = False  # This works if n_min is aligned?
            #   No, idx starts at 0. n_min + idx.
            #   Condition: (n_min + idx) % p == r
            #   idx % p == (r - n_min) % p = tr
            #   So idx = tr, tr+p, tr+2p...
            #   valid[tr::p] = False
            
            for tr in target_residues_idx:
                valid[tr::p] = False
            
            if not np.any(valid):
                return None
        
        if np.any(valid):
            # Find the first valid index
            idx = np.where(valid)[0][0]
            return n_min + idx
            
        return None

    # Iterate over subsets again with the corrected logic
    for k in range(1, len(primes) + 1):
        if time.time() > deadline:
            break
            
        for S in combinations(primes, k):
            if time.time() > deadline:
                break
                
            n_candidate = check_subset_v2(S)
            if n_candidate is not None:
                return int(n_candidate)
        
    # Fallback
    return 1000
