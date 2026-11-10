"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import lru_cache

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)
L = math.lcm(*DIVISORS)  # lcm of all divisors

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # 5s safety margin
    
    # For each divisor d, the set of multiples in [a, a+W] (inclusive) has size:
    # floor((a+W)/d) - floor((a-1)/d)
    # We need to pick one distinct multiple for each divisor.
    
    # Strategy: search over W (span) and offset a.
    # For a given W, the window is [a, a+W].
    # For divisor d, count_d(a, W) = floor((a+W)/d) - floor((a-1)/d)
    # We need count_d >= 1 for all d, and we need to assign distinct values.
    # This is a bipartite matching problem: divisors to values in the window.
    # But the window can be large. Instead, we can use a greedy/backtracking approach.
    
    # Better approach: iterate W from 1 upward. For each W, iterate a over a reduced range.
    # Due to periodicity L, the pattern of counts repeats every L in a.
    # So we only need to check a in [0, L-1] for each W... but L is huge.
    
    # Alternative: for a given W, the "tightness" is determined by how many multiples
    # each divisor has in the window. We want all divisors to have at least 1 multiple.
    # The minimum W such that for some a, every divisor has at least 1 multiple in [a, a+W]
    # is what we want. Then among those (a, W) pairs, we pick the one that allows a valid
    # distinct assignment (which is always possible if Hall's condition holds, which it
    # typically does when counts are >= 1 for small N).
    
    # Actually, since N=10 is small, we can do this:
    # For a candidate (a, W), build the set of available values for each divisor.
    # Then check if a perfect matching exists.
    
    # To speed up: iterate W from 1. For each W, find the "best" a.
    # The optimal a for a given W can be found by checking all residues mod L, but L is ~10^7.
    # Instead, we can be smarter: the critical constraint is that each divisor d must have
    # at least one multiple in [a, a+W]. This means a <= k*d <= a+W for some integer k.
    # Equivalently, a mod d must be in a certain set... 
    
    # Let me think differently. For a fixed W, define f(a) = min over d of (number of multiples of d in [a, a+W]).
    # We need f(a) >= 1 for all d, i.e., each d has at least one multiple.
    # The set of a where d has NO multiple in [a, a+W] is a union of intervals of length d - W (if W < d).
    # If W >= max(DIVISORS) = 39, then every divisor has at least one multiple in any window of length W.
    # So for W >= 39, the constraint is automatically satisfied.
    # For W < 39, some divisors might have 0 multiples.
    
    # So the minimum possible W is at most 39 (actually much less).
    # Let's just iterate W from 1 to, say, 200, and for each W, find if there exists an a
    # such that all divisors have at least 1 multiple in [a, a+W], and a perfect matching exists.
    
    # For a given W, the "bad" a values for divisor d are those where [a, a+W] contains no multiple of d.
    # The multiples of d are at ..., d, 2d, 3d, ...
    # [a, a+W] contains a multiple of d iff there exists k such that k*d is in [a, a+W].
    # This fails iff for all k, either k*d < a or k*d > a+W.
    # The gap between consecutive multiples is d. So if W < d, there are gaps of length d - W - 1 (in terms of integer positions where no multiple exists) ... 
    # Actually, the set of a such that [a, a+W] contains NO multiple of d is:
    # For each k, the interval (k*d - W, (k+1)*d) in terms of a. So a in (k*d - W, (k+1)*d - 1] (integer).
    # The length of each such interval is d - W - 1 + 1 = d - W if W < d, else 0.
    
    # This is getting complex. Let me just do a direct search for small W.
    
    best_result = None
    best_span = None
    
    # Upper bound: just use the divisors themselves, span = 39 - 6 = 33
    # But we can do much better.
    
    # Let's iterate W from 1 upward.
    # For each W, we need to find an a such that:
    # 1. For each d in DIVISORS, there exists k with a <= k*d <= a+W
    # 2. There exists a perfect matching assigning distinct values to divisors.
    
    # For efficiency, for a given W, we can find candidate a values by considering
    # the "tight" positions. The function g(a) = min_d (dist to nearest multiple of d in [a,a+W])
    # is periodic with period L. But L is large.
    
    # Alternative: for small W (say W <= 100), we can iterate a over a range of size L,
    # but that's too slow. Instead, note that the condition "d has a multiple in [a, a+W]"
    # depends only on a mod d. So the combined condition depends on a mod lcm(DIVISORS) = L.
    
    # For W < 39, let's just brute force a in [0, L-1] but that's ~10^7 which is too slow.
    
    # Better: for a given W, the set of valid a is an intersection of periodic conditions.
    # Let me just try a randomized/local search approach for small W.
    
    # Actually, let me think about what the optimal W might be.
    # We have 10 divisors. The smallest is 6, largest is 39.
    # In a window of length W, the number of multiples of d is roughly W/d.
    # For d=39, we need W >= 0 (at least 1 multiple if W >= 0 and a is chosen well).
    # The tightest constraint is for the largest divisor.
    
    # Let me just do a direct search: for W from 1 to 50, for a in a sampled set, check validity.
    
    # For a given (a, W), build the assignment problem.
    # For each divisor d_i, the available values are {k * d_i : a <= k * d_i <= a + W}
    # We need to pick one distinct value per divisor.
    
    def check_assignment(a, W):
        """Check if we can assign distinct multiples. Returns the assignment or None."""
        # Build list of available values for each divisor
        avail = []
        for d in DIVISORS:
            vals = []
            k_start = (a + d - 1) // d  # smallest k with k*d >= a
            k_end = (a + W) // d         # largest k with k*d <= a+W
            for k in range(k_start, k_end + 1):
                v = k * d
                if a <= v <= a + W:
                    vals.append(v)
            avail.append(vals)
        
        # Check Hall's condition or just do backtracking (N=10 is small)
        # Use backtracking with MRV heuristic
        # Sort divisors by number of available values
        indices = sorted(range(N), key=lambda i: len(avail[i]))
        
        assigned = [None] * N
        used = set()
        
        def backtrack(idx):
            if idx == N:
                return True
            i = indices[idx]
            for v in avail[i]:
                if v not in used:
                    used.add(v)
                    assigned[i] = v
                    if backtrack(idx + 1):
                        return True
                    used.remove(v)
                    assigned[i] = None
            return False
        
        if backtrack(0):
            return assigned
        return None
    
    # For each W, we need to find an a that works.
    # The condition depends on a mod L. Let's sample a smartly.
    # For a given W, the "critical" a values are near multiples of the divisors.
    
    # Let's iterate W and for each W, try a set of candidate a values.
    for W in range(1, 100):
        if time.time() > deadline:
            break
        
        # Candidate a values: for each divisor d, the multiples near 0..L
        # Actually, let's just try a in a range. The optimal a is periodic mod L.
        # For small W, let's try a in [0, 2*L] but that's too big.
        
        # Smarter: for a given W, the valid a form a periodic set with period L.
        # Let's find one valid a by checking a in [0, L-1] but sampling.
        
        # Actually, for W >= 1, let's just try a = 0, 1, 2, ..., up to some limit.
        # The pattern repeats every L, but L is ~10^7.
        
        # Alternative approach: for a given W, find all a in [0, L-1] where all divisors
        # have at least 1 multiple. This is an inclusion-exclusion / CRT type problem.
        
        # Let me just do a direct scan for small W. For W < 40, the window is small.
        # I'll scan a from 0 to some bound and check.
        
        # The key insight: for a given W, if we shift a by L, the pattern repeats.
        # So we only need to check a in [0, L-1]. But L is large.
        
        # Let me try a different approach: for each W, find the "density" of valid a.
        # For divisor d, the fraction of a where d has a multiple in [a, a+W] is min(1, (W+1)/d).
        # By union bound, the probability that ALL divisors have a multiple is roughly
        # prod_d min(1, (W+1)/d).
        
        # For W=10: prod = min(1,11/6)*min(1,11/10)*... = 1*1*1*1*1*1*1*1*1*11/39 ≈ 0.28
        # So about 28% of a values should work. We can find one by random sampling.
        
        # Let's try random sampling for each W.
        found = False
        for trial in range(200):
            if time.time() > deadline:
                break
            a = (trial * 7919) % L  # pseudo-random
            result = check_assignment(a, W)
            if result is not None:
                best_result = result
                best_span = W
                found = True
                break
        
        if found:
            # We found a valid assignment with span W. Since we're iterating W from 1,
            # this is the minimum span. But we should verify by checking all a for this W.
            # Actually, we just need ANY valid assignment with the minimum W.
            # Let's return this.
            break
    
    if best_result is None:
        # Fallback: use the divisors themselves
        best_result = DIVISORS[:]
        best_span = max(DIVISORS) - min(DIVISORS)
    
    # Ensure all values are Python int
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
