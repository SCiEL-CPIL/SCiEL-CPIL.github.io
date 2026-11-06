import time
import math
import random
from fractions import Fraction
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    n = 30
    
    # LCM of 1..30 for exact integer arithmetic
    # LCM(1..30) = 2329089562800
    LCM = 1
    for i in range(1, n + 1):
        LCM = LCM * i // math.gcd(LCM, i)
    
    # weights[i-1] = LCM // i, so sum c[i-1]/i = sum c[i-1]*weights[i-1] / LCM
    weights = [LCM // i for i in range(1, n + 1)]
    
    # Meet-in-the-middle: split 30 coefficients into two halves of 15
    half = n // 2  # 15
    
    # Generate all 3^15 = 14348907 sums for first half
    # Use iterative approach for memory efficiency
    # Actually 3^15 is about 14.3M, which might be too much for memory
    # Let's split into 3 groups of 10 instead: 3^10 = 59049 each
    
    third = n // 3  # 10
    # Groups: [0..9], [10..19], [20..29]
    
    def gen_sums(start, end):
        """Generate all possible signed sums for indices start..end-1"""
        sums = [0]
        for i in range(start, end):
            w = weights[i]
            new_sums = []
            for s in sums:
                new_sums.append(s)      # coefficient 0
                new_sums.append(s + w)   # coefficient +1
                new_sums.append(s - w)   # coefficient -1
            sums = new_sums
        return sums
    
    # Generate sums for each group
    sums1 = gen_sums(0, third)      # indices 0..9
    sums2 = gen_sums(third, 2*third) # indices 10..19
    sums3 = gen_sums(2*third, n)     # indices 20..29
    
    # Sort sums2 and sums3 for binary search
    sums2.sort()
    sums3.sort()
    
    # For each sum in group1, find best combination from groups 2 and 3
    # We want sum1 + sum2 + sum3 close to 0 but not 0
    
    best_abs = None
    best_coeffs = None
    best_num = None
    best_den = None
    
    # For efficiency, precompute sorted sums2
    # For each s1, we need s2 + s3 ≈ -s1
    # Use two-pointer or binary search approach
    
    # Better: sort sums2, and for each s1, do a two-pointer scan on sums2 and sums3
    # But that's O(3^10 * 3^10) = 3^20 which is too much
    
    # Alternative: for each s1, binary search in sums2 for values near -s1, 
    # and for each such s2, binary search in sums3 for -s1-s2
    
    # Actually, let's use a different approach:
    # Combine groups 2 and 3 first: for each s2, for each s3, get s2+s3
    # That's 3^20 = 3.4e9, too many
    
    # Let's use: sort sums2. For each s1, and for each s2 in sums2 (sorted),
    # binary search in sums3 for -s1-s2. That's 3^10 * 3^10 * log(3^10) ≈ 14.3M * 17 ≈ 243M ops
    # Still too slow.
    
    # Better approach: for each s1, we want min |s1 + s2 + s3| over s2 in sums2, s3 in sums3, nonzero
    # Precompute: for the pair (sums2, sums3), we can create a merged structure
    # 
    # Actually, let's just do: sort sums2. For each s1, iterate over sums2, 
    # and for each s2, binary search in sums3 for target = -s1-s2.
    # 3^10 * 3^10 = 3.5e9 iterations... too slow.
    
    # Let me reconsider. 3^10 = 59049. 
    # For each s1 (59049 values), for each s2 (59049 values), binary search in sums3 (59049 values).
    # Total: 59049 * 59049 * log2(59049) ≈ 59049 * 59049 * 16 ≈ 5.6e10. Way too slow.
    
    # New plan: use 2 groups of 15. 3^15 = 14348907. 
    # Memory: 14M integers ~ 112MB. Might be okay.
    # Time to generate: 14M * some constant. Should be fast.
    # Then sort: 14M * log(14M) ≈ 14M * 24 ≈ 340M ops. Might take a few seconds.
    # Then for each s1 in group1 (14M values), binary search in group2 for -s1.
    # 14M * log2(14M) ≈ 14M * 24 ≈ 340M ops. Total maybe 10-20 seconds.
    
    # Let's try this approach.
    
    # Clear previous group data
    del sums1, sums2, sums3
    
    half = 15
    
    # Generate sums for first half (indices 0..14)
    sumsA = gen_sums(0, half)
    
    # Generate sums for second half (indices 15..29)
    sumsB = gen_sums(half, n)
    
    # Sort sumsB for binary search
    sumsB.sort()
    
    # For each sA, find sB such that sA + sB is closest to 0 but not 0
    # Use bisect to find insertion point
    
    import bisect
    
    best_abs_val = None
    best_total = None
    best_sA = None
    best_sB = None
    
    for sA in sumsA:
        target = -sA
        # Find position in sumsB where target would be inserted
        pos = bisect.bisect_left(sumsB, target)
        
        # Check pos-1 and pos
        for p in [pos - 1, pos]:
            if 0 <= p < len(sumsB):
                total = sA + sumsB[p]
                if total != 0:
                    abs_total = abs(total)
                    if best_abs_val is None or abs_total < best_abs_val:
                        best_abs_val = abs_total
                        best_total = total
                        best_sA = sA
                        best_sB = sumsB[p]
    
    if best_abs_val is None:
        # Should not happen, but fallback
        return [0] * 28 + [1, -1]
    
    # Now we need to recover the actual coefficients
    # We need to find which coefficients in each half produced best_sA and best_sB
    
    # This is hard because gen_sums doesn't track which coefficients were used
    # We need to redo the search with coefficient tracking, or use a different approach
    
    # Alternative: since we know the target sums, we can do a DFS to find coefficients
    # that produce the desired sum. But there might be multiple solutions.
    
    # Actually, let's use a different strategy: instead of meeting in the middle on sums,
    # let's use a local search / hill climbing approach to find good coefficient vectors.
    
    # Let's try a randomized local search approach
    def compute_signed_sum(coeffs):
        """Compute the signed harmonic sum as a Fraction"""
        total = Fraction(0, 1)
        for i, c in enumerate(coeffs):
            if c != 0:
                total += Fraction(c, i + 1)
        return total
    
    # Start with the seed solution
    best_coeffs = [0] * 28 + [1, -1]
    best_val = abs(compute_signed_sum(best_coeffs))
    
    # Local search: try flipping single coefficients
    def local_search(coeffs, deadline):
        nonlocal best_coeffs, best_val
        current = list(coeffs)
        current_val = abs(compute_signed_sum(current))
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                for new_c in [-1, 0, 1]:
                    if new_c == current[i]:
                        continue
                    old_c = current[i]
                    current[i] = new_c
                    new_val = abs(compute_signed_sum(current))
                    if new_val < current_val:
                        current_val = new_val
                        if current_val < best_val:
                            best_val = current_val
                            best_coeffs = list(current)
                        improved = True
                    else:
                        current[i] = old_c
        
        return best_coeffs, best_val
    
    # Try a few random restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random initial solution
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        # Ensure nonzero sum
        val = compute_signed_sum(coeffs)
        if val == 0:
            # Flip a random coefficient
            idx = random.randint(0, n - 1)
            coeffs[idx] = random.choice([-1, 1])
        
        coeffs, best_val = local_search(coeffs, deadline)
        if best_val == 0:
            break
    
    # Also try the meet-in-the-middle result if we can recover coefficients
    # For now, use the local search result
    
    # Verify the result
    final_val = compute_signed_sum(best_coeffs)
    if final_val == 0:
        # Fallback
        return [0] * 28 + [1, -1]
    
    # Ensure all coefficients are in {-1, 0, 1}
    for i in range(n):
        if best_coeffs[i] not in (-1, 0, 1):
            best_coeffs[i] = 0
    
    return best_coeffs
# EVOLVE-BLOCK-END
