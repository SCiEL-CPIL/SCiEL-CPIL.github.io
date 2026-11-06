"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd
from itertools import combinations

def construct():
    """
    Return 1..256 distinct integer denominators in [5,10000000] whose reciprocal sum is exactly 1;
    minimize their count.
    
    Strategy:
    1. Try to find a solution with k=8 terms (likely minimum).
    2. If not found within time budget, fall back to a known valid solution with more terms.
    
    We use a meet-in-the-middle approach for k=8:
    - Split 8 terms into two groups of 4.
    - Precompute all sums of 4 distinct denominators from a candidate set.
    - Look for complementary pairs that sum to 1.
    
    However, the candidate set is huge [5, 10^7]. We need to be smarter.
    
    Alternative: Use a known identity or search with pruning.
    
    Known fact: The minimum number of distinct unit fractions (denominators >= 5) summing to 1 is 8.
    Example: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/11 + 1/15 + 1/35 = 1? Let's check.
    Actually, a known solution with 8 terms is:
    1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + ... no, that's 9 terms maybe.
    
    Let's use a search that tries to build the solution greedily with backtracking,
    using exact rational arithmetic to avoid floating point errors.
    
    We'll define a recursive function that picks the next denominator, ensuring:
    - It's larger than the previous one (to keep them distinct and sorted).
    - It's >= 5.
    - It's <= 10,000,000.
    - The remaining sum can be achieved with the remaining number of terms.
    
    Bounds for the remaining sum:
    If we need 'r' more terms, each >= 'next_min', the max possible sum is r * (1/next_min).
    The min possible sum is very small, but we can bound it by assuming the denominators grow.
    A tighter upper bound: if we pick denominators d_i, d_i+1, ..., d_i+r-1, the sum is roughly r/d_i.
    A lower bound: the sum of the smallest possible r denominators starting from next_min.
    
    Actually, a standard bound for Egyptian fractions:
    If we need to represent S with r terms, all >= m, then:
    S <= r/m  =>  m <= r/S
    Also, S >= 1/m + 1/(m+1) + ... (this is a lower bound if we must use distinct integers >= m)
    But since we want to *find* a solution, we can use:
    For the next denominator d:
    - d must be >= max(prev_d + 1, 5)
    - d must be <= floor(r / S)  (because 1/d <= S/r => d >= r/S, wait. 1/d + ... = S. Since each term <= 1/d, sum <= r/d => S <= r/d => d <= r/S. So d <= floor(r/S) is an upper bound on d? No.
      If d is too large, 1/d is too small. If d is too small, 1/d is too big.
      We need S = 1/d + S'. S' = S - 1/d.
      We need S' to be representable with r-1 terms, each > d.
      Max possible S' with r-1 terms starting from d+1 is (r-1)/(d+1).
      So we need S - 1/d <= (r-1)/(d+1).
      Also, S - 1/d > 0 => 1/d < S => d > 1/S.
    
    So for a given state (S, r, min_d):
    d ranges from max(min_d, floor(1/S) + 1) to floor(r / S).
    And we must also have S - 1/d <= (r-1)/(d+1).
    
    Let's implement this DFS with memoization or pruning.
    Since k=8 is small, DFS should be fast if pruned well.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds safety margin
    
    target = Fraction(1, 1)
    max_d = 10_000_000
    
    def dfs(rem_sum, rem_terms, min_den, current_list):
        if time.time() > deadline:
            return None
        
        if rem_terms == 0:
            if rem_sum == 0:
                return current_list
            return None
        
        if rem_sum <= 0:
            return None
        
        # Lower bound on next denominator:
        # 1/d < rem_sum => d > 1/rem_sum
        # Also d >= min_den
        low_d = max(min_den, int(1 / rem_sum) + 1)
        
        # Upper bound on next denominator:
        # rem_sum = 1/d + rest, where rest <= (rem_terms-1)/(d+1)
        # So rem_sum - 1/d <= (rem_terms-1)/(d+1)
        # Also, rest >= 0 => 1/d <= rem_sum => d >= 1/rem_sum (already used)
        # And 1/d is the largest term, so rem_sum >= 1/d => d >= 1/rem_sum
        # Also, the sum of the remaining terms is at most (rem_terms-1)/(d+1)
        # So rem_sum - 1/d <= (rem_terms-1)/(d+1)
        # rem_sum <= 1/d + (rem_terms-1)/(d+1)
        # This gives an upper bound on d? No, it gives a constraint.
        # If d is too large, 1/d + (rem_terms-1)/(d+1) becomes small.
        # We need rem_sum <= 1/d + (rem_terms-1)/(d+1)
        # Let's find the max d such that this holds.
        # For large d, RHS ~ rem_terms/d. So d <= rem_terms / rem_sum roughly.
        high_d = int(rem_terms / rem_sum)
        if high_d > max_d:
            high_d = max_d
        
        # We also need d >= low_d
        if low_d > high_d:
            return None
        
        # Iterate d from low_d to high_d
        # To improve chances, we can try smaller d first or larger d first?
        # Smaller d means larger 1/d, leaving smaller rem_sum for fewer terms.
        # Larger d means smaller 1/d, leaving larger rem_sum.
        # Usually, starting with smaller d (larger fractions) is better for Egyptian fractions.
        
        for d in range(low_d, high_d + 1):
            next_sum = rem_sum - Fraction(1, d)
            if next_sum < 0:
                break  # d is too small, 1/d > rem_sum
            
            # Check if next_sum can be represented with rem_terms-1 terms, all > d
            # Max possible sum with rem_terms-1 terms starting from d+1:
            # This is a harmonic-like sum. A simple upper bound is (rem_terms-1)/(d+1).
            if next_sum > Fraction(rem_terms - 1, d + 1):
                continue  # d is too small, even with smallest possible next denominators, sum is too big?
                # Wait, if next_sum > max_possible, then we can't achieve it.
                # Actually, if d is small, 1/d is large, next_sum is small.
                # If d is large, 1/d is small, next_sum is large.
                # The condition next_sum > (rem_terms-1)/(d+1) means the remaining sum is too large
                # to be formed by rem_terms-1 fractions each <= 1/(d+1).
                # So we skip this d.
            
            # Also, next_sum must be at least the sum of the smallest possible rem_terms-1 fractions
            # starting from d+1. This is a tighter lower bound but harder to compute.
            # For now, we rely on the recursion to fail quickly.
            
            result = dfs(next_sum, rem_terms - 1, d + 1, current_list + [d])
            if result is not None:
                return result
        
        return None
    
    # Try k=8
    for k in range(8, 13):
        if time.time() > deadline:
            break
        result = dfs(target, k, 5, [])
        if result is not None:
            return result
    
    # Fallback: return a known valid solution if DFS fails
    # A known solution with more terms:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + ...
    # Let's just use a greedy algorithm to generate a valid solution as fallback.
    
    def greedy_fallback():
        rem = Fraction(1, 1)
        denoms = []
        last_d = 4
        while rem > 0:
            # Find smallest d > last_d such that 1/d <= rem
            d = max(last_d + 1, int(1 / rem) + 1)
            if d > max_d:
                # Try to adjust
                break
            denoms.append(d)
            rem -= Fraction(1, d)
            last_d = d
            if len(denoms) > 256:
                break
        # Verify
        s = sum(Fraction(1, d) for d in denoms)
        if s == 1 and all(5 <= d <= max_d for d in denoms) and len(denoms) <= 256:
            return denoms
        return None
    
    result = greedy_fallback()
    if result is not None:
        return result
    
    raise RuntimeError("no valid representation found")

# EVOLVE-BLOCK-END
