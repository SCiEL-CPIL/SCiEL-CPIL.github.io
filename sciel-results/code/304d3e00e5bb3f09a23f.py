import time
from fractions import Fraction

def construct():
    """
    Return 1..256 distinct integer denominators in [5, 10000000] whose reciprocal sum is exactly 1.
    Minimize their count.
    """
    start_time = time.time()
    deadline = 45.0  # Safety margin before 90s kill

    def time_ok():
        return time.time() - start_time < deadline

    def find_solution(k):
        """
        Finds k distinct integers d_1 < ... < d_k >= 5 such that sum(1/d_i) = 1.
        Uses DFS with strong pruning.
        """
        
        def dfs(rem, min_d, depth, max_depth, path):
            if not time_ok():
                return None
            if rem == 0:
                return path
            if depth == max_depth:
                return None
            
            terms_left = max_depth - depth
            
            # Pruning 1: Upper bound on the sum of remaining terms
            # Max sum is 1/min_d + 1/(min_d+1) + ... + 1/(min_d+terms_left-1)
            # If rem > this sum, impossible.
            # We can approximate or compute exactly. For small terms_left, exact is fine.
            # Optimization: if rem > terms_left / min_d, then impossible? No, that's an upper bound on the *average* term, not the sum.
            # Actually, sum_{i=0}^{n-1} 1/(min_d+i) < n/min_d.
            # So if rem > terms_left / min_d, it's impossible? No, the sum is less than n/min_d, so if rem > n/min_d, then rem > max_sum.
            # Wait, 1/x is convex, so the sum is less than n * (1/min_d) is not tight.
            # But a simpler check: if rem > terms_left / min_d, return None?
            # Let's check: max_sum = sum(1/(min_d+i)). Since 1/(min_d+i) < 1/min_d, max_sum < terms_left/min_d.
            # So if rem >= terms_left/min_d, it might still be possible? No, if rem > max_sum, impossible.
            # Using the bound rem > terms_left/min_d is a sufficient condition for impossibility only if terms_left/min_d < max_sum? No.
            # max_sum < terms_left/min_d. So if rem > terms_left/min_d, then rem > max_sum.
            # So: if rem > Fraction(terms_left, min_d): return None
            
            # Let's use a tighter bound if possible, but this is fast.
            if rem > Fraction(terms_left, min_d):
                return None

            # Pruning 2: Lower bound
            # Min sum is terms_left / 10^7 (since d <= 10^7)
            if rem < Fraction(terms_left, 10**7):
                return None

            # Determine range for next denominator d
            # 1/d <= rem  => d >= ceil(1/rem)
            # Also d >= min_d
            d_min = max(min_d, (1 + rem.denominator - 1) // rem.numerator)
            
            # Upper bound for d:
            # We need to be able to form the remaining sum with terms_left-1 terms, each >= 1/(d+1).
            # So rem - 1/d <= (terms_left-1) / (d+1)
            # rem - (terms_left-1)/(d+1) <= 1/d
            # This is complex. A simpler loose bound:
            # 1/d must be <= rem, which is handled by d_min.
            # Also, the remaining sum rem - 1/d must be positive.
            # And the maximum sum of remaining terms is bounded by (terms_left-1)/(d+1).
            # So rem - 1/d <= (terms_left-1)/(d+1)
            # rem <= 1/d + (terms_left-1)/(d+1)
            # Let's solve for d.
            # For a loose upper bound, we can use d <= terms_left / rem.
            # Why? Because 1/d + ... + 1/(d+terms_left-1) < terms_left/d.
            # So if rem > terms_left/d, impossible.
            # Thus d > terms_left/rem is impossible.
            # So d_max = floor(terms_left / rem)
            
            d_max = (terms_left * rem.denominator) // rem.numerator
            
            if d_min > d_max:
                return None
            
            # Iterate
            # To find a solution quickly, we often want to try smaller denominators first?
            # Or larger?
            # Smaller denominators reduce the remainder faster, but might leave a remainder that's hard to fill with larger denominators.
            # Larger denominators leave a larger remainder, which is easier to fill with smaller (but still > d) denominators?
            # Actually, standard strategy: iterate d from d_min to d_max.
            
            for d in range(d_min, d_max + 1):
                if not time_ok():
                    return None
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    break
                res = dfs(new_rem, d + 1, depth + 1, max_depth, path + [d])
                if res is not None:
                    return res
            return None

        return dfs(Fraction(1, 1), 5, 0, k, [])

    # Theoretical minimum is 8.
    # Let's try to find a solution with k=8.
    # If it fails, try k=9, etc.
    
    # However, k=8 search space might be large.
    # Let's check if a known solution exists for k=8.
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 = 0.78 + ...
    # Sum 1/5..1/12 = H_12 - H_4 = (1+1/2+...+1/12) - (1+1/2+1/3+1/4)
    # H_12 approx 3.103, H_4 approx 2.083. Diff approx 1.02.
    # So 1/5..1/12 sums to > 1.
    # We need to replace some terms with larger ones to reduce the sum to exactly 1.
    # Or use a mix.
    
    # Let's run the search.
    # To speed up, we can try specific known structures or just let DFS do it.
    # DFS with pruning should be fast for k=8 if a solution exists.
    
    # Try k=8
    res = find_solution(8)
    if res is not None:
        return res
    
    # Try k=9
    res = find_solution(9)
    if res is not None:
        return res
        
    # Try k=10
    res = find_solution(10)
    if res is not None:
        return res

    # Fallback: Return a valid solution with more terms if search fails.
    # We can construct one using the identity:
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + ...
    # Let's just compute a greedy-like expansion or use a known one.
    # Actually, if the search fails, it might be due to time.
    # Let's try to find a solution with k=11, 12 quickly.
    for k in range(11, 16):
        if not time_ok():
            break
        res = find_solution(k)
        if res is not None:
            return res

    # If all else fails, we must return something valid.
    # We can use the fact that we can always add terms.
    # Start with 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12
    # Sum S. If S > 1, we need to reduce.
    # If S < 1, we need to increase (add smaller terms? No, we used smallest).
    # Since 1/5..1/12 > 1, we can replace the largest term 1/12 with something larger?
    # No, that decreases the sum.
    # We want sum = 1.
    # Let's try a different approach for fallback:
    # Use the identity 1/n = 1/(n+1) + 1/(n(n+1)).
    # Start with 1/5.
    # 1/5 = 1/6 + 1/30.
    # 1/6 = 1/7 + 1/42.
    # 1/30 = 1/31 + 1/(30*31).
    # This generates many terms.
    
    # A simpler fallback:
    # Just return a hardcoded list that sums to 1.
    # We can generate one offline.
    # For now, let's try to find one with k=15 quickly.
    res = find_solution(15)
    if res is not None:
        return res

    # Last resort: raise error? No, must return valid.
    # Let's construct a valid one using the splitting identity until we have enough terms or sum is 1.
    # Actually, the search for k=8..10 should succeed.
    # If not, it's a bug.
    
    # Let's try a greedy algorithm to find a valid set (not necessarily minimal).
    # This is not minimal, but valid.
    # However, the task asks to minimize count.
    # If we can't find minimal, we return a valid one with higher count.
    # But the score is based on the count (lower is better).
    # So we want the smallest k.
    
    # If find_solution fails for k=8..15, we are in trouble.
    # Let's assume k=8 or 9 works.
    
    # As a final safety, return a known valid solution if we can hardcode one.
    # I will not hardcode a long list.
    # I will trust the search.
    
    raise RuntimeError("No solution found within time limit")
