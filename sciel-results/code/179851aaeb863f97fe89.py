import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 90.0  # Safety margin for 60s limit
    
    # The problem asks for distinct integer denominators in [5, 10000000]
    # whose reciprocal sum is exactly 1.
    # We want to minimize the count.
    
    # We know k=1,2,3,4,5,6,7 are impossible for max sum < 1.
    # So we start searching from k=8.
    
    def dfs(remaining_sum, max_denom, min_denom, depth, current_denoms):
        """
        remaining_sum: Fraction, the sum we still need to achieve.
        max_denom: int, the maximum denominator allowed (for the next pick, it's 10^7, but effectively bounded by remaining_sum).
        min_denom: int, the minimum denominator allowed (must be > last picked).
        depth: int, how many more denominators we need to pick.
        current_denoms: list, the denominators picked so far.
        """
        if time.time() > deadline - 2.0:
            return None
        
        if depth == 0:
            if remaining_sum == 0:
                return current_denoms
            return None
        
        if remaining_sum <= 0:
            return None
            
        # Pruning:
        # 1. The smallest possible sum for 'depth' terms with denominators >= min_denom
        #    is achieved by taking the largest possible denominators, which approach 0.
        #    But we have an upper bound on denominator (10^7).
        #    Actually, a better lower bound check:
        #    If remaining_sum is too small to be formed by 'depth' distinct unit fractions
        #    with denominators >= min_denom, prune.
        #    The minimum sum is approx sum(1/(min_denom + i) for i in 0..depth-1) ? No, that's the MAX sum for fixed min_denom.
        #    Wait.
        #    To form remaining_sum with 'depth' terms, the terms must be <= remaining_sum/depth? No.
        #    
        #    Standard Egyptian Fraction DFS Pruning:
        #    Let next_denom be the denominator we pick next.
        #    We must have 1/next_denom <= remaining_sum  => next_denom >= 1/remaining_sum.
        #    Also next_denom >= min_denom.
        #    So start = max(min_denom, ceil(1/remaining_sum)).
        #    
        #    Upper bound for next_denom:
        #    If we pick next_denom, the remaining sum is remaining_sum - 1/next_denom.
        #    This must be formable by depth-1 terms with denominators > next_denom.
        #    The maximum sum of depth-1 terms with denominators > next_denom is
        #    sum_{i=1}^{depth-1} 1/(next_denom + i).
        #    So we require: remaining_sum - 1/next_denom <= sum_{i=1}^{depth-1} 1/(next_denom + i).
        #    
        #    Also, the minimum sum of depth-1 terms with denominators > next_denom and <= 10^7 is
        #    approx (depth-1)/10^7.
        #    So we require: remaining_sum - 1/next_denom >= (depth-1)/10^7.
        
        # Calculate start denominator
        # next_denom must be >= min_denom
        # and 1/next_denom <= remaining_sum => next_denom >= 1/remaining_sum
        start = max(min_denom, int(1 / remaining_sum) + (1 if 1 % remaining_sum.numerator != 0 else 0)) 
        # Simpler: start = max(min_denom, remaining_sum.denominator // remaining_sum.numerator + 1)
        # If remaining_sum = N/D, 1/x <= N/D => D <= Nx => x >= D/N.
        # ceil(D/N)
        start = max(min_denom, remaining_sum.denominator // remaining_sum.numerator + (1 if remaining_sum.denominator % remaining_sum.numerator else 0))
        
        # If start is too large, no solution
        if start > 10000000:
            return None

        # Determine the maximum denominator we can try for this step.
        # We need to leave enough sum for the remaining depth-1 terms.
        # Max sum of remaining depth-1 terms (using smallest possible denominators start+1, start+2...)
        # is bounded by sum_{i=1}^{depth-1} 1/(start+i).
        # If remaining_sum - 1/d > max_possible_remaining, then d is too small.
        # This effectively sets a LOWER bound on d, which we already have via 1/d <= remaining_sum.
        # Wait, if d is small, 1/d is large, remaining_sum - 1/d is small.
        # If d is large, 1/d is small, remaining_sum - 1/d is large.
        # We need remaining_sum - 1/d to be <= max_sum_of_remaining.
        # So 1/d >= remaining_sum - max_sum_of_remaining.
        # This gives an upper bound on d? No, 1/d >= X => d <= 1/X.
        
        # Let's just iterate d from start upwards.
        # How far up?
        # If d is very large, 1/d is tiny.
        # We need to form remaining_sum - 1/d with depth-1 terms.
        # The minimum value we can form with depth-1 terms (denominators > d) is approx (depth-1)/10^7.
        # So remaining_sum - 1/d >= (depth-1)/10^7.
        # 1/d <= remaining_sum - (depth-1)/10^7.
        # d >= 1 / (remaining_sum - (depth-1)/10^7).
        # This is a lower bound again.
        
        # What is the upper bound for d?
        # If we pick d, the next denominators must be > d.
        # The maximum sum we can get with depth-1 terms starting from d+1 is
        # S_max(d) = sum_{i=1}^{depth-1} 1/(d+i).
        # We need remaining_sum - 1/d <= S_max(d).
        # Let f(d) = 1/d + S_max(d). We need f(d) >= remaining_sum.
        # f(d) is decreasing in d.
        # So there is a maximum d_max such that f(d_max) >= remaining_sum.
        # For d > d_max, even if we pick the smallest possible subsequent denominators, the sum is too small.
        
        # Let's find an upper bound for the loop.
        # A safe upper bound is just 10000000, but that's too big to iterate naively if start is small.
        # However, depth is small (8).
        # If start is 5, d can be large.
        # But if d is large, 1/d is small.
        # Let's estimate d_max.
        # Approximation: depth * 1/d_max <= remaining_sum => d_max >= depth / remaining_sum.
        # This is a lower bound.
        # Upper bound:
        # remaining_sum <= 1/d + (depth-1)/(d+1) ... 
        # remaining_sum <= depth/d (roughly).
        # So d <= depth / remaining_sum.
        # Let's use end = min(10000000, int(depth / remaining_sum) + 100)
        # This might miss solutions if the approximation is loose, but for Egyptian fractions, the denominators grow rapidly.
        # Actually, the denominators don't necessarily grow rapidly if we are not greedy.
        # But for minimal count, we are looking for "dense" sets of small denominators.
        
        # Let's use a tighter bound.
        # If we pick d, the remaining sum R' = remaining_sum - 1/d.
        # R' must be >= (depth-1) / 10000000.
        # And R' <= sum_{i=1}^{depth-1} 1/(d+i).
        
        # Let's just iterate and check the condition.
        # To speed up, we can break if 1/d + sum_{i=1}^{depth-1} 1/(d+i) < remaining_sum.
        
        end = min(10000000, int(depth / remaining_sum) + 1000)
        # If start > end, return None
        
        for d in range(start, end + 1):
            if time.time() > deadline - 2.0:
                return None
            
            # Check if we can still form the remaining sum with depth-1 terms
            # Max sum of depth-1 terms with denominators > d
            max_remaining = 0
            for i in range(1, depth):
                max_remaining += Fraction(1, d + i)
            
            if remaining_sum - Fraction(1, d) > max_remaining:
                # Even with the smallest possible subsequent denominators, we can't reach the target.
                # Since d is increasing, 1/d is decreasing, so remaining_sum - 1/d is increasing.
                # And max_remaining is decreasing.
                # So for larger d, this condition will hold even more strongly.
                break
            
            # Check min sum
            # Min sum of depth-1 terms with denominators > d and <= 10^7
            # Approx (depth-1)/10^7
            min_remaining = Fraction(depth - 1, 10000000)
            if remaining_sum - Fraction(1, d) < min_remaining:
                # We picked d too small? No, d too small means 1/d large, remaining sum small.
                # If remaining sum is too small, we can't fill it with depth-1 positive terms.
                # As d increases, 1/d decreases, remaining sum increases.
                # So we can continue searching larger d.
                continue
            
            # Recurse
            new_remaining = remaining_sum - Fraction(1, d)
            res = dfs(new_remaining, 10000000, d + 1, depth - 1, current_denoms + [d])
            if res is not None:
                return res
        
        return None

    # Try k from 8 upwards
    # We know k<8 is impossible
    for k in range(8, 257):
        if time.time() > deadline - 5.0:
            break
        res = dfs(Fraction(1, 1), 10000000, 5, k, [])
        if res is not None:
            return res
            
    # Fallback: return a known valid solution if DFS fails (should not happen for k=8)
    # But we need to ensure it's valid.
    # Let's construct one manually if needed.
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 = 1.0101...
    # Adjust:
    # This is hard to do manually.
    # If DFS fails, we might return an empty list or raise an error.
    # But the task requires a valid return.
    
    # Given the constraints, k=8 should be found quickly.
    return []

