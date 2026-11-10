import time
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin for 60s limit

    # The seed program returns a list of 20 denominators.
    # We aim to find a solution with fewer terms (k < 20).
    # Theoretical lower bound for k is 7 (since sum_{i=5}^11 1/i < 1 and sum_{i=5}^12 1/i > 1).
    # Actually, sum_{i=5}^{11} 1/i = 1/5+1/6+1/7+1/8+1/9+1/10+1/11 ≈ 0.2+0.166+0.142+0.125+0.111+0.1+0.09 ≈ 0.934.
    # Sum_{i=5}^{12} 1/i ≈ 0.934 + 1/12 ≈ 1.017.
    # So k must be at least 7? No, the terms don't have to be consecutive.
    # But the max sum for k terms with min denominator 5 is sum_{i=5}^{5+k-1} 1/i.
    # For k=7, max sum is sum_{i=5}^{11} 1/i ≈ 0.934 < 1. So k must be at least 8?
    # Wait, the denominators are distinct integers >= 5.
    # To maximize the sum for a fixed k, we pick the smallest k denominators: 5, 6, ..., 5+k-1.
    # Let's check k=8: sum_{i=5}^{12} 1/i ≈ 1.017 > 1. So k=8 is possible in principle.
    # k=7: sum_{i=5}^{11} 1/i ≈ 0.934 < 1. So k=7 is impossible.
    # Thus the minimum k is at least 8.

    # We will try to find a solution for k = 8, 9, 10, ... up to 19.
    # If we find one, we return it.
    # If we run out of time, we fall back to the seed solution (k=20).

    def dfs(k, idx, remaining, min_d, path):
        if time.time() > deadline:
            return None

        if idx == k:
            if remaining == 0:
                return path[:]
            return None

        # Pruning:
        # 1. If remaining == 0, we are done (but idx < k means we have extra terms, which is invalid for exact k)
        #    Actually, if remaining == 0 and idx < k, we cannot add more positive terms. So return None.
        if remaining == 0:
            return None

        # 2. Max possible sum with remaining k-idx terms starting from min_d
        #    To maximize sum, pick smallest denominators: min_d, min_d+1, ...
        #    If this max sum < remaining, we can't reach remaining.
        max_sum = 0
        for i in range(k - idx):
            max_sum += Fraction(1, min_d + i)
            if max_sum >= remaining:
                break
        if max_sum < remaining:
            return None

        # 3. Min possible sum?
        #    We need to pick a denominator d >= min_d such that 1/d <= remaining.
        #    Also, we need to be able to form remaining - 1/d with k-idx-1 terms from d+1 onwards.
        #    Lower bound for d: 1/d <= remaining => d >= 1/remaining.
        #    Also d >= min_d.
        #    Also, we need enough "room" in the remaining sum.
        #    The max sum of the remaining k-idx-1 terms starting from d+1 must be >= remaining - 1/d.
        
        d_min = max(min_d, int(1 / remaining) + 1)
        if d_min < 5:
            d_min = 5

        # Upper bound for d?
        # If we pick a very large d, 1/d is small.
        # We need remaining - 1/d to be formable by k-idx-1 terms.
        # The max sum of k-idx-1 terms starting from d+1 is sum_{i=0}^{k-idx-2} 1/(d+1+i).
        # This sum must be >= remaining - 1/d.
        # As d increases, 1/d decreases, so remaining - 1/d increases.
        # But the max sum of remaining terms decreases as d increases.
        # So there is a finite upper bound for d.
        # However, calculating it precisely is complex.
        # A simpler heuristic: d cannot be so large that 1/d is negligible.
        # Let's just iterate d from d_min upwards, with a cap to avoid infinite loops.
        # A safe cap: if d > 10000000, break (task limit).
        # But we can prune earlier.
        # If 1/d_min > remaining, we already handled by d_min >= 1/remaining.
        
        # Let's iterate d
        # To speed up, we can estimate the maximum d we need to try.
        # If we pick d, the remaining sum is rem' = remaining - 1/d.
        # We need to form rem' with k-idx-1 terms from d+1.
        # The max sum of these terms is approx (k-idx-1)/(d+1).
        # So we need (k-idx-1)/(d+1) >= remaining - 1/d.
        # This gives a bound on d.
        
        # For now, let's just loop d from d_min to a reasonable limit.
        # Since k is small (8-19), the search space is constrained.
        
        # Optimization: If k-idx == 1, we must have remaining = 1/d.
        if k - idx == 1:
            if remaining.numerator == 1:
                d_last = remaining.denominator
                if d_last >= min_d and d_last <= 10000000:
                    return path + [d_last]
            return None

        # General case
        # We iterate d from d_min upwards.
        # We stop if 1/d becomes too small to help.
        # Specifically, if 1/d < remaining / (k-idx), then even if we picked the largest possible sum for the rest,
        # we might not reach remaining? No, that's not right.
        
        # Let's just loop.
        # To avoid too many iterations, we can bound d.
        # If 1/d < remaining / (k - idx), then d > (k - idx) / remaining.
        # This is the average denominator needed.
        # If we pick d > (k-idx)/remaining, then 1/d < remaining/(k-idx).
        # This doesn't immediately prune, but it suggests d shouldn't be too large.
        
        # Let's set an upper bound for d in this iteration.
        # If we pick d, the remaining sum is rem' = remaining - 1/d.
        # We need to form rem' with k-idx-1 terms, each >= 1/(d+1) (max term).
        # So rem' <= (k-idx-1) / (d+1).
        # remaining - 1/d <= (k-idx-1) / (d+1)
        # This inequality can be solved for d, but it's messy.
        # Let's just iterate d up to a limit, say 100000, or until time runs out.
        
        # Actually, a better bound:
        # The smallest possible sum for k-idx-1 terms starting from d+1 is when denominators are huge, approaching 0.
        # So we just need rem' > 0, which implies 1/d < remaining, which is guaranteed by d >= 1/remaining.
        
        # The real constraint is the MAX sum of the remaining terms.
        # max_sum_rem = sum_{i=0}^{k-idx-2} 1/(d+1+i)
        # We need max_sum_rem >= remaining - 1/d.
        
        # Let's iterate d.
        # We can break if d is too large such that even the max sum of remaining terms is < remaining - 1/d.
        # As d increases, remaining - 1/d increases (approaches remaining), while max_sum_rem decreases.
        # So there is a crossover point.
        
        # Let's just loop d from d_min to a limit.
        # Limit: if d > 10000000, break.
        # But we can prune: if d > remaining * (k-idx) * 10, it's unlikely to work.
        # Let's use a dynamic limit.
        
        d = d_min
        while d <= 10000000:
            if time.time() > deadline:
                return None
            
            # Check if 1/d <= remaining
            if Fraction(1, d) > remaining:
                d += 1
                continue
            
            rem_new = remaining - Fraction(1, d)
            if rem_new == 0:
                # We used one term, but we have k-idx-1 terms left.
                # If k-idx-1 == 0, this is a solution.
                # But we handled k-idx==1 above.
                # So if k-idx > 1 and rem_new == 0, it's invalid because we must use exactly k-idx terms.
                # Wait, the problem asks for exactly k terms?
                # "Return 1..256 distinct integer denominators ... minimize their count."
                # It doesn't say "exactly k". It says minimize count.
                # So if we find a solution with m < k terms, that's better.
                # But my DFS is structured to find exactly k terms.
                # If I find a solution with fewer terms, I should return it.
                # However, the DFS is called for a specific k.
                # If I find a solution with m < k, it would have been found in a previous iteration of the outer loop (for k=m).
                # So if I'm in the k loop, I assume solutions with < k terms were not found (or I'm skipping them).
                # To be safe, if rem_new == 0 and k-idx-1 > 0, I can't add more terms. So this path is dead.
                d += 1
                continue

            # Check max sum of remaining terms
            # max_sum_rem = sum_{i=0}^{k-idx-2} 1/(d+1+i)
            max_sum_rem = 0
            for i in range(k - idx - 1):
                max_sum_rem += Fraction(1, d + 1 + i)
                if max_sum_rem >= rem_new:
                    break
            
            if max_sum_rem < rem_new:
                # Even with smallest denominators, we can't reach rem_new.
                # And as d increases, max_sum_rem decreases and rem_new increases.
                # So we can break the loop entirely.
                break

            # Recursive call
            path.append(d)
            result = dfs(k, idx + 1, rem_new, d + 1, path)
            path.pop()
            
            if result:
                return result
            
            d += 1
        
        return None

    # Try k from 8 to 19
    for k in range(8, 20):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize path
        path = []
        # Start with min_d = 5
        # Remaining sum = 1
        result = dfs(k, 0, Fraction(1, 1), 5, path)
        if result:
            return result

    # Fallback to seed solution (k=20)
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
        if time.time() > deadline:
            break

    if target in reached:
        return reached[target]
    raise RuntimeError("no baseline representation")
