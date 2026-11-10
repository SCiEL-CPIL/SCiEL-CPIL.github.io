"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # The minimum number of terms is 8.
    # 1/5 + ... + 1/11 < 1, so 7 terms are impossible.
    # We search for a solution with exactly 8 terms first.
    
    def find_solution_k(k):
        """
        Finds k distinct denominators >= 5 summing to 1.
        Uses DFS with pruning.
        """
        def dfs(rem_num, rem_den, min_d, curr, depth):
            if time.time() > deadline:
                return None
            
            if depth == k:
                if rem_num == 0:
                    return curr
                return None
            
            # Pruning: Check if max possible sum of remaining terms is sufficient
            # Max sum is achieved by picking the smallest available denominators:
            # min_d, min_d+1, ..., min_d + (k-depth-1)
            # Sum = sum(1/(min_d + i) for i in range(k-depth))
            # If this sum < rem_num/rem_den, prune.
            # To avoid floating point issues, compare fractions or use high precision.
            # Or simply: if (k-depth) / min_d < rem_num / rem_den, prune?
            # This is a lower bound on the max sum? No, (k-depth)/min_d is an upper bound on the sum
            # if all denominators were min_d. But they are distinct and increasing.
            # So actual max sum < (k-depth) / min_d.
            # If (k-depth) / min_d <= rem_num / rem_den, it's definitely impossible?
            # No, if (k-depth)/min_d < rem_num/rem_den, then even if we could use min_d for all,
            # we wouldn't reach the target. So we can prune.
            if (k - depth) * rem_den <= rem_num * min_d:
                return None
            
            # Bounds for next denominator d
            # 1/d <= rem_num/rem_den => d >= rem_den / rem_num
            min_d_calc = rem_den // rem_num + 1
            if rem_num * (min_d_calc - 1) >= rem_den:
                 # Handle case where rem_den/rem_num is integer
                 # If rem_num/rem_den = 1/2, d >= 2.
                 # rem_den=2, rem_num=1. min_d_calc = 2//1 + 1 = 3.
                 # 1 * (3-1) = 2 >= 2. True.
                 # But 1/2 <= 1/2 is valid. So d=2 is allowed.
                 # My logic for min_d_calc needs to be careful.
                 # Condition: 1/d <= R => d >= 1/R.
                 # If R = 1/2, d >= 2.
                 # rem_den // rem_num = 2 // 1 = 2.
                 # So min_d_calc should be 2.
                 # My formula: rem_den // rem_num + 1 = 3. This is too high.
                 # Correct formula: d >= ceil(rem_den / rem_num).
                 # If rem_den % rem_num == 0: min_d_calc = rem_den // rem_num
                 # else: min_d_calc = rem_den // rem_num + 1
                 
                 if rem_den % rem_num == 0:
                     min_d_calc = rem_den // rem_num
                 else:
                     min_d_calc = rem_den // rem_num + 1
            else:
                 if rem_den % rem_num == 0:
                     min_d_calc = rem_den // rem_num
                 else:
                     min_d_calc = rem_den // rem_num + 1
            
            # 1/d >= rem_num/rem_den / (k-depth) => d <= (k-depth) * rem_den / rem_num
            max_d_calc = ((k - depth) * rem_den) // rem_num
            
            start_d = max(min_d, min_d_calc, 5)
            
            for d in range(start_d, max_d_calc + 1):
                if d in curr:
                    continue
                
                new_num = rem_num * d - rem_den
                if new_num < 0:
                    break
                new_den = rem_den * d
                
                g = gcd(new_num, new_den)
                if g > 1:
                    new_num //= g
                    new_den //= g
                
                res = dfs(new_num, new_den, d + 1, curr + [d], depth + 1)
                if res:
                    return res
            return None
        
        return dfs(1, 1, 5, [], 0)

    # Try k=8 first, as it's the theoretical minimum
    for k in range(8, 15):
        result = find_solution_k(k)
        if result is not None:
            # Verify
            total = sum(Fraction(1, d) for d in result)
            if total == 1 and len(result) == len(set(result)) and all(5 <= d <= 10000000 for d in result):
                return result

    # Fallback: use the dynamic programming approach from the seed if DFS fails
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
    
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
