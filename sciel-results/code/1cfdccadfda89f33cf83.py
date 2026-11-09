"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit

    # We know k >= 7 is required.
    # We will search for k=7 first. If found, it's optimal.
    # If not, try k=8, etc.
    
    def dfs_search(k, min_den=5):
        """
        Search for k distinct denominators >= min_den summing to 1.
        Returns list of denominators or None.
        """
        target = Fraction(1, 1)
        
        def dfs(start_d, remaining_sum, terms_remaining, current_list):
            # Time check
            if time.time() > deadline:
                return None

            # Base case
            if terms_remaining == 0:
                if remaining_sum == 0:
                    return current_list[:]
                return None

            if terms_remaining == 1:
                # remaining_sum must be 1/d
                # d = 1 / remaining_sum
                if remaining_sum > 0:
                    # Check if 1/remaining_sum is an integer
                    d_candidate = 1 / remaining_sum
                    if d_candidate == int(d_candidate):
                        d = int(d_candidate)
                        if d > start_d and d >= min_den and d <= 10000000:
                            return current_list + [d]
                return None

            # Pruning: Max possible sum with 'terms_remaining' terms starting from start_d+1
            # The smallest denominators are start_d+1, start_d+2, ...
            max_possible = Fraction(0, 1)
            for i in range(terms_remaining):
                max_possible += Fraction(1, start_d + 1 + i)
            if max_possible < remaining_sum:
                return None

            # Determine range for next denominator d
            # 1. 1/d <= remaining_sum  => d >= 1/remaining_sum
            d_min = max(start_d + 1, int(1 / remaining_sum) + 1)
            
            # 2. We need to be able to form (remaining_sum - 1/d) with (terms_remaining - 1) terms > d.
            # The maximum sum of (terms_remaining - 1) reciprocals with denominators > d 
            # is achieved by the smallest denominators: d+1, d+2, ...
            # So, remaining_sum - 1/d <= sum(1/(d+i) for i in 1..terms_remaining-1)
            # This provides an upper bound for d.
            # Let's iterate d from d_min upwards.
            # We can break if 1/d becomes too small such that even the max sum of remaining terms can't catch up?
            # Actually, if d is very large, 1/d is small, so remaining_sum - 1/d is close to remaining_sum.
            # The max sum of remaining terms (with denominators > d) decreases as d increases.
            # So there is a d_max.
            
            d = d_min
            while True:
                if time.time() > deadline:
                    return None
                
                # Upper bound check for d
                # If 1/d > remaining_sum, break (but d_min ensures 1/d <= remaining_sum)
                if Fraction(1, d) > remaining_sum:
                    break

                # Check if it's possible to fill the rest
                # rest_sum = remaining_sum - 1/d
                # max_rest_sum = sum(1/(d+i) for i in 1..terms_remaining-1)
                if terms_remaining > 1:
                    max_rest_sum = Fraction(0, 1)
                    for i in range(1, terms_remaining):
                        max_rest_sum += Fraction(1, d + i)
                    if remaining_sum - Fraction(1, d) > max_rest_sum:
                        break # d is too large, even with smallest next denominators, we can't reach the sum? 
                        # Wait. If d increases, 1/d decreases, so rest_sum increases.
                        # Also max_rest_sum decreases (denominators get larger).
                        # So if rest_sum > max_rest_sum, it's impossible for this d AND any larger d.
                        # Because for larger d', 1/d' < 1/d, so rest_sum' > rest_sum.
                        # And max_rest_sum' < max_rest_sum.
                        # So the gap widens. Thus we can break.
                        pass

                new_remaining = remaining_sum - Fraction(1, d)
                if new_remaining < 0:
                    break
                
                result = dfs(d, new_remaining, terms_remaining - 1, current_list + [d])
                if result is not None:
                    return result
                
                d += 1
                # Optional: limit d to avoid infinite loops if no solution, but bounds should handle it.
                # If d gets very large, 1/d is tiny.
                # If d > 10000000, break.
                if d > 10000000:
                    break

            return None

        return dfs(min_den - 1, target, k, [])

    # Try k=7 first
    for k in range(7, 10): # Try up to k=9 just in case
        if time.time() > deadline:
            break
        result = dfs_search(k)
        if result is not None:
            # Validate
            s = Fraction(0, 1)
            for d in result:
                s += Fraction(1, d)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result

    # Fallback: DP method from original (guarantees valid, but not minimal)
    target = 2520
    reached = {0: (0, [])}
    
    for d in range(5, target + 1):
        if target % d != 0:
            continue
        w = target // d
        for total, (terms, values) in list(reached.items()):
            new_total = total + w
            if new_total > target:
                continue
            new_terms = terms + 1
            if new_total not in reached or new_terms < reached[new_total][0]:
                reached[new_total] = (new_terms, values + [d])
    
    if target in reached:
        result = reached[target][1]
        s = Fraction(0, 1)
        for d in result:
            s += Fraction(1, d)
        if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
            return result
            
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
