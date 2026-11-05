"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 90s kill

    def search_k(k, deadline):
        """Find k distinct denominators >= 5 summing reciprocals to 1 using integer arithmetic."""
        found = [None]

        def dfs(rem_num, rem_den, min_den, current, depth):
            if time.time() > deadline:
                return
            if depth == k:
                if rem_num == 0:
                    found[0] = list(current)
                return
            if rem_num <= 0:
                return

            remaining_slots = k - depth

            # Upper bound: max sum with remaining_slots terms >= min_den is remaining_slots/min_den
            if rem_num * min_den > rem_den * remaining_slots:
                return

            # Lower bound: min sum with remaining_slots distinct denominators from min_den
            min_sum_num = 0
            min_sum_den = 1
            for i in range(remaining_slots):
                d = min_den + i
                new_num = min_sum_num * d + min_sum_den
                new_den = min_sum_den * d
                g = math.gcd(new_num, new_den)
                min_sum_num = new_num // g
                min_sum_den = new_den // g
            if rem_num * min_sum_den < rem_den * min_sum_num:
                return

            d_low = max(min_den, (rem_den + rem_num - 1) // rem_num)
            d_high = min(10000000, max(d_low, rem_den * k + 1000))

            for d in range(d_low, d_high + 1):
                if time.time() > deadline:
                    return
                new_num = rem_num * d - rem_den
                new_den = rem_den * d
                if new_num < 0:
                    break
                if new_num == 0:
                    if depth + 1 == k:
                        current.append(d)
                        found[0] = list(current)
                        current.pop()
                        return
                    else:
                        continue
                g = math.gcd(new_num, new_den)
                new_num_r = new_num // g
                new_den_r = new_den // g
                current.append(d)
                dfs(new_num_r, new_den_r, d + 1, current, depth + 1)
                current.pop()
                if found[0] is not None:
                    return

        dfs(1, 1, 5, [], 0)
        return found[0]

    for k in range(5, 15):
        if time.time() > deadline - 3:
            break
        result = search_k(k, deadline - 3)
        if result is not None:
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
                return result

    # Fallback DP
    target = 420
    divisors = [d for d in range(5, target + 1) if target % d == 0]
    dp = {0: []}
    for d in divisors:
        w = target // d
        for total, values in list(dp.items()):
            new_total = total + w
            if new_total <= target and new_total not in dp:
                dp[new_total] = values + [d]
        if target in dp:
            result = dp[target]
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
                return result

    raise RuntimeError("no representation found")

# EVOLVE-BLOCK-END
