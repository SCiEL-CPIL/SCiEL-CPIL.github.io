"""Rewritten program: targeted DFS for few-term Egyptian fractions with denominators > 4."""
import time
import math

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s kill

    def check_time():
        return time.time() > deadline

    # Known small representations to try first (fewer terms = better)
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
    # But we want fewer. Let's try known 5-term, 6-term, etc.
    # 1 = 1/2 + 1/3 + 1/6 (not valid, denominators must be > 4)
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 (26 terms)
    # Better: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
    # Let's try to find shorter ones via DFS

    def dfs(remaining_num, remaining_den, min_den, current, depth, max_terms, best_result, best_count):
        """
        DFS to find Egyptian fraction representation.
        Returns (found, best_result, best_count)
        """
        if check_time():
            return False, best_result, best_count

        if remaining_num == 0:
            if len(current) < best_count:
                best_count = len(current)
                best_result = list(current)
            return True, best_result, best_count

        if depth >= max_terms or remaining_num <= 0:
            return False, best_result, best_count

        # Pruning: max possible sum with remaining terms
        # We need sum of 1/d_i for i in [depth, max_terms) >= remaining_num/remaining_den
        # Max sum is achieved with smallest denominators: 1/min_den + 1/(min_den+1) + ...
        # But simpler: if remaining_num * min_den > remaining_den * (max_terms - depth), we can't fit
        if remaining_num * min_den > remaining_den * (max_terms - depth):
            return False, best_result, best_count

        d_low = max(min_den, (remaining_den + remaining_num - 1) // remaining_num)

        K_rem = max_terms - depth - 1
        if K_rem < 0:
            return False, best_result, best_count

        # Estimate d_high: with K_rem remaining terms, the largest denominator d can be
        # such that 1/d + 1/(d+1) + ... >= remaining - 1/d
        # Approximation: d_high = K_rem * remaining_den / remaining_num
        d_high = int(K_rem * remaining_den / remaining_num) + 100
        d_high = min(d_high, 10000000)

        if d_high < d_low:
            return False, best_result, best_count

        # Limit the range to avoid huge iterations
        if d_high - d_low > 100000:
            d_high = d_low + 100000

        for d in range(d_low, d_high + 1):
            if check_time():
                return False, best_result, best_count

            # Check if 1/d <= remaining
            if d * remaining_num < remaining_den:
                continue

            # Check distinctness
            if d in current:
                continue

            new_num = remaining_num * d - remaining_den
            new_den = remaining_den * d

            if new_num == 0:
                current.append(d)
                if len(current) < best_count:
                    best_count = len(current)
                    best_result = list(current)
                current.pop()
                return True, best_result, best_count

            current.append(d)
            found, best_result, best_count = dfs(new_num, new_den, d + 1, current, depth + 1, max_terms, best_result, best_count)
            current.pop()

            if best_count <= depth + 1:
                return True, best_result, best_count

        return False, best_result, best_count

    # Try to find shortest representation
    best_result = None
    best_count = 256  # max allowed

    # Start from small number of terms and increase
    for k in range(5, 257):
        if check_time():
            break
        found, best_result, best_count = dfs(1, 1, 5, [], 0, k, best_result, best_count)
        if found and best_count == k:
            break

    if best_result is None:
        # Fallback: use a known valid representation
        # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
        # Verify: sum of 1/i for i=5..30
        # Let's just use a simpler known one or compute
        # Actually, let's use the greedy algorithm as fallback
        def greedy_egyptian(target_num, target_den, min_den=5, max_terms=256):
            result = []
            num, den = target_num, target_den
            for _ in range(max_terms):
                if num == 0:
                    return result
                if num * min_den > den * (max_terms - len(result)):
                    return None

                d_low = max(min_den, (den + num - 1) // num)
                d = d_low

                if num * d < den:
                    d += 1

                if d in result or d > 10000000:
                    for d_try in range(d + 1, d + 1000):
                        if d_try not in result and d_try <= 10000000:
                            d = d_try
                            break
                    else:
                        return None

                result.append(d)

                new_num = num * d - den
                new_den = den * d

                if new_num == 0:
                    return result

                g = math.gcd(new_num, new_den)
                num = new_num // g
                den = new_den // g

                min_den = d + 1

                if check_time():
                    return None

            if num == 0:
                return result
            return None

        best_result = greedy_egyptian(1, 1, 5, 256)

    if best_result is None:
        raise RuntimeError("no representation found")

    return best_result
