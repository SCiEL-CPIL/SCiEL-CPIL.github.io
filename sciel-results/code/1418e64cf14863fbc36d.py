"""Few-term Egyptian fractions with denominators greater than four."""
from fractions import Fraction
from math import gcd
import time

def construct():
    """Return 1..256 distinct integer denominators in [5,10000000]
    whose reciprocal sum is exactly 1; minimize their count."""
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill

    # Search for minimal k from small to large
    for k in range(8, 16):
        if time.time() > deadline - 5:
            break
        result = _search(k, deadline)
        if result is not None:
            return result

    # Fallback: use a known valid solution
    return _fallback()


def _max_sum(remaining, start_d):
    """Return (num, den) for sum of 1/(start_d+i) for i in range(remaining).
    Uses iterative addition to avoid huge intermediate fractions."""
    if remaining == 0:
        return (0, 1)
    num, den = 0, 1
    for i in range(remaining):
        d = start_d + i
        # num/den + 1/d = (num*d + den) / (den*d)
        num = num * d + den
        den = den * d
        g = gcd(num, den)
        if g > 1:
            num //= g
            den //= g
    return (num, den)


def _search(k, deadline):
    """Search for k distinct denominators >= 5 summing to 1 using integer arithmetic."""

    def dfs(idx, num, den, last_denom, chosen):
        """Find k-idx denominators > last_denom, each >= 5, summing to num/den."""
        if time.time() > deadline:
            return False

        remaining = k - idx

        if remaining == 0:
            return num == 0

        if num <= 0 or num > den:
            return False

        # Prune: max possible sum of remaining terms with smallest denominators
        m_num, m_den = _max_sum(remaining, last_denom + 1)
        if num * m_den > m_num * den:
            return False

        if remaining == 1:
            # Need 1/d = num/den, so d = den/num must be integer
            if den % num == 0:
                d_val = den // num
                if d_val > last_denom and d_val >= 5 and d_val <= 10000000:
                    chosen.append(d_val)
                    if dfs(idx + 1, 0, 1, d_val, chosen):
                        return True
                    chosen.pop()
            return False

        # Bounds for next denominator d:
        # 1/d < num/den (strictly, since we need more positive terms)
        # So d > den/num
        min_d = max(last_denom + 1, den // num + 1)

        # Upper bound: we need 1/d + max_sum(remaining-1, d+1) >= num/den
        # Loose bound: d <= remaining * den // num
        max_d = min(10000000, (remaining * den) // num + 10)

        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return False

            # Check if 1/d_val < num/den (strictly)
            if d_val * num <= den:
                # 1/d_val >= num/den, too large or equal
                continue

            # new_num/den_new = num/den - 1/d_val = (num*d_val - den) / (den*d_val)
            new_num = num * d_val - den
            new_den = den * d_val

            if new_num <= 0:
                continue

            # Simplify
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g

            chosen.append(d_val)
            if dfs(idx + 1, new_num, new_den, d_val, chosen):
                return True
            chosen.pop()

        return False

    chosen = []
    if dfs(0, 1, 1, 4, chosen):
        return chosen
    return None


def _fallback():
    """Return a known valid solution."""
    # Verify the 11-term solution
    fb = [5, 6, 7, 10, 14, 20, 28, 42, 70, 105, 210]
    s = sum(Fraction(1, d) for d in fb)
    if s == 1 and all(5 <= d <= 10000000 for d in fb) and len(set(fb)) == len(fb):
        return fb

    # Dynamic programming fallback over divisors of 2520
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]

    raise RuntimeError("no valid representation found")
