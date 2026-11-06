"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def _divisors(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return sorted(divs)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s/90s limit

    n = 5040
    target = 4979
    divs = _divisors(n)
    div_set = set(divs)
    D = len(divs)
    # D is 60 for 5040

    # Filter divisors to those <= target (all are, since max divisor is 5040 > 4979, so filter out 5040)
    # Actually 5040 > 4979, so we can remove it.
    valid_divs = [d for d in divs if d <= target]
    valid_set = set(valid_divs)
    D_valid = len(valid_divs)

    # k=1
    for d in valid_divs:
        if d == target:
            return [d]

    # k=2
    for i in range(D_valid):
        a = valid_divs[i]
        b = target - a
        if b in valid_set and b != a:
            return [a, b]

    # k=3: Use meet-in-the-middle style or optimized triple loop
    # Since D_valid is small (~59), a triple loop is feasible but we can be smart.
    # Sort descending to find large numbers first? No, we just need ANY solution.
    # The seed program's triple loop is O(D^2) with set lookups, which is fast.
    for i in range(D_valid):
        a = valid_divs[i]
        for j in range(i + 1, D_valid):
            b = valid_divs[j]
            rem = target - a - b
            if rem in valid_set and rem != a and rem != b:
                return [a, b, rem]

    # k=4: 
    # Seed program has O(D^3) loop. With D=59, that's ~59^3/6 ~ 34k iterations, each with set lookup.
    # This is very fast.
    for i in range(D_valid):
        a = valid_divs[i]
        for j in range(i + 1, D_valid):
            b = valid_divs[j]
            rem = target - a - b
            if rem <= 0:
                continue
            # We need c + d = rem, with c, d distinct from a, b, and each other.
            # And c, d must be in valid_set.
            # To avoid O(D^2) inside, we can iterate c and check d = rem - c.
            # But we must ensure c > b (to maintain distinctness and order? Not strictly required if we just check set)
            # However, to avoid duplicates and ensure distinctness, let's iterate.
            for k in range(j + 1, D_valid):
                c = valid_divs[k]
                d = rem - c
                if d in valid_set and d != a and d != b and d != c:
                    return [a, b, c, d]
                # Time check
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # k=5:
    # The seed program hardcodes a solution: [2520, 1680, 560, 210, 9]
    # Let's verify this is valid.
    # 2520 + 1680 + 560 + 210 + 9 = 4979.
    # 2520 is a divisor of 5040? 5040 / 2520 = 2. Yes.
    # 1680 is a divisor of 5040? 5040 / 1680 = 3. Yes.
    # 560 is a divisor of 5040? 5040 / 560 = 9. Yes.
    # 210 is a divisor of 5040? 5040 / 210 = 24. Yes.
    # 9 is a divisor of 5040? 5040 / 9 = 560. Yes.
    # They are distinct.
    # So this is a valid k=5 solution.
    # Is there a k=4 solution? The loop above should have found it if it exists and completed.
    # If the loop was interrupted by time, we might miss a k=4 solution, but k=5 is guaranteed.
    # However, the problem asks to MINIMIZE the number.
    # If a k=4 solution exists, we must return it.
    # The O(D^3) loop for k=4 is fast enough to complete well within 55s.
    # So if we reach here, no k=4 solution was found.
    
    result = [2520, 1680, 560, 210, 9]
    # Verify again just in case
    if all(x in valid_set for x in result) and len(set(result)) == len(result) and sum(result) == target:
        return result

    # Fallback: This should never happen for this specific instance
    # But for robustness, let's do a greedy or random search?
    # Or just return the hardcoded one even if verification fails? No, that would be wrong.
    # But since we know it's valid, we can just return it.
    return result

# EVOLVE-BLOCK-END
