"""Rewritten Egyptian-fraction solver: fixed small solution + fast verification."""
import time
from fractions import Fraction

# A known 10-term Egyptian fraction representation of 1 with denominators in [5, 10000000].
# 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + ... (need exact)
# Let's use a verified solution:
# 1 = 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + ...
# Actually, let me use a well-known compact solution:
# 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24
# That's 20 terms. Let me find a shorter verified one.

# Verified: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24
# Sum check: H(24) - H(4) = (1+1/2+1/3+1/4+...+1/24) - (1+1/2+1/3+1/4) = 1/5+...+1/24
# H(24) ≈ 3.7759576, H(4) = 2.0833333, diff ≈ 1.6926243 ≠ 1

# Let me use a known exact solution:
# 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40
# That's 36 terms. Too many.

# Better approach: use the known minimal solution.
# 1 = 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50
# H(50)-H(5) ≈ 4.4992 - 2.2833 = 2.2159, still not 1.

# The original program uses target=2520 and finds a subset of divisors. Let me just use that approach but make it faster.

def _construct_fast():
    """Use the original subset DP approach with target=2520, which is fast."""
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total in list(reached.keys()):
            new_total = total + w
            if new_total <= target and new_total not in reached:
                reached[new_total] = reached[total] + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")

def construct():
    """Return 1..256 distinct integer denominators in [5,10000000] whose reciprocal sum is exactly 1."""
    start = time.time()
    deadline = start + 55.0  # 55s safety margin within 60s limit
    
    # Try the fast DP approach first
    try:
        result = _construct_fast()
        # Verify
        total = sum(Fraction(1, d) for d in result)
        if total == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result) and 1 <= len(result) <= 256:
            return result
    except Exception:
        pass
    
    # Fallback: brute force search for a small solution
    # Try known solutions or construct one
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50
    # Let's just use a verified hardcoded solution
    
    # Verified solution: 1 = 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + 1/51 + 1/52 + 1/53 + 1/54 + 1/55 + 1/56 + 1/57 + 1/58 + 1/59 + 1/60
    # H(60)-H(5) ≈ 4.6799 - 2.2833 = 2.3966, not 1.
    
    # Let me just return the result from the DP which is guaranteed to work
    # The DP with target=2520 always finds a solution
    result = _construct_fast()
    total = sum(Fraction(1, d) for d in result)
    assert total == 1, f"Sum is {total}, not 1"
    assert all(5 <= d <= 10000000 for d in result)
    assert len(set(result)) == len(result)
    assert 1 <= len(result) <= 256
    return result
