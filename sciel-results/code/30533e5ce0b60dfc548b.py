"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _sigma_sieve(n):
    """Compute sigma(i) for all i in 1..n using a divisor-sum sieve."""
    sigma = [0] * (n + 1)
    sigma[0] = 0
    for d in range(1, n + 1):
        for m in range(d, n + 1, d):
            sigma[m] += d
    return sigma

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    MAXV = 10000000

    # Precompute sigma up to MAXV
    sigma = _sigma_sieve(MAXV)

    # Group indices by their sigma value
    groups = {}
    for i in range(1, MAXV + 1):
        s = sigma[i]
        if s in groups:
            groups[s].append(i)
        else:
            groups[s] = [i]

    best_err = None  # (num, den) exact rational for |a/b - 3/2|
    best_pair = None

    # Strategy: for small |k| = |2a - 3b|, large b, check sigma equality.
    # a = (3b + k) / 2, so b must have parity matching k.
    # Error = |k| / (2b). Minimize by small |k|, large b.

    # Precompute for each b, the candidate a values for small k
    # But that's too many. Instead, iterate k from 1 upward, and for each k,
    # check large b values where a = (3b+k)//2 is valid and sigma matches.

    # For efficiency, build a set-based lookup: for each sigma value, store the set of indices.
    # But sets for large groups are expensive. Instead, use the list and check membership
    # via a dict mapping sigma -> set only for groups we actually need.

    # Better approach: iterate over k from 1 to K_MAX, and for each k, iterate b from high to low.
    # For each (b, k), compute a = (3*b + k) // 2, check bounds, then check sigma[a] == sigma[b].

    # To make this fast, precompute a set for each sigma value only when needed.
    # Actually, let's just use the sigma array directly for O(1) lookup.

    K_MAX = 200  # search up to |k| = 200
    # For each k, b must satisfy: 1 <= a <= MAXV, 1 <= b <= MAXV, a != b
    # a = (3b + k) / 2, so 3b + k must be even, and 2 <= 3b+k <= 2*MAXV

    for k in range(1, K_MAX + 1):
        if time.time() > deadline:
            break
        # b must have same parity as k for (3b+k) to be even
        # 3b + k is even iff b and k have same parity (since 3b has same parity as b)
        start_b = k % 2  # 0 if k even, 1 if k odd
        # b range: 3b + k >= 2 => b >= (2-k)/3, so b >= max(1, start_b)
        # 3b + k <= 2*MAXV => b <= (2*MAXV - k) / 3
        max_b = (2 * MAXV - k) // 3
        min_b = max(1, start_b)
        
        # Check b from high to low (larger b gives smaller error)
        # But we also need a = (3b+k)//2 to be in [1, MAXV] and a != b
        # a != b means (3b+k)/2 != b => b + k != 0 => always true for positive
        # a != b: 3b+k != 2b => b != -k, always true for positive b, k
        
        # We want to find ANY b with sigma[(3b+k)//2] == sigma[b]
        # Check from high b down, stop at first match for this k
        # But we also want the overall best, so track best across all k
        
        for b in range(min(max_b, MAXV), min_b - 1, -1):
            if (3 * b + k) % 2 != 0:
                continue
            a = (3 * b + k) // 2
            if a < 1 or a > MAXV:
                continue
            if a == b:
                continue
            if sigma[a] == sigma[b]:
                # Found a valid pair with error k/(2b)
                err_num = k
                err_den = 2 * b
                # Reduce fraction
                g = gcd(err_num, err_den)
                err_num //= g
                err_den //= g
                if best_err is None:
                    best_err = (err_num, err_den)
                    best_pair = (a, b)
                else:
                    # Compare err_num/err_den < best_err_num/best_err_den
                    if err_num * best_err[1] < best_err[0] * err_den:
                        best_err = (err_num, err_den)
                        best_pair = (a, b)
                break  # For this k, we found the best (largest b), move to next k
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if best_pair is None:
        # Fallback: return a known valid pair
        return [11, 6]
    
    return [int(best_pair[0]), int(best_pair[1])]
# EVOLVE-BLOCK-END
