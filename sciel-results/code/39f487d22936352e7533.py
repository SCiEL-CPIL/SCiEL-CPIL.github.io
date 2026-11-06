"""Rewritten program: meet-in-the-middle powerful number generation + frequency-based candidate search."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def _gen_powerful(limit):
    """Generate all powerful numbers up to limit.
    
    A powerful number is of the form p1^e1 * p2^e2 * ... where every ei >= 2.
    We generate by considering square-free kernel * square.
    Actually, every powerful number can be written as a^2 * b^3 where b is square-free.
    So we enumerate square-free b and a such that a^2 * b^3 <= limit.
    """
    powerful = set()
    # b is square-free, b^3 <= limit => b <= limit^(1/3)
    max_b = int(limit ** (1/3)) + 2
    
    # Precompute square-free numbers up to max_b
    # A number is square-free if no square of a prime divides it
    is_square_free = [True] * (max_b + 1)
    is_square_free[0] = False
    if max_b >= 1:
        is_square_free[1] = True
    for p in range(2, isqrt(max_b) + 1):
        # Check if p is prime
        is_prime = True
        for q in range(2, isqrt(p) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            p2 = p * p
            for j in range(p2, max_b + 1, p2):
                is_square_free[j] = False
    
    for b in range(1, max_b + 1):
        if not is_square_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        # a^2 <= limit / b3
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin within 150s limit
    
    N_LIMIT = 10**18
    
    # Phase 1: Generate powerful numbers up to a reasonable limit
    # We need enough to find good candidates. Let's start with a moderate limit.
    # The number of powerful numbers up to X is ~ c * X^(1/3) + ...
    # For X = 10^12, that's about 10^4 numbers, manageable.
    # For X = 10^15, that's about 10^5 numbers.
    # For X = 10^18, that's about 10^6 numbers.
    
    # Strategy: Generate powerful numbers up to some limit L, then find N values
    # with many representations as sums of two powerful numbers from this set.
    # The key insight: if we use powerful numbers up to L, we can find N up to 2L.
    # To maximize representations, we want a dense set of powerful numbers.
    
    # Let's generate powerful numbers up to L where L is chosen to fit in time.
    # Start with a smaller L and increase if time permits.
    
    # First, generate powerful numbers up to a base limit
    base_limit = 10**12
    powerful = _gen_powerful(base_limit)
    p_set = set(powerful)
    n_p = len(powerful)
    
    if time.time() > deadline:
        # Fallback: use small set
        best_N = 2
        best_pairs = [[1, 1]]
        return {'N': best_N, 'pairs': best_pairs}
    
    # Phase 2: Count sums of pairs of powerful numbers
    # For each pair (i, j) with i <= j, compute sum = powerful[i] + powerful[j]
    # Count frequency of each sum
    # This is O(n_p^2) which for n_p = 10^4 is 10^8 - too slow in Python
    # For n_p = 10^3 it's 10^6 - fine
    # For n_p = 5000 it's 2.5 * 10^7 - borderline
    
    # Let's use a smarter approach: for each powerful number a, for each powerful number b >= a,
    # record the sum. But we need to be careful about time.
    
    # Alternative: use the fact that we want to find N with many representations.
    # Let's try a different approach: generate powerful numbers up to a smaller limit
    # and do pairwise sum counting.
    
    # Let's determine how many powerful numbers we can handle
    # O(n^2) pairwise sums. In Python, each iteration takes ~100ns.
    # 10^8 iterations = 10s. So n ~ 10^4 gives 10^8/2 = 5*10^7 ~ 5s.
    # Let's cap at n_p such that n_p^2/2 < 5*10^7, i.e., n_p < 10^4.
    
    # If n_p is too large, reduce the limit
    while n_p > 8000 and base_limit > 10**8:
        base_limit = int(base_limit * 0.5)
        powerful = _gen_powerful(base_limit)
        p_set = set(powerful)
        n_p = len(powerful)
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        # Use whatever we have
        pass
    
    # Now count pairwise sums
    sum_count = defaultdict(int)
    # For efficiency, iterate and count
    for i in range(n_p):
        a = powerful[i]
        for j in range(i, n_p):
            b = powerful[j]
            s = a + b
            sum_count[s] += 1
    
    if time.time() > deadline:
        # Fallback
        best_N = 2
        best_pairs = [[1, 1]]
        return {'N': best_N, 'pairs': best_pairs}
    
    # Find the N with the most representations
    # But we need N <= 10^18 and a, b powerful with a <= b, a + b = N
    # The sums we computed are all valid since a, b are powerful
    
    # Get top candidates
    sorted_sums = sorted(sum_count.items(), key=lambda x: -x[1])
    
    best_N = None
    best_count = 0
    best_pairs = []
    
    for N, count in sorted_sums:
        if N > N_LIMIT:
            continue
        if count > best_count:
            # Verify and collect pairs
            pairs = []
            for i in range(n_p):
                a = powerful[i]
                if 2 * a > N:
                    break
                b = N - a
                if b in p_set and b >= a:
                    pairs.append([a, b])
            if len(pairs) > best_count:
                best_count = len(pairs)
                best_N = N
                best_pairs = pairs
            if time.time() > deadline - 10:
                break
    
    if best_N is None:
        best_N = 2
        best_pairs = [[1, 1]]
    
    # Phase 3: Try to improve with a larger set if time permits
    if time.time() < deadline - 30:
        # Try with a larger limit
        larger_limit = min(base_limit * 4, 10**14)
        if larger_limit > base_limit:
            powerful2 = _gen_powerful(larger_limit)
            p_set2 = set(powerful2)
            n_p2 = len(powerful2)
            
            if n_p2 <= 12000:
                sum_count2 = defaultdict(int)
                for i in range(n_p2):
                    a = powerful2[i]
                    for j in range(i, n_p2):
                        b = powerful2[j]
                        s = a + b
                        sum_count2[s] += 1
                
                if time.time() < deadline - 10:
                    sorted_sums2 = sorted(sum_count2.items(), key=lambda x: -x[1])
                    for N, count in sorted_sums2:
                        if N > N_LIMIT:
                            continue
                        if count > best_count:
                            pairs = []
                            for i in range(n_p2):
                                a = powerful2[i]
                                if 2 * a > N:
                                    break
                                b = N - a
                                if b in p_set2 and b >= a:
                                    pairs.append([a, b])
                            if len(pairs) > best_count:
                                best_count = len(pairs)
                                best_N = N
                                best_pairs = pairs
                            if time.time() > deadline - 10:
                                break
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
