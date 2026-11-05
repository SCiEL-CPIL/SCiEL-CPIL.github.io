import time
from math import isqrt
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # Generate powerful numbers up to a limit
    # Powerful number: every prime factor appears with exponent >= 2
    # Form: n = a^2 * b^3 where b is square-free
    # We'll generate up to a large limit to get many powerful numbers
    
    limit = 80_000_000  # 80 million
    
    powerful_set = set()
    
    # Generate all square-free numbers up to limit^(1/3)
    # b^3 <= limit => b <= limit^(1/3)
    b_max = int(limit ** (1.0/3.0)) + 1
    
    # For each square-free b, generate a^2 * b^3 for valid a
    for b in range(1, b_max + 1):
        # Check if b is square-free
        if not _is_square_free(b):
            continue
        b_cubed = b ** 3
        if b_cubed > limit:
            break
        a_max = isqrt(limit // b_cubed)
        for a in range(1, a_max + 1):
            val = a * a * b_cubed
            if val <= limit:
                powerful_set.add(val)
    
    powerful = sorted(powerful_set)
    powerful_members = set(powerful)
    n_powerful = len(powerful)
    
    # Count pair sums: for each pair (i, j) with i <= j, count powerful[i] + powerful[j]
    # Use a dictionary to count occurrences of each sum
    # To save time, only consider pairs where sum <= some max_N we care about
    # We want to find N with many representations, so let's count all sums up to 2*limit
    
    # Use Counter for efficiency
    sum_counts = Counter()
    
    # For each powerful number a, for each powerful number b >= a,
    # add to sum_counts[a + b]
    # This is O(K^2) where K = n_powerful
    
    # To be safe with time, limit the search
    # If n_powerful is ~23000, K^2/2 ~ 265 million which is too slow in Python
    # We need a smarter approach or accept partial work
    
    # Alternative: for each powerful number a, check how many b in powerful satisfy
    # that a + b gives a good N. But we don't know which N is good in advance.
    
    # Better: count all pair sums but with early termination if time runs out
    # Since K ~ 23000, K^2/2 ~ 265M iterations is too slow
    # Let's use a subset or a different strategy
    
    # Strategy: Use the fact that we want to maximize the count.
    # Generate pairs and count sums, but stop if time runs out.
    # We can process in chunks and keep the best found so far.
    
    best_N = 2
    best_count = 1  # at minimum, 2 = 1 + 1 if 1 is powerful
    best_pairs = [[1, 1]]
    
    # Check if 1 is in powerful
    if 1 not in powerful_members:
        # Add 1 as it's powerful by definition
        powerful.append(1)
        powerful.sort()
        powerful_members.add(1)
        n_powerful += 1
    
    # Count pair sums using a more efficient method
    # For each a in powerful, for each b in powerful where b >= a,
    # increment sum_counts[a+b]
    
    # To handle large K, we'll do this in a time-bounded way
    # Process pairs in order, tracking best
    
    # Actually, let's just do the full O(K^2) but with time check
    # If K is ~23000, that's ~265M iterations. In Python, ~100M/sec optimistically
    # So ~3 seconds. Should be fine.
    
    t0 = time.time()
    
    for i in range(n_powerful):
        a = powerful[i]
        for j in range(i, n_powerful):
            b = powerful[j]
            s = a + b
            if s > 10**18:
                break  # since powerful is sorted, further b will only increase s
            sum_counts[s] += 1
    
    t1 = time.time()
    
    # Find the N with maximum count
    if sum_counts:
        best_N, best_count = max(sum_counts.items(), key=lambda x: x[1])
        
        # Reconstruct pairs for best_N
        best_pairs = []
        for a in powerful:
            b = best_N - a
            if a <= b and b in powerful_members:
                best_pairs.append([a, b])
    
    # Ensure we have at least one valid pair
    if not best_pairs:
        # Fallback: find any valid representation
        for a in powerful:
            b = best_N - a
            if a <= b and b in powerful_members:
                best_pairs.append([a, b])
    
    # Final safety: if no pairs found, return a trivial case
    if not best_pairs:
        # 2 = 1 + 1, both powerful
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': best_pairs}


def _is_square_free(n):
    """Check if n is square-free (no prime factor appears more than once)."""
    if n <= 1:
        return True
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True
