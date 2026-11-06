"""Rewritten program: maximize number of powerful-number pairs summing to N <= 1e18."""
# EVOLVE-BLOCK-START
import time
import math
import sys
from collections import defaultdict

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers <= limit using n = a^2 * b^3 form."""
    pw = set()
    max_b = int(limit ** (1.0/3.0)) + 2
    # Sieve for squarefree numbers up to max_b
    is_sf = [True] * (max_b + 1)
    if max_b >= 0:
        is_sf[0] = False
    for p in range(2, int(math.isqrt(max_b)) + 1):
        if is_sf[p]:
            p2 = p * p
            for j in range(p2, max_b + 1, p2):
                is_sf[j] = False
    for b in range(1, max_b + 1):
        if not is_sf[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            continue
        max_a = int(math.isqrt(limit // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                pw.add(val)
    return sorted(pw)

def construct():
    deadline = time.time() + 135  # safety margin
    
    # Generate powerful numbers up to L
    L = 5_000_000
    powerful = _generate_powerful_numbers(L)
    pw_set = set(powerful)
    n_pw = len(powerful)
    sys.stderr.write(f"Generated {n_pw} powerful numbers up to {L}\n")
    
    if time.time() > deadline - 60:
        # Not enough time for pair enumeration; use a heuristic
        best_N = None
        best_count = 0
        # Try N values that are sums of two small powerful numbers
        for a in powerful[:100]:
            for b in powerful[:100]:
                if a > b:
                    continue
                N = a + b
                cnt = sum(1 for x in powerful if x <= N and (N - x) in pw_set)
                if cnt > best_count:
                    best_count = cnt
                    best_N = N
        if best_N is None:
            best_N = 2
        pairs = [[a, best_N - a] for a in powerful if a <= best_N - a and (best_N - a) in pw_set]
        if not pairs:
            pairs = [[1, 1]]
            best_N = 2
        return {'N': best_N, 'pairs': pairs}
    
    # Count pairs using a dict
    pair_counts = defaultdict(int)
    
    # Double loop with periodic deadline check
    for i in range(n_pw):
        if time.time() > deadline - 20:
            break
        ai = powerful[i]
        for j in range(i, n_pw):
            N = ai + powerful[j]
            pair_counts[N] += 1
    
    if not pair_counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find N with max count
    best_N = max(pair_counts, key=pair_counts.get)
    best_count = pair_counts[best_N]
    sys.stderr.write(f"Best N = {best_N}, count = {best_count}\n")
    
    # Collect all valid pairs for best_N
    pairs = []
    for a in powerful:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in pw_set:
            pairs.append([a, b])
    
    if not pairs:
        # Fallback
        for a in powerful:
            b = best_N - a
            if a <= b and b in pw_set:
                pairs.append([a, b])
                break
        if not pairs:
            pairs = [[1, 1]]
            best_N = 2
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
