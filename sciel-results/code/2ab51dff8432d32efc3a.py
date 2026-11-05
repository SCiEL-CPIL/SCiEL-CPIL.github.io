"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s to be safe under 90s limit
    N = 512
    
    # 1. Identify the best "Divisor Clique" candidates
    # A set of divisors of k is always a valid clique if k <= N.
    # We want to maximize the size of such a set, which is tau(k).
    
    def count_divisors(x):
        cnt = 0
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                cnt += 1
                if d != x // d:
                    cnt += 1
        return cnt

    # Find all k in [1, 512] and their divisor counts
    div_counts = []
    for k in range(1, N + 1):
        div_counts.append((count_divisors(k), k))
    
    # Sort by divisor count descending
    div_counts.sort(key=lambda x: -x[0])
    
    # Top candidates for initial cliques
    top_candidates = []
    for cnt, k in div_counts[:10]:
        # Get the set of divisors of k
        divs = set()
        for d in range(1, int(k**0.5) + 1):
            if k % d == 0:
                divs.add(d)
                divs.add(k // d)
        # Filter to [1, 512] (all are <= k <= 512)
        top_candidates.append(divs)

    best_set = set()
    best_size = 0
    
    # Function to check if a set is a valid clique
    def is_valid_clique(s):
        s_list = list(s)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                a = s_list[i]
                b = s_list[j]
                if a * b // gcd(a, b) > N:
                    return False
        return True

    # Function to try to expand a clique
    def expand_clique(s):
        s = set(s)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x in s:
                    continue
                # Check if x is compatible with all in s
                ok = True
                for y in s:
                    if x * y // gcd(x, y) > N:
                        ok = False
                        break
                if ok:
                    s.add(x)
                    changed = True
        return s

    # Evaluate top candidates
    for cand in top_candidates:
        # Expand the candidate set
        expanded = expand_clique(cand)
        if len(expanded) > best_size:
            best_size = len(expanded)
            best_set = expanded

    # If the best size is still small (unlikely), fall back to a generic search
    # But given the structure, we expect best_size >= 24
    
    # Final verification and conversion
    if not best_set:
        best_set = {1}
        
    # Double check validity of the best set found
    if not is_valid_clique(best_set):
        # Fallback to the largest valid divisor set if expansion broke something (shouldn't happen)
        for cand in top_candidates:
            if is_valid_clique(cand):
                best_set = cand
                break

    result = sorted(list(best_set))
    return result

# EVOLVE-BLOCK-END
