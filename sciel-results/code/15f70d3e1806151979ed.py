"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from math import lcm
from itertools import product

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
DEADLINE_FRACTION = 0.85  # safety margin for 90s wall-clock


def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION

    divisors = DIVISORS
    n = len(divisors)

    # Precompute: for each divisor d_i, the set of residues mod LCM
    # Actually, we just need to know which multiples of d_i fall in a window [L, L+W]
    # A multiple of d_i is in [L, L+W] iff there exists k such that k*d_i >= L and k*d_i <= L+W
    # i.e., ceil(L/d_i) <= k <= floor((L+W)/d_i)
    # Such a k exists iff ceil(L/d_i) <= floor((L+W)/d_i)

    # We want to minimize W such that for some L, for every i, there's a multiple of d_i in [L, L+W],
    # AND we can assign distinct multiples (one per divisor) within that window.

    # The distinctness constraint is key. If two divisors share the same multiple in the window,
    # we need at least two distinct multiples for those divisors.

    # Strategy: sliding window with backtracking/matching
    # But first, let's find a good W via a smarter approach.

    # Key insight: The LCM of all divisors is:
    LCM_VAL = 1
    for d in divisors:
        LCM_VAL = lcm(LCM_VAL, d)
    # LCM_VAL = lcm(6,10,14,15,21,22,26,33,35,39)
    # Let's compute: 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # LCM = 2*3*5*7*11*13 = 30030

    # Within any interval of length LCM_VAL, the pattern of multiples repeats.
    # So we only need to search L in [0, LCM_VAL-1] and W from 1 upward.

    # For a given W, check if there exists L in [0, LCM_VAL-1] such that
    # we can assign distinct multiples.

    # For each divisor d_i, the multiples in [L, L+W] are:
    # k*d_i where k ranges from ceil(L/d_i) to floor((L+W)/d_i)
    # Let count_i = floor((L+W)/d_i) - ceil(L/d_i) + 1 (if >= 0, else 0)

    # We need a system of distinct representatives (SDR) from the sets S_i = {k*d_i : k in [ceil(L/d_i), floor((L+W)/d_i)]}

    # Use Hall's theorem or bipartite matching.

    # First, let's try small W values. The minimum possible W is at least 1 (since distinct).
    # Let's do a binary search on W, or just iterate W from 1 upward.

    # Actually, let's iterate W from 1 upward, and for each W, check all L in [0, LCM_VAL-1].
    # But LCM_VAL = 30030, and W could be up to... let's see. The seed gives W = 39-6 = 33.
    # We can do better.

    # Let's use a more efficient check: for a given (L, W), build the bipartite graph
    # and run a matching algorithm.

    # To speed up, for each L, we can compute the available multiples for each divisor.

    # Let's precompute for each L in [0, LCM_VAL-1] and for each W, the sets.
    # But that's 30030 * many W values. Let's be smart.

    # Alternative: for each W, iterate L from 0 to LCM_VAL-1, and for each L,
    # compute the multiples for each divisor. If any divisor has 0 multiples, skip.
    # Then check if SDR exists via bipartite matching.

    # To make this fast, we'll use a greedy + backtracking approach or Hopcroft-Karp.
    # But with only 10 divisors, a simple DFS-based matching should be fast enough.

    def check_window(L, W):
        """Check if we can assign distinct multiples for each divisor in [L, L+W]."""
        # For each divisor, compute the list of multiples in [L, L+W]
        sets = []
        for i, d in enumerate(divisors):
            k_min = (L + d - 1) // d  # ceil(L/d)
            k_max = (L + W) // d       # floor((L+W)/d)
            if k_min > k_max:
                return None  # No multiple for this divisor
            multiples = [k * d for k in range(k_min, k_max + 1)]
            sets.append(multiples)

        # Now check if SDR exists using bipartite matching
        # Left side: divisors (0..n-1), Right side: multiples
        # Build adjacency
        # Use DFS-based augmenting path

        # Collect all unique multiples
        all_multiples = set()
        for s in sets:
            all_multiples.update(s)
        all_multiples = list(all_multiples)
        # Map multiple to index
        mult_to_idx = {m: i for i, m in enumerate(all_multiples)}
        m_count = len(all_multiples)

        # Adjacency: adj[i] = list of right-side indices for left node i
        adj = []
        for i in range(n):
            adj.append([mult_to_idx[m] for m in sets[i]])

        # Bipartite matching via DFS
        match_r = [-1] * m_count  # match_r[j] = left node matched to right node j, or -1

        def dfs(u, visited):
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if match_r[v] == -1 or dfs(match_r[v], visited):
                    match_r[v] = u
                    return True
            return False

        match_count = 0
        for u in range(n):
            visited = [False] * m_count
            if dfs(u, visited):
                match_count += 1

        if match_count != n:
            return None

        # Extract the assignment
        assignment = [0] * n
        for v in range(m_count):
            if match_r[v] != -1:
                assignment[match_r[v]] = all_multiples[v]

        return assignment

    # Now search: for W from 1 upward, for L from 0 to LCM_VAL-1
    # But this could be slow if W is large. Let's start with W=1 and go up.
    # The seed has W=33, so we should find something better quickly.

    # To speed up, we can skip L values that are "equivalent" in some sense,
    # but let's just iterate.

    # Actually, let's be smarter: for a given W, the condition for divisor d_i to have
    # at least one multiple in [L, L+W] is that the interval [L, L+W] intersects
    # the set of multiples of d_i. The multiples of d_i are spaced d_i apart.
    # So [L, L+W] contains a multiple of d_i iff L mod d_i <= W or (L+W) mod d_i < L mod d_i ... 
    # Actually, it's simpler: there exists k such that k*d_i in [L, L+W].
    # This is equivalent to: floor((L+W)/d_i) >= ceil(L/d_i)

    # Let's just iterate L and W.

    best_result = None
    best_W = float('inf')

    # We'll iterate W from 1 upward. For each W, iterate L from 0 to LCM_VAL-1.
    # But 30030 L values per W, and W could be up to ~33 or less.
    # 30030 * 33 = ~1M checks, each with a matching of size 10. Should be fast.

    # Let's cap W at some reasonable value, say 100, to be safe.
    # But we'll stop early if we find a good W.

    for W in range(1, 200):
        if time.time() > deadline:
            break
        found = False
        for L in range(LCM_VAL):
            if time.time() > deadline:
                break
            result = check_window(L, W)
            if result is not None:
                # Verify: all distinct, all <= MAX_VAL, all divisible
                if len(set(result)) == n and all(r <= MAX_VAL for r in result):
                    # Check divisibility
                    valid = True
                    for i, r in enumerate(result):
                        if r % divisors[i] != 0:
                            valid = False
                            break
                    if valid:
                        # This is a valid solution with width W
                        if W < best_W:
                            best_W = W
                            best_result = result[:]
                        found = True
                        # We can break out of L loop for this W since we found one
                        # But maybe a different L gives same W, doesn't matter
                        break
        if found:
            # We found the minimal W (since we iterate W from 1)
            # Actually, we should verify it's truly minimal, but since we go in order, it is.
            break

    if best_result is not None:
        # Ensure types are Python int
        return [int(x) for x in best_result]
    else:
        # Fallback: return the seed solution
        return DIVISORS[:]
# EVOLVE-BLOCK-END
