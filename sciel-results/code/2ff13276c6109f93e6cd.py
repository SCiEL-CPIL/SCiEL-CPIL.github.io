"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

N = 12
PERIOD = 27720  # lcm(1..12)
MAX_START = 1000000
DEADLINE = 85.0  # safety margin under 90s

def construct():
    t0 = time.time()
    
    # Precompute: for each residue r in 0..PERIOD-1, and each L in 1..144,
    # check if (m, m+L] has a valid matching for all i=1..12.
    # m mod PERIOD = r means m = r + k*PERIOD.
    # For i, valid b_i in (m, m+L] are those with b_i ≡ 0 (mod i).
    # Since PERIOD is a multiple of i, the set of valid residues mod i is the same
    # for all m with the same r mod PERIOD.
    
    # For a given r and L, the candidates for i are:
    #   b_i in (m, m+L] with i | b_i
    #   => b_i = m + d where 1 <= d <= L and (m + d) % i == 0
    #   => d % i == (-m) % i == (PERIOD - r) % i  ... wait
    #   m = r + k*PERIOD, so m % i = r % i (since i | PERIOD)
    #   So we need (r + k*PERIOD + d) % i == 0 => (r + d) % i == 0
    #   => d ≡ -r (mod i), 1 <= d <= L
    
    # For each r, precompute for each i the valid d values in 1..144
    # Then for each L, check if there's a system of distinct representatives.
    
    # Precompute valid_d[i][r] = list of d in 1..144 such that (r + d) % i == 0
    valid_d = [[[] for _ in range(PERIOD)] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for r in range(PERIOD):
            # d in 1..144 with (r + d) % i == 0 => d ≡ (-r) % i (mod i)
            rem = (-r) % i
            # d values: rem, rem+i, rem+2i, ... but d >= 1
            # if rem == 0, d = i, 2i, 3i, ...
            # if rem != 0, d = rem, rem+i, rem+2i, ...
            if rem == 0:
                start = i
            else:
                start = rem
            vals = []
            d = start
            while d <= 144:
                vals.append(d)
                d += i
            valid_d[i][r] = vals
    
    # For each r, find the maximum L such that matching exists.
    # Use bipartite matching (Hopcroft-Karp or DFS augmenting path).
    # Left side: i = 1..12. Right side: d = 1..144.
    # Edge (i, d) exists if d in valid_d[i][r].
    # We want to check if for a given L, there's a matching using only d <= L.
    
    def has_matching(r, L):
        """Check if there's a matching for all i=1..12 using d in 1..L."""
        # Build adjacency: for each i, list of d in 1..L that are valid
        adj = []
        for i in range(1, N + 1):
            adj.append([d for d in valid_d[i][r] if d <= L])
        
        # DFS-based bipartite matching
        match_d = [-1] * (L + 1)  # match_d[d] = i matched to d, or -1
        
        def dfs(i, visited):
            for d in adj[i - 1]:
                if visited[d]:
                    continue
                visited[d] = True
                if match_d[d] == -1 or dfs(match_d[d], visited):
                    match_d[d] = i
                    return True
            return False
        
        count = 0
        for i in range(1, N + 1):
            visited = [False] * (L + 1)
            if dfs(i, visited):
                count += 1
        return count == N
    
    # For each r, binary search or linear scan for max L
    # L ranges from 1 to 144 (max possible)
    # Actually f(12,m) is the LEAST L such that matching exists, so we want to
    # MAXIMIZE this least L. So for each r, find the smallest L with matching,
    # then pick the r that gives the largest such L.
    
    # Wait - re-reading: "the LEAST positive L such that (m,m+L] contains distinct b_1,...,b_12"
    # So f(12,m) = min{L >= 1 : matching exists in (m, m+L]}
    # We want to MAXIMIZE f(12,m) over m.
    
    # For each r, f(12, r) = smallest L with matching.
    # We want max over r of f(12, r).
    
    best_L = 0
    best_r = 0
    for r in range(PERIOD):
        # Find smallest L with matching
        # L must be at least 12 (need 12 distinct d values)
        # Try L from 12 upward
        found = False
        for L in range(12, 145):
            if time.time() - t0 > DEADLINE:
                break
            if has_matching(r, L):
                if L > best_L:
                    best_L = L
                    best_r = r
                found = True
                break
        if time.time() - t0 > DEADLINE:
            break
    
    # Now find the actual m in [0, MAX_START] with m % PERIOD == best_r
    if best_r <= MAX_START:
        m = best_r
    else:
        # best_r > MAX_START? No, best_r < PERIOD = 27720 < 1000000
        m = best_r
    
    return int(m)

# EVOLVE-BLOCK-END
