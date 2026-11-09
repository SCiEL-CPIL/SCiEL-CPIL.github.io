"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

N = 12
MAX_M = 1000000
DEADLINE_MARGIN = 2.0  # seconds to spare before the 90s kill

def _lcm(a, b):
    return a // gcd(a, b) * b

# LCM of 1..12
L = 1
for i in range(1, N + 1):
    L = _lcm(L, i)
# L = 27720

def _has_matching(m, L_win):
    """
    Check if there exist distinct b_1,...,b_12 in (m, m+L_win] such that i | b_i.
    Use bipartite matching (Hopcroft-Karp or simple DFS since n=12 is tiny).
    """
    # For each index i in 1..12, the candidate numbers are multiples of i in (m, m+L_win]
    # We need to assign distinct numbers to distinct indices.
    # n=12 is small enough for a simple DFS backtracking.
    
    # Precompute candidates for each i
    # candidates[i] = list of numbers in (m, m+L_win] divisible by i
    # To speed up, we can use a bitset or just lists.
    
    # Since L_win <= 144, the window size is small.
    # Let's just do a simple recursive matching.
    
    # For efficiency, order indices by number of candidates (most constrained first)
    # candidates for i: first multiple of i > m, then +i, +2i, ... up to m+L_win
    # Let's compute the list of candidates for each i
    
    cands = []
    for i in range(1, N + 1):
        # first multiple of i > m
        first = (m // i + 1) * i
        # last multiple of i <= m + L_win
        last = (m + L_win) // i * i
        # candidates: first, first+i, ..., last
        lst = list(range(first, last + 1, i))
        cands.append(lst)
    
    # Order by length (ascending) for better pruning
    order = sorted(range(N), key=lambda i: len(cands[i]))
    
    # Now do DFS matching
    # used: set of numbers already assigned
    # We'll assign in the order of 'order'
    
    def dfs(idx, used):
        if idx == N:
            return True
        i = order[idx]
        for num in cands[i]:
            if num not in used:
                used.add(num)
                if dfs(idx + 1, used):
                    return True
                used.remove(num)
        return False
    
    return dfs(0, set())

def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN
    
    # We need to find m in [0, MAX_M] that maximizes f(12, m)
    # f(12, m) is the least L such that matching exists in (m, m+L]
    # The divisibility pattern is periodic with period L = 27720
    # So f(12, m) depends only on m mod 27720
    
    # Strategy: 
    # 1. For each residue r in [0, 27720), compute f(12, r)
    # 2. Find the residue r* that maximizes f(12, r*)
    # 3. Return the smallest m in [0, MAX_M] with m mod 27720 == r*
    #    That would be r* itself if r* <= MAX_M, which is always true since 27720 < 1000000
    
    # But wait: f(12, m) for m > 27720 might be different from f(12, m mod 27720)?
    # Let's check: the condition is that i divides b_i, and b_i in (m, m+L]
    # If we replace m by m + k*27720, the window becomes (m+k*27720, m+k*27720+L]
    # The divisibility by i is the same (since 27720 is divisible by all i in 1..12)
    # So the relative positions of multiples are the same.
    # Thus f(12, m) = f(12, m mod 27720) for all m.
    
    # So we just need to find the best residue in [0, 27720)
    
    best_f = 0
    best_r = 0
    
    # Precompute: for each r in [0, 27720), compute f(12, r)
    # f(12, r) >= 12 (need 12 distinct numbers)
    # f(12, r) <= 144 (given in problem)
    
    # To speed up, note that for large L, matching is always possible.
    # We can binary search on L for each r.
    # But since L is at most 144, linear search is also fine.
    
    # Let's try to be smart: for each r, start with L=12 and increase until matching works.
    # Most r will have small f(12, r). The worst case might be ~30-50.
    
    # Process residues in chunks to respect deadline
    processed = 0
    for r in range(L):
        if time.time() > deadline:
            break
        # Compute f(12, r)
        found = False
        for L_win in range(12, 145):
            if _has_matching(r, L_win):
                if L_win > best_f:
                    best_f = L_win
                    best_r = r
                found = True
                break
        # If not found up to 144, use 144 (shouldn't happen per problem statement)
        processed += 1
    
    # Now find the smallest m in [0, MAX_M] with m mod 27720 == best_r
    # That's just best_r if best_r <= MAX_M
    m = best_r
    if m > MAX_M:
        # Shouldn't happen since best_r < 27720 < 1000000
        m = best_r % (MAX_M + 1)
    
    return int(m)

# EVOLVE-BLOCK-END
