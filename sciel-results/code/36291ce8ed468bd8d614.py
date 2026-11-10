"""Rewritten program: construct largest pairwise non-coprime subset of [1,255] containing 255."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """
    Find the maximum subset S ⊆ [1, 255] containing 255 such that gcd(a,b) > 1 
    for all distinct pairs a,b in S.
    
    255 = 3*5*17. Every x in S must share a prime factor with 255.
    Two numbers are compatible iff they share at least one prime factor from {3,5,17}.
    
    Key structural insight: The multiples of 3 in [1,255] form a valid clique of size 85
    (including 255). No number outside the multiples of 3 can be added to this set,
    because any number not divisible by 3 has mask lacking bit 0, while all multiples
    of 3 have bit 0 set, and compatibility requires (mask_a & mask_b) != 0. A number
    with mask 2 (only 5) or mask 4 (only 17) has (mask & 1) = 0 with pure-3 numbers.
    Numbers with mask 6 (3&5) or mask 5 (3&17) or mask 7 (3&5&17) are already
    multiples of 3. Hence the maximum is exactly 85.
    
    We verify this via a fast branch-and-bound to be certain.
    """
    
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin before 60s limit
    
    # Generate candidates: x in [1,255] with gcd(x, 255) > 1
    C = [x for x in range(1, 256) if gcd(x, 255) > 1]
    V = [x for x in C if x != 255]
    n = len(V)
    
    # Prime factor masks: bit 0 = div by 3, bit 1 = div by 5, bit 2 = div by 17
    masks = [0] * n
    for i, x in enumerate(V):
        m = 0
        if x % 3 == 0: m |= 1
        if x % 5 == 0: m |= 2
        if x % 17 == 0: m |= 4
        masks[i] = m
    
    # Build adjacency using mask intersection (faster than gcd)
    adj = [set() for _ in range(n)]
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            if (mi & masks[j]) != 0:
                adj[i].add(j)
                adj[j].add(i)
    
    # Fast Tomita maximum clique with color-based pruning
    best_clique = []
    best_size = 0
    
    def color_sort(candidates):
        if not candidates:
            return []
        verts = list(candidates)
        verts.sort(key=lambda v: len(adj[v] & candidates), reverse=True)
        color_of = {}
        for v in verts:
            nc = set()
            for u in adj[v]:
                if u in color_of:
                    nc.add(color_of[u])
            c = 1
            while c in nc:
                c += 1
            color_of[v] = c
        sorted_verts = sorted(verts, key=lambda v: color_of[v], reverse=True)
        return [(v, color_of[v]) for v in sorted_verts]
    
    def expand(clique, candidates):
        nonlocal best_clique, best_size
        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
        if time.time() > deadline:
            return
        colored = color_sort(candidates)
        if not colored:
            return
        if len(clique) + colored[-1][1] <= best_size:
            return
        for idx, (v, c) in enumerate(colored):
            if time.time() > deadline:
                return
            if len(clique) + c <= best_size:
                break
            expand(clique + [v], candidates & adj[v])
            candidates.remove(v)
    
    expand([], set(range(n)))
    
    result_values = [V[i] for i in best_clique]
    result_values.append(255)
    
    # Verify
    s = set(result_values)
    assert 255 in s and len(s) == len(result_values)
    for a in s:
        assert 1 <= a <= 255
    for i in range(len(result_values)):
        for j in range(i + 1, len(result_values)):
            assert gcd(result_values[i], result_values[j]) > 1
    return result_values
# EVOLVE-BLOCK-END
