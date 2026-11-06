import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
_DEADLINE = 145.0

def _has_q3_check(bs):
    """Fast Q3 check using bit operations."""
    # Q3 bipartite: find A,B 4-sets with K4,4 minus 1-factor
    # Use bit operations for speed
    for A in combinations(range(16), 4):
        A_mask = 0
        for a in A:
            A_mask |= (1 << a)
        # Check degrees
        for a in A:
            if bin(bs[a] & ~A_mask).count('1') < 3:
                break
        else:
            cand = []
            for b in range(16):
                if not (A_mask >> b) & 1:
                    if bin(bs[b] & A_mask).count('1') == 3:
                        cand.append(b)
            if len(cand) >= 4:
                for B in combinations(cand, 4):
                    B_mask = 0
                    for b in B:
                        B_mask |= (1 << b)
                    if all(bin(bs[a] & B_mask).count('1') == 3 for a in A):
                        am, bm, cnt = set(), set(), 0
                        for a in A:
                            for b in B:
                                if not (bs[a] >> b) & 1:
                                    cnt += 1
                                    am.add(a)
                                    bm.add(b)
                        if cnt == 4 and len(am) == 4 and len(bm) == 4:
                            return True
    return False

def construct():
    global _DEADLINE
    start = time.time()
    best_count = 0
    best_edges = None
    
    all_edges = [(u, v) for u in range(16) for v in range(u+1, 16)]
    
    while time.time() - start < _DEADLINE - 5:
        bs = [0] * 16
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        count = 0
        for (u, v) in shuffled:
            bs[u] |= (1 << v)
            bs[v] |= (1 << u)
            if _has_q3_check(bs):
                bs[u] &= ~(1 << v)
                bs[v] &= ~(1 << u)
            else:
                count += 1
        if count > best_count:
            best_count = count
            best_edges = []
            for u in range(16):
                for v in range(u+1, 16):
                    if (bs[u] >> v) & 1:
                        best_edges.append([u, v])
    
    if best_edges is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
    return best_edges
# EVOLVE-BLOCK-END
