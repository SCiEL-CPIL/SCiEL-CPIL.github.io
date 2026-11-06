"""Q3-free graph on 16 vertices."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

_Q3_EDGES = []
for a in range(8):
    for b in range(a + 1, 8):
        x = a ^ b
        if x & (x - 1) == 0:
            _Q3_EDGES.append((a, b))

def _has_q3_fast(edge_set, verts8):
    """Fast check: does induced subgraph on verts8 contain Q3?"""
    # Count edges in induced subgraph
    cnt = 0
    for i in range(8):
        for j in range(i+1, 8):
            u, v = verts8[i], verts8[j]
            if u > v: u, v = v, u
            if (u, v) in edge_set:
                cnt += 1
    if cnt < 12:
        return False
    # Try all permutations (40320) - acceptable for 12870 subsets if we skip most
    from itertools import permutations
    labels = list(verts8)
    for perm in permutations(labels):
        ok = True
        for (a, b) in _Q3_EDGES:
            u, v = perm[a], perm[b]
            if u > v: u, v = v, u
            if (u, v) not in edge_set:
                ok = False
                break
        if ok:
            return True
    return False

def _is_q3_free(edge_set):
    for subset in itertools.combinations(range(16), 8):
        if _has_q3_fast(edge_set, subset):
            return False
    return True

def construct():
    start = time.time()
    DEADLINE = 110.0
    rng = random.Random(12345)
    
    n = 16
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Seed: K3,13
    edge_set = set((a, b) for a in range(3) for b in range(3, 16))
    best = set(edge_set)
    best_cnt = len(edge_set)
    
    # Greedy: try adding random edges
    for _ in range(5000):
        if time.time() - start > DEADLINE - 20:
            break
        missing = [p for p in all_pairs if p not in edge_set]
        if not missing:
            break
        p = rng.choice(missing)
        test = edge_set | {p}
        # Quick necessary condition: check only 8-subsets containing p
        u, v = p
        found = False
        for other6 in itertools.combinations([x for x in range(16) if x != u and x != v], 6):
            s8 = tuple([u, v] + list(other6))
            if _has_q3_fast(test, s8):
                found = True
                break
        if not found:
            edge_set = test
            if len(edge_set) > best_cnt:
                best = set(edge_set)
                best_cnt = len(edge_set)
    
    # Verify best is Q3-free
    if not _is_q3_free(best):
        # Fallback to seed
        best = set((a, b) for a in range(3) for b in range(3, 16))
    
    return [[u, v] for (u, v) in sorted(best)]

# EVOLVE-BLOCK-END
