import itertools
import random
import time
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
M = N * N  # 256

def _sq_dist(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def _build_cover():
    xs = np.array([p[0] for p in POINTS], dtype=np.int64)
    ys = np.array([p[1] for p in POINTS], dtype=np.int64)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    D2 = dx * dx + dy * dy
    A = np.arange(M)
    cover = []
    for p in range(M):
        d_pa = D2[p, :]  # (M,)
        d_pb = D2[:, p]  # (M,)
        # For each pair (a, b), check if p, a, b is isosceles
        # d_pa[a] == d_pa[b] OR d_pa[a] == d_pb[a] OR d_pa[b] == d_pb[a]
        # i.e., d_pa[a] == d_pa[b] OR d_pa[a] == d_ab OR d_pa[b] == d_ab
        # where d_ab = D2[a, b]
        pairs = []
        for a in range(M):
            if a == p:
                continue
            for b in range(a + 1, M):
                if b == p:
                    continue
                dap = d_pa[a]
                dbp = d_pa[b]
                dab = D2[a, b]
                if dap == dbp or dap == dab or dbp == dab:
                    pairs.append((a, b))
        cover.append(pairs)
    return cover

# Precompute cover at module level
COVER = _build_cover()

def _make_inS(S_set):
    inS = np.zeros(M, dtype=bool)
    for p in S_set:
        inS[p] = True
    return inS

def _compatible(p, inS):
    """Check if point p can be added to S (represented by inS) without creating isosceles."""
    # p is compatible with S iff no pair (a,b) in COVER[p] has both a and b in S
    for a, b in COVER[p]:
        if inS[a] and inS[b]:
            return False
    return True

def _is_maximal(S_set, inS):
    """Check if S is maximal: every point outside S is blocked."""
    for p in range(M):
        if inS[p]:
            continue
        # p is blocked iff some pair (a,b) in COVER[p] has both in S
        blocked = False
        for a, b in COVER[p]:
            if inS[a] and inS[b]:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _cover_counts(S_set, inS):
    """For each point p not in S, count how many pairs in S block p."""
    counts = np.zeros(M, dtype=np.int32)
    S_list = list(S_set)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            a = S_list[i]
            b = S_list[j]
            # Which points p are blocked by pair (a,b)?
            # p is blocked by (a,b) iff (a,b) in COVER[p], i.e., p,a,b is isosceles
            # We need an inverse: for pair (a,b), which p have (a,b) in COVER[p]?
            # This is equivalent to: p such that p,a,b is isosceles.
            # We can compute this on the fly or precompute.
            # For now, iterate over all p not in S and check.
            pass
    # This is O(|S|^2 * M * avg_cover). Let's do it differently.
    # For each p not in S, count blocking pairs.
    for p in range(M):
        if inS[p]:
            continue
        cnt = 0
        for a, b in COVER[p]:
            if inS[a] and inS[b]:
                cnt += 1
        counts[p] = cnt
    return counts

def _shrink(S_set, inS):
    """Try to remove points from S while maintaining maximality and no-isosceles."""
    S_list = list(S_set)
    inS = inS.copy()
    changed = True
    while changed:
        changed = False
        for p in S_list[:]:
            if p not in S_set:
                continue
            # Try removing p
            inS[p] = False
            # Check if still maximal
            if _is_maximal(S_set, inS):
                S_set.remove(p)
                S_list.remove(p)
                changed = True
            else:
                inS[p] = True
    return S_set, inS

def _repair(S_set, inS, deadline):
    """Repair S if it's not maximal by adding points."""
    S_list = list(S_set)
    while True:
        if time.time() > deadline:
            break
        # Find a point that's not blocked
        added = False
        for p in range(M):
            if inS[p]:
                continue
            # Check if p is blocked
            blocked = False
            for a, b in COVER[p]:
                if inS[a] and inS[b]:
                    blocked = True
                    break
            if not blocked:
                # Add p
                # Check compatibility
                if _compatible(p, inS):
                    inS[p] = True
                    S_set.add(p)
                    S_list.append(p)
                    added = True
                    break
                else:
                    # Can't add p due to isosceles, but it's not blocked.
                    # This means S is not maximal and we can't fix by adding.
                    # We need a different strategy.
                    pass
        if not added:
            break
    return S_set, inS

def _greedy_maximal(deadline):
    """Build a maximal no-isosceles set using greedy."""
    order = list(range(M))
    random.shuffle(order)
    S_set = set()
    inS = np.zeros(M, dtype=bool)
    for p in order:
        if time.time() > deadline:
            break
        if _compatible(p, inS):
            S_set.add(p)
            inS[p] = True
    # Make it maximal
    S_set, inS = _repair(S_set, inS, deadline)
    return S_set, inS

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    best_size = float('inf')
    best_set = []
    
    random.seed(42)
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # 1. Greedy Init
        S_set, inS = _greedy_maximal(deadline)
        
        # 2. Shrink
        S_set, inS = _shrink(S_set, inS)
        
        # 3. Perturbation: remove k random points, then repair
        k = random.randint(1, 5)
        if len(S_set) > k:
            to_remove = random.sample(list(S_set), k)
            for p in to_remove:
                S_set.remove(p)
                inS[p] = False
            S_set, inS = _repair(S_set, inS, deadline)
            if S_set:
                S_set, inS = _shrink(S_set, inS)
        
        # Update best
        if S_set and len(S_set) < best_size:
            best_size = len(S_set)
            best_set = list(S_set)
    
    return [POINTS[i] for i in best_set]
