"""No-isosceles set: maximize number of grid points in [0,N)^2 with no isosceles triangle."""
import time
import random
import sys

N = 64
SIZE = N * N
MAX_TIME = 300.0
DEADLINE = time.time() + MAX_TIME - 10.0  # safety margin


def _precompute_conflict_pairs():
    """For each grid point p, list of (q, r) pairs such that p,q,r form an isosceles triangle
    with p as the apex (i.e., dist(p,q)==dist(p,r), q!=r). Also includes collinear degenerate? 
    No - isosceles triangle requires 3 non-collinear points. But the standard definition in these
    problems usually counts any 3 points where two distances from a common vertex are equal.
    We'll use: three points form an isosceles triangle if any two of the three pairwise
    distances are equal AND the three points are not collinear. However, many formulations
    simply require all three pairwise distances to be distinct (which automatically excludes
    isosceles including degenerate). Let's use: no three points have two equal pairwise distances.
    This is stricter and simpler. Actually the task says "no three points forming an isosceles
    triangle". A degenerate isosceles (collinear with equal legs) - is that a triangle? Usually
    no. But to be safe and match common interpretation in these combinatorial geometry problems,
    I'll require all three pairwise squared distances to be distinct for every triple.
    This means: for each point p in set, all distances from p to other set members must be distinct.
    """
    # For each point p (index 0..SIZE-1), we need to know: for any two other points q, r,
    # if dist(p,q) == dist(p,r) then {q,r} is a "conflict pair" at apex p.
    # When we have a set S, it's valid iff for every p in S, no conflict pair at p is fully in S\{p}.
    # 
    # Data structure: for each p, a list of (q, r) where sq_dist(p,q) == sq_dist(p,r).
    # But that's huge. Instead, for each p, group points by distance from p.
    # For a group of size m, there are C(m,2) conflict pairs.
    #
    # Better approach for dynamic maintenance:
    # For each point p, maintain a dict: distance -> count of set-members at that distance from p.
    # p is "bad" if any distance has count >= 2.
    # When adding point x: for each existing point p in S, the distance d = sq_dist(p,x) 
    #   increments count[p][d]. If count[p][d] becomes 2, p becomes bad.
    # When removing point x: decrement.
    # 
    # To check if adding x is valid: for each p in S, check if count[p][sq_dist(p,x)] >= 1
    #   (meaning there's already a point at that distance from p). If any such p exists, adding x
    #   would create an isosceles with apex p.
    #
    # This is O(k) per add/remove check, which is what we want.
    
    # Precompute squared distances between all pairs? That's SIZE^2 = 16M entries, each int.
    # 16M * 4 bytes = 64MB. Acceptable.
    # Actually we can compute on the fly: sq_dist((x1,y1),(x2,y2)) = (x1-x2)^2 + (y1-y2)^2
    # Computing on the fly in Python is slow. Let's precompute a 2D array.
    
    import numpy as np
    coords = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)
    # coords[i] = (x, y) for i in 0..SIZE-1, where i = y*N + x
    # Precompute squared distance matrix? SIZE x SIZE = 4096 x 4096 = 16.7M entries.
    # Use int32: 67MB. OK.
    diff = coords[:, None, :] - coords[None, :, :]
    sqdist = (diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2).astype(np.int32)
    return coords, sqdist


def _greedy_build(sqdist, random_seed):
    """Greedy: start empty, repeatedly add a random point that doesn't conflict."""
    rng = random.Random(random_seed)
    S = set()
    # count[p][d] = number of points in S at squared distance d from p
    # For efficiency, use a dict of dicts, or a flat structure.
    # Since k will be small (~10-20), use dict: for each p in S, a Counter of distances.
    dist_counts = {}  # p -> {d: count}
    
    candidates = list(range(SIZE))
    rng.shuffle(candidates)
    
    for cand in candidates:
        # Check if adding cand is valid
        valid = True
        for p in S:
            d = sqdist[p, cand]
            if dist_counts.get(p, {}).get(d, 0) >= 1:
                valid = False
                break
        if valid:
            S.add(cand)
            dist_counts[cand] = {}
            for p in list(S):
                if p == cand:
                    continue
                d = sqdist[p, cand]
                if p not in dist_counts:
                    dist_counts[p] = {}
                dist_counts[p][d] = dist_counts[p].get(d, 0) + 1
    return S, dist_counts


def _local_search(sqdist, S_init, dist_counts_init, rng, max_iters=5000):
    """Min-conflicts style: try swaps (remove one, add another) to increase size."""
    S = set(S_init)
    dist_counts = {p: dict(d) for p, d in dist_counts_init.items()}
    best = len(S)
    
    def try_add(cand):
        """Check if cand can be added. Return True if valid."""
        for p in S:
            d = sqdist[p, cand]
            if dist_counts.get(p, {}).get(d, 0) >= 1:
                return False
        return True
    
    def do_add(cand):
        S.add(cand)
        dist_counts[cand] = {}
        for p in list(S):
            if p == cand:
                continue
            d = sqdist[p, cand]
            if p not in dist_counts:
                dist_counts[p] = {}
            dist_counts[p][d] = dist_counts[p].get(d, 0) + 1
    
    def do_remove(cand):
        S.remove(cand)
        for p in list(S):
            if p == cand:
                continue
            d = sqdist[p, cand]
            if p in dist_counts:
                dist_counts[p][d] -= 1
                if dist_counts[p][d] == 0:
                    del dist_counts[p][d]
                if not dist_counts[p]:
                    del dist_counts[p]
        del dist_counts[cand]
    
    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        # Try to add a random point
        cand = rng.randrange(SIZE)
        if cand not in S and try_add(cand):
            do_add(cand)
            if len(S) > best:
                best = len(S)
            continue
        # Try a swap: remove a random point from S, add a random point not in S
        if S:
            to_remove = rng.choice(list(S))
            do_remove(to_remove)
            cand = rng.randrange(SIZE)
            attempts = 0
            while attempts < 20:
                if cand not in S and try_add(cand):
                    do_add(cand)
                    if len(S) > best:
                        best = len(S)
                    break
                cand = rng.randrange(SIZE)
                attempts += 1
    
    return S, dist_counts, best


def construct():
    coords, sqdist = _precompute_conflict_pairs()
    
    best_S = set()
    best_dist = {}
    best_size = 0
    
    rng = random.Random(42)
    restart_count = 0
    
    while time.time() < DEADLINE:
        restart_count += 1
        # Greedy build
        S, dist_counts = _greedy_build(sqdist, rng.randint(0, 2**31))
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            best_dist = {p: dict(d) for p, d in dist_counts.items()}
        
        # Local search from this state
        S2, dist2, local_best = _local_search(sqdist, S, dist_counts, rng, max_iters=3000)
        if len(S2) > best_size:
            best_size = len(S)
            best_size = len(S2)
            best_S = set(S2)
            best_dist = {p: dict(d) for p, d in dist2.items()}
        
        # Occasionally do a "shuffled restart": take best, remove random points, rebuild
        if restart_count % 5 == 0 and best_S:
            S3 = set(best_S)
            dist3 = {p: dict(d) for p, d in best_dist.items()}
            # Remove 30% random points
            to_remove = list(S3)
            rng.shuffle(to_remove)
            n_remove = max(1, len(to_remove) // 3)
            for p in to_remove[:n_remove]:
                S3.discard(p)
                for q in list(S3):
                    d = sqdist[q, p]
                    if q in dist3:
                        dist3[q][d] -= 1
                        if dist3[q][d] <= 0:
                            dist3[q].pop(d, None)
                            if not dist3[q]:
                                del dist3[q]
                dist3.pop(p, None)
            # Greedy rebuild from this partial state
            candidates = [c for c in range(SIZE) if c not in S3]
            rng.shuffle(candidates)
            for cand in candidates:
                valid = True
                for p in S3:
                    d = sqdist[p, cand]
                    if dist3.get(p, {}).get(d, 0) >= 1:
                        valid = False
                        break
                if valid:
                    S3.add(cand)
                    dist3[cand] = {}
                    for p in list(S3):
                        if p == cand:
                            continue
                        d = sqdist[p, cand]
                        if p not in dist3:
                            dist3[p] = {}
                        dist3[p][d] = dist3[p].get(d, 0) + 1
            if len(S3) > best_size:
                best_size = len(S3)
                best_S = set(S3)
                best_dist = {p: dict(d) for p, d in dist3.items()}
    
    # Convert set of indices to (x,y) tuples
    # index = y*N + x
    result = []
    for idx in best_S:
        y = idx // N
        x = idx % N
        result.append((x, y))
    
    return result


if __name__ == "__main__":
    pts = construct()
    print(f"Found {len(pts)} points")
