"""Fresh design: maximize distinct 3-term progression differences for a 32-subset of 1..512."""
import random
import time
import math
from collections import deque

N = 512
K = 32

# Precompute d_shift[d] = (1 << d) - 1  (bitmask of positions 0..d-1)
_d_shift = [(1 << d) - 1 for d in range(0, N + 1)]


def _count_d(bs):
    """Return number of distinct d>0 with a 3-AP in bitset bs (positions 0..N-1)."""
    c = 0
    for d in range(1, N + 1):
        if (bs & (bs >> d) & (bs >> (2 * d))) != 0:
            c += 1
    return c


def _make_delta(bs):
    """delta[d] = signed change in score from removing bit a and adding bit b.
    delta[d][a][b] = new_count - old_count for the pair (a,b)."""
    D = []
    for d in range(1, N + 1):
        sh = _d_shift[d]
        row = []
        for a in range(N):
            ra = []
            base_a = (bs >> a) & 1
            for b in range(N):
                if a == b:
                    ra.append(0)
                    continue
                # d contributes iff some triple x,x+d,x+2d has all three in the set.
                # triple exists iff ( (bs>>x) & (bs>>(x+d)) & (bs>>(x+2d)) ) for some x.
                # Equivalent: (bs & (bs>>d) & (bs>>(2d))) >> x has a 1.
                # Instead, compute membership of the three positions in the *candidate* set.
                # Candidate set = bs with a cleared and b set.
                # For each x, triple present iff all of x, x+d, x+2d in candidate.
                # We precompute which x have the triple in the ORIGINAL set:
                #   orig[x] = 1 iff x,x+d,x+2d in bs
                # and which x would be in candidate:
                #   cand[x] = 1 iff x,x+d,x+2d in (bs - a + b)
                # diff = popcount(cand) - popcount(orig)
                # This is per (a,b); we compute via bit tricks.
                # orig_mask = bs & (bs>>d) & (bs>>(2d))  (already independent of a,b)
                # cand_mask = (bs ^ (1<<a) ^ (1<<b)) & ... but a==b handled; a!=b so XOR both.
                # We need a fast per-(a,b) popcount of cand_mask.
                # cand_mask = ((bs ^ (1<<a) ^ (1<<b)) & (bs ^ (1<<a) ^ (1<<b)) >> d) & ...
                # That's expensive. Instead compute contribution change analytically:
                # A triple x is present in candidate iff for all k in {0,1,2}: (x+k*d) in candidate.
                # candidate contains position p iff (p==b) or (p in bs and p!=a).
                # So triple x present in candidate iff:
                #   for each k: (x+k*d == b) or (x+k*d in bs and x+k*d != a)
                # = [not (any k with x+k*d == a)] and [for all k: (x+k*d == b) or (x+k*d in bs)]
                # Define:
                #   hit_a = 1 if any k: x+k*d == a
                #   need_b = 1 if any k: x+k*d == b and that position not in bs
                #   has_b = 1 if any k: x+k*d == b
                # triple in candidate = (not hit_a) and (for all k: (x+k*d in bs) or (x+k*d==b))
                # = (not hit_a) and [ (for all k: x+k*d in bs) or (all the k not in bs are == b) ]
                # The "all k not in bs are == b" means: the set of k with x+k*d not in bs is subset of {k: x+k*d==b}.
                # Since b is a single position, at most one k can equal b. So we need: the number of k with x+k*d not in bs is <=1, and if ==1 then that position is b.
                # Let miss = {k : x+k*d not in bs}. Then triple in candidate iff (not hit_a) and (len(miss)<=1 and (len(miss)==0 or (the single miss position == b))).
                # And triple in original iff (for all k: x+k*d in bs) i.e. len(miss)==0.
                # So:
                #   orig_present = (len(miss)==0)
                #   cand_present = (not hit_a) and (len(miss)==0 or (len(miss)==1 and miss_pos==b))
                # diff contribution for this x = cand_present - orig_present.
                # Sum over x in [0, N-2d].
                # We can precompute for each x: len(miss) and miss_pos and hit_a depends on a.
                # For fixed d, precompute for each x:
                #   miss_count[x], miss_pos[x] (position, or -1)
                #   a_hit[x] = 1 if a in {x, x+d, x+2d}  -> depends on a; but a_hit[x]=1 iff a==x or a==x+d or a==x+2d.
                # For given a, hit_a for x is 1 iff x in {a, a-d, a-2d} (and within range).
                # So for fixed a, the set of x with hit_a=1 is {a, a-d, a-2d} intersect [0,N-2d].
                # Then cand_present for x = (not hit_a) and (miss_count[x]==0 or (miss_count[x]==1 and miss_pos[x]==b)).
                # diff(x) = cand_present - (miss_count[x]==0).
                # We sum over x. For fixed (a,b), we can compute quickly using precomputed arrays.
                # Let's just compute directly but vectorized-ish per x is O(N) -> too slow for 512*512*256.
                # So we need a smarter O(1) per (a,b).
                #
                # Alternative: precompute for each d and each x: state = (miss_count, miss_pos).
                # For fixed a, define H_a = {x in [0,N-2d] : a in {x,x+d,x+2d}} = {a, a-d, a-2d} clipped.
                # For x not in H_a: cand_present = (miss_count[x]==0) or (miss_count[x]==1 and miss_pos[x]==b).
                #   orig_present = (miss_count[x]==0).
                #   diff = (miss_count[x]==1 and miss_pos[x]==b)   [since if miss_count==0 both 1; if miss_count==1: cand= (miss_pos==b), orig=0 -> diff=(miss_pos==b); if miss_count==2: cand=0,orig=0 ->0]
                #   Wait if miss_count==1 and miss_pos==b: cand_present=1, orig=0 -> diff=1.
                #   if miss_count==1 and miss_pos!=b: cand=0,orig=0 ->0.
                #   if miss_count==0: cand=1,orig=1 ->0.
                #   if miss_count==2: cand=0,orig=0 ->0.
                # So for x not in H_a: diff = 1 iff (miss_count[x]==1 and miss_pos[x]==b) else 0.
                # For x in H_a: hit_a=1 -> cand_present=0 always. diff = 0 - (miss_count[x]==0) = -1 if miss_count[x]==0 else 0.
                # So total diff = [sum over x not in H_a of (miss_count[x]==1 and miss_pos[x]==b)] - [number of x in H_a with miss_count[x]==0].
                # Precompute for each d:
                #   For each b: count of x with miss_count[x]==1 and miss_pos[x]==b  -> call C1[b]. (x over [0,N-2d])
                #   For each a: number of x in H_a with miss_count[x]==0 -> call ZH[a].
                # Then diff(a,b) = C1[b] - ZH[a]  ... but wait, the x in H_a are excluded from the first sum. C1[b] counts ALL x with miss_count==1 and miss_pos==b, including those in H_a. But for x in H_a, diff is computed by the second term, not the first. So we must subtract from C1[b] the count of x in H_a with (miss_count==1 and miss_pos==b).
                # So diff(a,b) = (C1[b] - count_{x in H_a, miss_count==1, miss_pos==b}) - ZH[a].
                # Precompute for each a: for x in H_a, whether (miss_count==1 and miss_pos==b) for each b -> that's at most 3 x's, so for each a, the set of b where a "blocks" a C1 contribution is at most 3 values (the miss_pos of the x in H_a with miss_count==1).
                # So diff(a,b) = C1[b] - ZH[a] - block[a][b], where block[a][b] = number of x in H_a with miss_count[x]==1 and miss_pos[x]==b (0 or 1, at most).
                # This is O(1) per (a,b) with precomputation O(N^2) per d (to build C1 and ZH and block). Total O(N^3) ~ 512^3 = 1.3e8, too slow in Python.
                #
                # Given complexity, fall back to a simpler but still fast approach:
                # For local search, instead of precomputing delta for all (a,b), compute the score of the
                # candidate set directly using _count_d (fast: ~256 big-int ops) and only evaluate a
                # SMALL random subset of the neighborhood each step. This is simpler and fast enough.
                ra.append(0)  # placeholder; we won't use this table
            row.append(ra)
        D.append(row)
    return D


def _score_set(s):
    bs = 0
    for x in s:
        bs |= 1 << (x - 1)
    return _count_d(bs)


def _bitset(s):
    bs = 0
    for x in s:
        bs |= 1 << (x - 1)
    return bs


def _greedy_build(rng):
    """Greedy: add the element that maximizes distinct d, starting from a few random seeds."""
    chosen = set()
    # start with a few random
    for _ in range(6):
        chosen.add(rng.randint(1, N))
    while len(chosen) < K:
        bs = _bitset(chosen)
        best = None
        best_score = -1
        # candidate: try all not-chosen, but to be fast, sample a subset if large
        cands = [x for x in range(1, N + 1) if x not in chosen]
        # evaluate each candidate's score (fast)
        for x in cands:
            bs2 = bs | (1 << (x - 1))
            sc = _count_d(bs2)
            if sc > best_score:
                best_score = sc
                best = x
        chosen.add(best)
    return list(chosen)


def _ap_family(rng):
    """Build a set that is a union of a few arithmetic progressions (rich in 3-APs)."""
    chosen = set()
    while len(chosen) < K:
        a = rng.randint(1, N)
        d = rng.randint(1, 200)
        for t in range(0, K):
            v = a + t * d
            if 1 <= v <= N:
                chosen.add(v)
            if len(chosen) >= K:
                break
    return list(chosen)


def _random_build(rng):
    return rng.sample(range(1, N + 1), K)


def _hill_climb(s, rng, time_limit, tabu_len=20):
    """Hill-climb with tabu. Returns best set found."""
    s = set(s)
    best = list(s)
    best_sc = _score_set(best)
    bs = _bitset(best)
    tabu = deque(maxlen=tabu_len)
    last_improve = time.time()
    start = time.time()
    stall = 0
    while time.time() < time_limit:
        # pick a random element to remove
        a = rng.choice(list(s))
        # try adding a random b not in s
        improved = False
        # sample several b
        for _ in range(40):
            b = rng.randint(1, N)
            if b in s or b == a:
                continue
            new_s = s - {a} | {b}
            sc = _score_set(new_s)
            if sc > best_sc or (sc == best_sc and rng.random() < 0.1):
                # accept
                s = new_s
                best = list(s)
                best_sc = sc
                bs = _bitset(best)
                tabu.append((a, b))
                last_improve = time.time()
                stall = 0
                improved = True
                break
            else:
                tabu.append((a, b))
        if not improved:
            stall += 1
            if stall > 200:
                break
    return best, best_sc


def _sim_anneal(s, rng, time_limit):
    s = set(s)
    best = list(s)
    best_sc = _score_set(best)
    T = 2.0
    start = time.time()
    stall = 0
    while time.time() < time_limit:
        a = rng.choice(list(s))
        b = rng.randint(1, N)
        if b == a:
            continue
        if b in s:
            continue
        new_s = s - {a} | {b}
        sc = _score_set(new_s)
        delta = sc - best_sc
        if delta >= 0 or rng.random() < math.exp(delta / max(T, 1e-6)):
            s = new_s
            if sc > best_sc:
                best = list(s)
                best_sc = sc
                stall = 0
            else:
                stall += 1
        else:
            stall += 1
        T *= 0.9999
        if T < 0.01:
            T = 0.01
        if stall > 300:
            break
    return best, best_sc


def construct():
    start = time.time()
    DEADLINE = 135.0  # safety margin under 150s
    rng = random.Random(12345)

    # Seed families
    seeds = []
    # greedy builds
    for _ in range(4):
        seeds.append(_greedy_build(rng))
    # AP families
    for _ in range(4):
        seeds.append(_ap_family(rng))
    # random
    for _ in range(4):
        seeds.append(_random_build(rng))
    # structured: consecutive blocks
    seeds.append(list(range(1, K + 1)))
    seeds.append(list(range(1, N + 1, N // K)))

    # Evaluate seeds, keep best
    best = None
    best_sc = -1
    for sd in seeds:
        sc = _score_set(sd)
        if sc > best_sc:
            best_sc = sc
            best = sd

    # Multi-start local search with hill-climb and SA
    n_starts = 14
    per_start = (DEADLINE - (time.time() - start)) / n_starts
    for i in range(n_starts):
        if time.time() > DEADLINE:
            break
        # choose a seed (rotate)
        sd = seeds[i % len(seeds)]
        t_limit = min(time.time() + per_start, DEADLINE - 2.0)
        if i % 2 == 0:
            cand, sc = _hill_climb(sd, rng, t_limit)
        else:
            cand, sc = _sim_anneal(sd, rng, t_limit)
        if sc > best_sc:
            best_sc = sc
            best = cand

    # Final polish pass (hill-climb on best with remaining time)
    if time.time() < DEADLINE - 3.0:
        cand, sc = _hill_climb(best, rng, DEADLINE - 3.0)
        if sc > best_sc:
            best_sc = sc
            best = cand

    # Ensure exactly 32 distinct in 1..512
    best = sorted(set(best))
    if len(best) != K:
        # fix
        if len(best) > K:
            best = best[:K]
        else:
            # add random
            cur = set(best)
            while len(cur) < K:
                cur.add(rng.randint(1, N))
            best = sorted(cur)
    return list(best)
