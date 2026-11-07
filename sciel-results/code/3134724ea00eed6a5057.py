import time, random, itertools
from collections import defaultdict

N = 128
DEADLINE = 55.0  # safety margin under 60s
START = time.time()

def _time_left():
    return DEADLINE - (time.time() - START)

def _conflicts_of(x, present, pair_conf):
    """Return set of y in present such that {x,y} forms a bad triple.
    A bad triple {a,b,c} has all three pair-sums present.
    For a fixed x, we count triples where x is one of a,b,c and the other two plus all three sums are present.
    We approximate conflict as: y such that x+y in present AND there exists z in present, z!=x,y, with x+z,y+z in present.
    That is O(|present|) per (x,y) if naive; we precompute.
    """
    # For speed we precompute sum_present as a set
    pass

def _count_triples_involving(x, present, sumset):
    """Count distinct triples {a,b,c} (all in present, containing x) for which all three pair-sums are in present."""
    cnt = 0
    seen = set()
    for y in present:
        if y == x:
            continue
        if (x + y) not in sumset:
            continue
        for z in present:
            if z <= y or z == x:
                continue
            if (x + z) in sumset and (y + z) in sumset:
                t = (min(x,y,z), x if x not in (min(x,y,z),) else None, y, z)
                key = tuple(sorted((x, y, z)))
                if key not in seen:
                    seen.add(key)
                    cnt += 1
    return cnt

def _total_triples(present, sumset):
    """Count all bad triples in present."""
    cnt = 0
    pl = list(present)
    n = len(pl)
    for i in range(n):
        for j in range(i+1, n):
            a, b = pl[i], pl[j]
            if (a + b) not in sumset:
                continue
            for k in range(j+1, n):
                c = pl[k]
                if (a + c) in sumset and (b + c) in sumset:
                    cnt += 1
    return cnt

def _greedy_build(rng):
    """Greedy: repeatedly add the candidate that creates fewest new conflicts, breaking ties randomly.
    Start from a small seed, then grow."""
    # Start with a few large numbers to anchor
    present = set()
    sumset = set()
    # initial anchors: a few large values
    anchors = list(range(N - 5, N + 1))
    rng.shuffle(anchors)
    for a in anchors:
        present.add(a)
    sumset = set(present)

    candidates = [x for x in range(1, N + 1) if x not in present]

    while _time_left() > 0.5:
        best_x = None
        best_score = None
        # evaluate adding each candidate: number of new bad triples created
        # To keep it fast, sample if candidate list is large
        cand_sample = candidates
        if len(candidates) > 200:
            cand_sample = rng.sample(candidates, 200)
        for x in cand_sample:
            # count new triples involving x
            new_cnt = 0
            for y in present:
                if (x + y) not in sumset and (x + y) != x and (x + y) != y:
                    # x+y not present, so any triple with pair (x,y) won't complete
                    # but triple needs x+y present, so skip
                    pass
                if (x + y) in sumset:
                    # need a third z with x+z and y+z present
                    for z in present:
                        if z == x or z == y:
                            continue
                        if (x + z) in sumset and (y + z) in sumset:
                            new_cnt += 1
            # normalize by something? just use raw count
            score = new_cnt
            if best_score is None or score < best_score or (score == best_score and rng.random() < 0.3):
                best_score = score
                best_x = x
        if best_x is None:
            break
        if best_score > 0 and len(present) > 80:
            # stop adding if it would create conflicts and we're already large
            # still add if it's the best (maybe 0)
            if best_score > 0:
                # try to add anyway only if it doesn't blow up; break
                break
        present.add(best_x)
        sumset.add(best_x)
        candidates.remove(best_x)

    return present

def _local_search(present, rng, max_iters=2000):
    """Local search: try to remove elements to reduce triples to 0, then try to add back."""
    present = set(present)
    sumset = set(present)

    def triples():
        return _total_triples(present, sumset)

    cur = triples()
    for it in range(max_iters):
        if _time_left() < 0.3:
            break
        if cur == 0:
            break
        # find element whose removal reduces triples the most
        best_el = None
        best_delta = 0
        # sample elements
        els = list(present)
        if len(els) > 100:
            els = rng.sample(els, 100)
        for el in els:
            present.discard(el)
            sumset.discard(el)
            newc = triples()
            delta = cur - newc
            present.add(el)
            sumset.add(el)
            if delta > best_delta:
                best_delta = delta
                best_el = el
        if best_el is None or best_delta <= 0:
            break
        present.discard(best_el)
        sumset.discard(best_el)
        cur -= best_delta

    # now try to add back elements that don't create triples
    added = True
    while added and _time_left() > 0.3:
        added = False
        for x in range(1, N + 1):
            if x in present:
                continue
            # check if adding x creates any triple
            ok = True
            for y in present:
                if (x + y) in sumset:
                    for z in present:
                        if z == x or z == y:
                            continue
                        if (x + z) in sumset and (y + z) in sumset:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                present.add(x)
                sumset.add(x)
                added = True
    return present

def construct():
    rng = random.Random(12345)
    best = set()
    best_size = 0

    # Multiple random greedy starts
    for trial in range(200):
        if _time_left() < 2.0:
            break
        p = _greedy_build(rng)
        p = _local_search(p, rng)
        if len(p) > best_size:
            best = p
            best_size = len(p)

    # Final polish on best: more aggressive local search
    if _time_left() > 1.0:
        best = _local_search(best, rng, max_iters=5000)
        # try adding back
        sumset = set(best)
        for x in range(1, N + 1):
            if x in best:
                continue
            ok = True
            for y in best:
                if (x + y) in sumset:
                    for z in best:
                        if z == x or z == y:
                            continue
                        if (x + z) in sumset and (y + z) in sumset:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                best.add(x)
                sumset.add(x)

    result = [int(x) for x in sorted(best)]
    # final validation
    s = set(result)
    for a, b, c in itertools.combinations(result, 3):
        if (a + b) in s and (a + c) in s and (b + c) in s:
            # invalid, remove the worst offender: just return as is? No, must be valid.
            # Fallback: return a known-valid set
            return list(range(N//2 + 1, N + 1))
    return result
