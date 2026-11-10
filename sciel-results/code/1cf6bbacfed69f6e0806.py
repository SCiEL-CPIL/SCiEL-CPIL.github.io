"""Linear triple systems on 16 points with small independence number."""
import random
import time
import itertools

N = 16
FULL = (1 << N) - 1

def _mask(t):
    return (1 << t[0]) | (1 << t[1]) | (1 << t[2])

def _is_linear(triples):
    for i in range(len(triples)):
        mi = _mask(triples[i])
        for j in range(i + 1, len(triples)):
            if (mi & _mask(triples[j])).bit_count() > 1:
                return False
    return True

def _independence_number(triples):
    """Exact alpha: largest point set containing no triple.

    Precompute a 'bad' array over all 2^16 subsets by marking every superset
    of each triple mask as non-independent, then take the max popcount of a
    non-bad subset. 2^16 = 65536 is cheap.
    """
    if not triples:
        return N
    bad = [False] * (1 << N)
    for t in triples:
        m = _mask(t)
        free = FULL ^ m
        s = free
        while True:
            bad[m | s] = True
            if s == 0:
                break
            s = (s - 1) & free
    best = 0
    for S in range(1 << N):
        if not bad[S]:
            pc = S.bit_count()
            if pc > best:
                best = pc
    return best

def _greedy_independent_set(triples, rng):
    """A large (not necessarily maximum) independent set via greedy add."""
    masks = [_mask(t) for t in triples]
    avail = list(range(N))
    rng.shuffle(avail)
    chosen = []
    chosen_mask = 0
    for p in avail:
        pm = 1 << p
        new_mask = chosen_mask | pm
        ok = True
        for m in masks:
            if (m & new_mask) == m:
                ok = False
                break
        if ok:
            chosen.append(p)
            chosen_mask = new_mask
    return chosen

def _random_greedy_packing(rng):
    """Build a dense linear triple system by greedily adding random triples.

    Linearity for triples <=> no pair is used by two triples, so we track
    used pairs.
    """
    all_triples = list(itertools.combinations(range(N), 3))
    rng.shuffle(all_triples)
    used_pairs = set()
    triples = []
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        if any(p in used_pairs for p in pairs):
            continue
        for p in pairs:
            used_pairs.add(p)
        triples.append(list(t))
    return triples

def _structured_seed():
    """Cyclic Steiner triple system on 15 points + one point.

    v=15, k=3, lambda=1 => 35 triples, a linear triple system (a Steiner
    triple system). Point 15 is left mostly unused; search will place it.
    """
    triples = []
    for i in range(15):
        for d in (1, 2, 4, 8):
            a = i % 15
            b = (i + d) % 15
            c = (i + 2 * d) % 15
            t = tuple(sorted((a, b, c)))
            if t not in triples:
                triples.append(list(t))
    return triples

def _add_breaking_iset(triples, iset, rng):
    """Try to add a triple that is a subset of iset (breaks it) and is linear.

    Returns True if a triple was added.
    """
    iset_list = list(iset)
    if len(iset_list) < 3:
        return False
    candidates = list(itertools.combinations(iset_list, 3))
    rng.shuffle(candidates)
    for nt in candidates:
        nm = _mask(nt)
        ok = True
        for t in triples:
            if (nm & _mask(t)).bit_count() > 1:
                ok = False
                break
        if ok:
            triples.append(list(nt))
            return True
    return False

def _targeted_search(triples, best_alpha, deadline, rng):
    """Repeatedly find a large independent set and add a triple to break it."""
    current = [list(t) for t in triples]
    cur_alpha = _independence_number(current)
    best = [list(t) for t in current]
    best_a = cur_alpha
    while time.time() < deadline:
        if cur_alpha <= best_a:
            # refresh a fresh iset search
            pass
        iset = _greedy_independent_set(current, rng)
        if len(iset) <= best_a:
            # try a couple more random draws before giving up on this config
            improved = False
            for _ in range(3):
                iset2 = _greedy_independent_set(current, rng)
                if len(iset2) > best_a:
                    if _add_breaking_iset(current, iset2, rng):
                        na = _independence_number(current)
                        if na < best_a:
                            best_a = na
                            best = [list(t) for t in current]
                        cur_alpha = na
                        improved = True
                        break
            if not improved:
                break
            continue
        if _add_breaking_iset(current, iset, rng):
            na = _independence_number(current)
            if na < best_a:
                best_a = na
                best = [list(t) for t in current]
            cur_alpha = na
        else:
            # cannot break via add; try remove-then-add
            if len(current) > 6:
                idx = rng.randrange(len(current))
                removed = current.pop(idx)
                if _add_breaking_iset(current, iset, rng):
                    na = _independence_number(current)
                    if na < best_a:
                        best_a = na
                        best = [list(t) for t in current]
                    cur_alpha = na
                else:
                    current.append(removed)
                    break
            else:
                break
    return best, best_a

def _local_search(triples, best_alpha, deadline, rng):
    """Random triple replacement / addition to reduce alpha."""
    current = [list(t) for t in triples]
    cur_alpha = _independence_number(current)
    best = [list(t) for t in current]
    best_a = cur_alpha
    while time.time() < deadline:
        move = rng.random()
        if move < 0.7 and current:
            idx = rng.randrange(len(current))
            old = current[idx]
            for _ in range(15):
                nt = tuple(sorted(rng.sample(range(N), 3)))
                nm = _mask(nt)
                ok = True
                for j, t in enumerate(current):
                    if j == idx:
                        continue
                    if (nm & _mask(t)).bit_count() > 1:
                        ok = False
                        break
                if not ok:
                    continue
                current[idx] = list(nt)
                na = _independence_number(current)
                if na < cur_alpha:
                    cur_alpha = na
                    if na < best_a:
                        best_a = na
                        best = [list(t) for t in current]
                    break
                else:
                    current[idx] = old
                    break
        else:
            # add a random triple
            nt = tuple(sorted(rng.sample(range(N), 3)))
            nm = _mask(nt)
            ok = True
            for t in current:
                if (nm & _mask(t)).bit_count() > 1:
                    ok = False
                    break
            if ok:
                current.append(list(nt))
                na = _independence_number(current)
                if na < cur_alpha:
                    cur_alpha = na
                    if na < best_a:
                        best_a = na
                        best = [list(t) for t in current]
                else:
                    current.pop()
    return best, best_a

def construct():
    start = time.time()
    deadline = start + 150.0 - 8.0  # safety margin before the 150s kill
    rng = random.Random(20240611)

    best_triples = None
    best_alpha = N + 1

    def consider(triples):
        nonlocal best_triples, best_alpha
        if not _is_linear(triples):
            return
        a = _independence_number(triples)
        if a < best_alpha:
            best_alpha = a
            best_triples = [list(t) for t in triples]

    # Phase 0: strong structured seed
    seed = _structured_seed()
    consider(seed)

    # Phase 1: several random greedy packings, each followed by targeted search
    for k in range(6):
        if time.time() > deadline - 20:
            break
        triples = _random_greedy_packing(rng)
        consider(triples)
        sub_deadline = min(deadline - 10, time.time() + 12)
        triples, _ = _targeted_search(triples, best_alpha, sub_deadline, rng)
        consider(triples)

    # Phase 2: targeted search on the best system
    if best_triples is not None and time.time() < deadline:
        sub_deadline = min(deadline - 5, time.time() + 30)
        best_triples, best_alpha = _targeted_search(best_triples, best_alpha, sub_deadline, rng)
        consider(best_triples)

    # Phase 3: local search (replace/add) on the best system
    if best_triples is not None and time.time() < deadline:
        sub_deadline = deadline - 3
        best_triples, best_alpha = _local_search(best_triples, best_alpha, sub_deadline, rng)
        consider(best_triples)

    # Final validation + output
    if best_triples is None:
        best_triples = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    if not _is_linear(best_triples):
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    return [list(t) for t in best_triples]
