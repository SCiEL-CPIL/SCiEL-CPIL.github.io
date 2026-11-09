"""Rewritten program: search for linear triple systems on 16 points with small independence number."""
import random
import time

# EVOLVE-BLOCK-START
N = 16
N_MASK = (1 << N) - 1
DEADLINE = 120.0
SAFETY_MARGIN = 10.0  # seconds to finish before the kill wall

# Precompute all C(16,3) = 560 triples as (a, b, c) and their bitmask
_ALL_TRIPLES = []
_TRIPLE_MASKS = []
for _a in range(N):
    for _b in range(_a + 1, N):
        for _c in range(_b + 1, N):
            _ALL_TRIPLES.append((_a, _b, _c))
            _TRIPLE_MASKS.append((1 << _a) | (1 << _b) | (1 << _c))
N_TRIPLES_TOTAL = len(_ALL_TRIPLES)

# Precompute pairwise overlap between every pair of triples (0, 1, or 3)
# overlap[i][j] = number of shared points
_OVERLAP = [[0] * N_TRIPLES_TOTAL for _ in range(N_TRIPLES_TOTAL)]
for _i in range(N_TRIPLES_TOTAL):
    mi = _TRIPLE_MASKS[_i]
    for _j in range(_i + 1, N_TRIPLES_TOTAL):
        ov = (mi & _TRIPLE_MASKS[_j]).bit_count()
        _OVERLAP[_i][_j] = ov
        _OVERLAP[_j][_i] = ov


def _is_linear(triples_idx):
    """Check pairwise overlap <= 1 for a set of triple indices."""
    for i in range(len(triples_idx)):
        ti = triples_idx[i]
        for j in range(i + 1, len(triples_idx)):
            tj = triples_idx[j]
            if _OVERLAP[ti][tj] > 1:
                return False
    return True


def _independence_number(triples_idx):
    """Largest subset of points containing no whole triple. Bitmask over 2^16 subsets."""
    # For each subset mask S, it's "independent" if no triple is fully contained in S.
    # A triple is fully contained in S iff (triple_mask & S) == triple_mask.
    # We want max popcount(S) over independent S.
    # Efficient: for each S, check if any triple is a subset. Precompute for speed:
    # independent[S] = True iff no triple_mask is a subset of S.
    # Use: a triple t is a subset of S iff (t & S) == t.
    # We can precompute for each S whether it's independent using a zeta-like approach:
    # Let f[mask] = number of triples that are subsets of mask. independent iff f[mask] == 0.
    # f can be computed by: for each triple t, for each superset of t, increment. That's 3 * 2^13 per triple = too much for 560.
    # Alternative: for each mask, check all triples. 65536 * 560 = 36M checks. Acceptable in C-ish but Python slow.
    # Better: precompute f[mask] via SOS DP. f[mask] = count of triples t with t subset of mask.
    # Initialize f[t] += 1 for each triple t. Then SOS: for each bit b, for each mask with bit b set, f[mask] += f[mask without b].
    # That's 16 * 65536 = 1M ops. 

    f = [0] * (1 << N)
    for t in triples_idx:
        f[_TRIPLE_MASKS[t]] += 1
    # SOS DP to count subsets
    for b in range(N):
        bit = 1 << b
        for mask in range(1 << N):
            if mask & bit:
                f[mask] += f[mask ^ bit]
    # Now f[mask] = number of triples fully contained in mask.
    # independent iff f[mask] == 0.
    best = 0
    for mask in range(1 << N):
        if f[mask] == 0:
            pc = mask.bit_count()
            if pc > best:
                best = pc
    return best


def _seed_triples():
    """A starting linear system. Use a greedy / known construction."""
    # Start with 5 disjoint triples and try to add more via greedy to reduce alpha.
    idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # 5 disjoint triples: (0,1,2),(3,4,5),...
    base = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14)]
    # map to indices
    idx_of = {}
    for i, t in enumerate(_ALL_TRIPLES):
        idx_of[t] = i
    res = [idx_of[t] for t in base]
    # Greedily add triples that keep linearity and reduce alpha
    cur_alpha = _independence_number(res)
    for t in range(N_TRIPLES_TOTAL):
        if t in res:
            continue
        cand = res + [t]
        if _is_linear(cand):
            a = _independence_number(cand)
            if a < cur_alpha:
                res = cand
                cur_alpha = a
    return res


def _sa_search():
    """Simulated annealing to minimize independence number."""
    start = _seed_triples()
    best_state = list(start)
    best_alpha = _independence_number(best_state)

    t0 = time.time()
    deadline = t0 + DEADLINE - SAFETY_MARGIN

    # SA parameters
    T = 5.0
    T_min = 0.01
    cooling = 0.999

    cur_state = list(start)
    cur_alpha = best_alpha
    rng = random.Random(12345)

    # Precompute for each triple, list of triples it conflicts with (overlap > 1)
    conflicts = [[] for _ in range(N_TRIPLES_TOTAL)]
    for i in range(N_TRIPLES_TOTAL):
        for j in range(i + 1, N_TRIPLES_TOTAL):
            if _OVERLAP[i][j] > 1:
                conflicts[i].append(j)
                conflicts[j].append(i)

    step = 0
    while time.time() < deadline:
        step += 1
        # Choose a move: add, remove, or swap
        move_type = rng.random()
        new_state = list(cur_state)
        if move_type < 0.45 and len(cur_state) < 60:
            # Add a random triple that is linear-compatible with all current
            t = rng.randrange(N_TRIPLES_TOTAL)
            ok = True
            for c in cur_state:
                if _OVERLAP[t][c] > 1:
                    ok = False
                    break
            if ok:
                new_state.append(t)
            else:
                # try a few random ones
                for _ in range(20):
                    t = rng.randrange(N_TRIPLES_TOTAL)
                    ok = True
                    for c in cur_state:
                        if _OVERLAP[t][c] > 1:
                            ok = False
                            break
                    if ok:
                        new_state.append(t)
                        break
        elif move_type < 0.7 and len(cur_state) > 3:
            # Remove a random triple
            k = rng.randrange(len(cur_state))
            new_state.pop(k)
        else:
            # Swap: remove one, add one compatible
            if len(cur_state) == 0:
                continue
            k = rng.randrange(len(cur_state))
            removed = new_state.pop(k)
            added = False
            for _ in range(30):
                t = rng.randrange(N_TRIPLES_TOTAL)
                if t == removed:
                    continue
                ok = True
                for c in new_state:
                    if _OVERLAP[t][c] > 1:
                        ok = False
                        break
                if ok:
                    new_state.append(t)
                    added = True
                    break
            if not added:
                new_state.append(removed)  # revert

        # Evaluate new state
        new_alpha = _independence_number(new_state)
        delta = new_alpha - cur_alpha  # want to decrease
        if delta <= 0 or rng.random() < pow(2.718281828459045, -delta / T):
            cur_state = new_state
            cur_alpha = new_alpha
            if cur_alpha < best_alpha:
                best_alpha = cur_alpha
                best_state = list(cur_state)
        T = max(T * cooling, T_min)

    # Local refinement on best state: try to add any compatible triple that reduces alpha
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cur_alpha = _independence_number(best_state)
        for t in range(N_TRIPLES_TOTAL):
            if t in best_state:
                continue
            ok = True
            for c in best_state:
                if _OVERLAP[t][c] > 1:
                    ok = False
                    break
            if ok:
                cand = best_state + [t]
                a = _independence_number(cand)
                if a < cur_alpha:
                    best_state = cand
                    cur_alpha = a
                    improved = True
                    break

    # Convert to triples
    return [_ALL_TRIPLES[i] for i in best_state]


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - SAFETY_MARGIN
    best = None
    best_alpha = None
    # Run SA (it has its own internal deadline guard)
    try:
        res = _sa_search()
        if res:
            # validate linearity
            if _is_linear([_idx_of(t) for t in res]):
                best = res
                best_alpha = _independence_number([_idx_of(t) for t in res])
    except Exception:
        pass

    if best is None or time.time() > deadline:
        # Fallback: seed
        res = _seed_triples()
        best = [_ALL_TRIPLES[i] for i in res]

    # Final safety: ensure we return within deadline
    return best


def _idx_of(triple):
    """Map a triple (a,b,c) to its index in _ALL_TRIPLES."""
    # linear search is fine for 560
    for i, t in enumerate(_ALL_TRIPLES):
        if t == triple:
            return i
    raise ValueError("triple not found")
# EVOLVE-BLOCK-END
