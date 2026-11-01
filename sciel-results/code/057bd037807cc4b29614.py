"""Local-search / simulated-annealing covering optimizer with NumPy coverage accounting."""
import time
import random
import itertools
import numpy as np

# ---------- precomputation (per v,k,t) ----------
_TRIPLE_PRE = {}


def _pre(v, k, t):
    key = (v, k, t)
    if key in _TRIPLE_PRE:
        return _TRIPLE_PRE[key]
    pts = np.arange(v, dtype=np.int64)
    triples = np.array(list(itertools.combinations(range(v), t)), dtype=np.int64)
    nT = triples.shape[0]
    # index lookup for each triple
    idx = {}
    for i, tr in enumerate(triples):
        idx[tuple(tr)] = i
    # for each point p, the triple indices that contain p
    per_point = []
    for p in range(v):
        per_point.append(np.array(
            [i for i, tr in enumerate(triples) if p in tr], dtype=np.int64))
    # for each (p, q) with p<q, triple indices containing both
    per_pair = {}
    for p in range(v):
        for q in range(p + 1, v):
            per_pair[(p, q)] = np.array(
                [i for i, tr in enumerate(triples) if p in tr and q in tr],
                dtype=np.int64)
    # for each point p, list of (q, triple_indices containing both p and q), q != p
    pair_lists = []
    for p in range(v):
        lst = []
        for q in range(v):
            if q == p:
                continue
            a, b = (p, q) if p < q else (q, p)
            lst.append((q, per_pair[(a, b)]))
        pair_lists.append(lst)
    _TRIPLE_PRE[key] = (pts, nT, idx, per_point, pair_lists)
    return _TRIPLE_PRE[key]


def _triple_indices(block, idx):
    """Flat int array of triple indices covered by a k-subset block."""
    out = []
    for tr in itertools.combinations(block, 3):
        out.append(idx[tr])
    return np.array(out, dtype=np.int64)


def _valid(blocks, v, k, t, idx):
    # distinct, size, and coverage check
    seen = set()
    for b in blocks:
        if len(b) != k:
            return False
        s = tuple(sorted(b))
        if len(set(s)) != k:
            return False
        if s in seen:
            return False
        seen.add(s)
    counts = np.zeros(len(idx), dtype=np.int64)
    for b in blocks:
        counts[_triple_indices(b, idx)] += 1
    return bool(np.all(counts >= 1))


# ---------- greedy construction (fallback / seed) ----------
def _greedy(v, k, t, idx, per_point, pair_lists, deadline):
    uncovered = np.ones(len(idx), dtype=bool)
    blocks = []
    while uncovered.any():
        if time.time() > deadline:
            break
        # pick an uncovered triple
        uidx = np.flatnonzero(uncovered)
        ti = uidx[random.randrange(len(uidx))]
        triple = [int(x) for x in np.array(list(itertools.combinations(range(v), 3)))[ti]]
        block = list(triple)
        # fill: choose point maximizing newly covered uncovered triples
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for (q, arr) in pair_lists[x]:
                    if q in block:
                        gain += int(np.count_nonzero(uncovered[arr]))
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        block.sort()
        blocks.append(block)
        uncovered[_triple_indices(block, idx)] = False
        if time.time() > deadline:
            break
    return blocks


# ---------- local search ----------
def _local_search(v, k, t, init_blocks, idx, per_point, pair_lists, deadline, rng):
    blocks = [list(b) for b in init_blocks]
    counts = np.zeros(len(idx), dtype=np.int64)
    for b in blocks:
        counts[_triple_indices(b, idx)] += 1
    m = len(blocks)

    def redundancy():
        return int(np.sum(counts[counts > 1] - 1))

    def objective():
        return m + redundancy()

    best_blocks = [list(b) for b in blocks]
    best_obj = objective()

    # temperature schedule
    base = 0.5
    T = base
    cool = 0.999
    min_T = 1e-4

    def add_block(b):
        nonlocal m
        counts[_triple_indices(b, idx)] += 1
        m += 1

    def remove_block(b):
        nonlocal m
        counts[_triple_indices(b, idx)] -= 1
        m -= 1

    def swap_block(old, new):
        counts[_triple_indices(old, idx)] -= 1
        counts[_triple_indices(new, idx)] += 1

    while time.time() < deadline:
        T = max(min_T, T * cool)
        r = rng.random()
        if r < 0.55:
            # swap: replace a block with a random candidate
            i = rng.randrange(m)
            old = blocks[i]
            new = rng.sample(range(v), k)
            new.sort()
            if new == old:
                continue
            old_idx = _triple_indices(old, idx)
            new_idx = _triple_indices(new, idx)
            # delta in m + redundancy
            om = 0
            nm = 0
            for a in old_idx:
                c = counts[a]
                if c == 1:
                    om += 1          # becomes uncovered: +1
                elif c > 2:
                    om -= 1          # redundancy -1
            for a in new_idx:
                c = counts[a]
                if c == 1:
                    nm += 1
                elif c > 2:
                    nm -= 1
            delta = (nm - om)
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                swap_block(old, new)
                blocks[i] = new
                obj = objective()
                if obj < best_obj:
                    best_obj = obj
                    best_blocks = [list(b) for b in blocks]
        elif r < 0.8:
            # add a block (grow)
            new = sorted(rng.sample(range(v), k))
            add_block(new)
            blocks.append(new)
            obj = objective()
            if obj < best_obj:
                best_obj = obj
                best_blocks = [list(b) for b in blocks]
        else:
            # remove a block only if it does not break coverage
            i = rng.randrange(m)
            old = blocks[i]
            old_idx = _triple_indices(old, idx)
            if not np.any(counts[old_idx] == 1):
                remove_block(old)
                blocks.pop(i)
                obj = objective()
                if obj < best_obj:
                    best_obj = obj
                    best_blocks = [list(b) for b in blocks]
    return best_blocks, best_obj


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 140.0          # safety margin under 150s
    pts, nT, idx, per_point, pair_lists = _pre(v, k, t)

    # seed with greedy (fast)
    rng = random.Random(12345)
    seed_blocks = _greedy(v, k, t, idx, per_point, pair_lists, deadline)

    best_blocks = seed_blocks
    best_obj = len(seed_blocks) + _redund_of(seed_blocks, idx)

    # multiple restarts with varied seeds
    restart = 0
    while time.time() < deadline - 5.0:
        restart += 1
        rng = random.Random(1000 + 7 * restart)
        # perturb the greedy seed: randomize a few blocks
        init = [list(b) for b in seed_blocks]
        for _ in range(min(len(init), 3 + restart % 5)):
            j = rng.randrange(len(init))
            init[j] = sorted(rng.sample(range(v), k))
        cand, obj = _local_search(v, k, t, init, idx, per_point, pair_lists,
                                  deadline - 2.0, rng)
        if obj < best_obj:
            best_obj = obj
            best_blocks = cand

    # final polish: try to remove redundant blocks from the best
    cand = [list(b) for b in best_blocks]
    counts = np.zeros(nT, dtype=np.int64)
    for b in cand:
        counts[_triple_indices(b, idx)] += 1
    changed = True
    while changed and time.time() < deadline - 0.5:
        changed = False
        for i in range(len(cand)):
            oi = _triple_indices(cand[i], idx)
            if not np.any(counts[oi] == 1):
                counts[oi] -= 1
                cand.pop(i)
                changed = True
                break
    if len(cand) < len(best_blocks):
        best_blocks = cand

    # ensure validity; fall back to greedy if something went wrong
    if not _valid(best_blocks, v, k, t, idx):
        best_blocks = _greedy(v, k, t, idx, per_point, pair_lists, deadline)

    return best_blocks


def _redund_of(blocks, idx):
    counts = np.zeros(len(idx), dtype=np.int64)
    for b in blocks:
        counts[_triple_indices(b, idx)] += 1
    return int(np.sum(counts[counts > 1] - 1))
