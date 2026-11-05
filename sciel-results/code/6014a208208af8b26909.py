import time
import random
import itertools
import sys

N = 16
DEADLINE = 140.0

ALL_EDGES = [(i, j) for i in range(N) for j in range(i+1, N)]
NUM_EDGES = len(ALL_EDGES)
EDGE_IDX = {e: i for i, e in enumerate(ALL_EDGES)}

_PERM_MATCHINGS = [tuple(zip(range(4), p)) for p in itertools.permutations(range(4))]

_SUBSETS4 = []
_BIPARTITIONS = []
_initialized = False

def _init():
    global _initialized
    if _initialized:
        return
    _initialized = True
    all_mask = (1 << N) - 1
    for comb in itertools.combinations(range(N), 4):
        mask = 0
        for v in comb:
            mask |= (1 << v)
        _SUBSETS4.append((mask, comb))
    for comb in itertools.combinations(range(N), 8):
        s_mask = 0
        for v in comb:
            s_mask |= (1 << v)
        comp_mask = all_mask ^ s_mask
        pairs = []
        for a_mask, a_list in _SUBSETS4:
            if a_mask & ~s_mask:
                continue
            for b_mask, b_list in _SUBSETS4:
                if b_mask & ~comp_mask:
                    continue
                if a_mask & b_mask:
                    continue
                pairs.append((a_list, b_list))
        _BIPARTITIONS.append((s_mask, pairs))

def _has_q3_fast(edge_mask):
    for s_mask, pairs in _BIPARTITIONS:
        for a_list, b_list in pairs:
            missing = []
            for a in a_list:
                mm = 0
                for j, b in enumerate(b_list):
                    e = (a, b) if a < b else (b, a)
                    if not (edge_mask >> EDGE_IDX[e]) & 1:
                        mm |= (1 << j)
                missing.append(mm)
            for perm in _PERM_MATCHINGS:
                ok = True
                for r, c in perm:
                    if not (missing[r] >> c) & 1:
                        ok = False
                        break
                if ok:
                    return True
    return False

def _remove_q3s(edge_mask):
    changed = True
    while changed and time.time() < DEADLINE:
        changed = False
        for s_mask, pairs in _BIPARTITIONS:
            for a_list, b_list in pairs:
                missing = []
                for a in a_list:
                    mm = 0
                    for j, b in enumerate(b_list):
                        e = (a, b) if a < b else (b, a)
                        if not (edge_mask >> EDGE_IDX[e]) & 1:
                            mm |= (1 << j)
                    missing.append(mm)
                for perm in _PERM_MATCHINGS:
                    ok = True
                    for r, c in perm:
                        if not (missing[r] >> c) & 1:
                            ok = False
                            break
                    if ok:
                        for i in range(4):
                            for j in range(4):
                                if perm[i][0] == i and perm[i][1] == j:
                                    continue
                                ee = (a_list[i], b_list[j]) if a_list[i] < b_list[j] else (b_list[j], a_list[i])
                                idx = EDGE_IDX[ee]
                                if (edge_mask >> idx) & 1:
                                    edge_mask &= ~(1 << idx)
                                    changed = True
                                    break
                            if changed:
                                break
                        if changed:
                            break
                    if changed:
                        break
            if changed:
                break
    return edge_mask

def _local_search(seed, start_mask):
    rng = random.Random(seed)
    mask = _remove_q3s(start_mask)
    candidates = [i for i in range(NUM_EDGES) if not (mask >> i) & 1]
    rng.shuffle(candidates)
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        for ci in candidates[:]:
            if time.time() > DEADLINE:
                break
            new_mask = mask | (1 << ci)
            if not _has_q3_fast(new_mask):
                mask = new_mask
                candidates.remove(ci)
                improved = True
                break
    return mask, bin(mask).count('1')

def construct():
    _init()
    best_mask = 0
    best_count = 0
    for a in range(3):
        for b in range(3, 16):
            best_mask |= (1 << EDGE_IDX[(a, b)])
    best_count = 39
    seed = 0
    while time.time() < DEADLINE:
        rng = random.Random(seed)
        start_mask = 0
        for i in range(NUM_EDGES):
            if rng.random() < 0.7:
                start_mask |= (1 << i)
        mask, count = _local_search(seed, start_mask)
        if count > best_count:
            best_count = count
            best_mask = mask
        seed += 1
    result = []
    for i in range(NUM_EDGES):
        if (best_mask >> i) & 1:
            result.append(list(ALL_EDGES[i]))
    return result
