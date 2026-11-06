"""Hypergraph Ramsey R(4,5;3): WALKSAT-style CSP solver over a COMPACT per-vertex
state. Each vertex keeps a 12-bit mask of which of its 12 incident triples (to
later vertices) are red. A red K4 = a 4-clique fully red in the derived red graph;
a blue K5 = a 5-set with no red triple. Search = randomized backtracking +
conflict-driven repair + restarts (a different family from the current flat
triple-flip local search / SA)."""
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0


def _t_idx(a, b, c, n):
    cnt = 0
    for x in range(a):
        cnt += (n - x - 1) * (n - x - 2) // 2
    for y in range(a + 1, b):
        cnt += (n - y - 1)
    cnt += (c - b - 1)
    return cnt


def _build_data(n):
    triples = list(combinations(range(n), 3))
    m = len(triples)

    quads = []
    for q in combinations(range(n), 4):
        a, b, c, d = q
        idxs = [_t_idx(a, b, c, n), _t_idx(a, b, d, n),
                _t_idx(a, c, d, n), _t_idx(b, c, d, n)]
        quads.append(idxs)
    n_quads = len(quads)

    pents = []
    for p in combinations(range(n), 5):
        idxs = []
        for t in combinations(p, 3):
            idxs.append(_t_idx(t[0], t[1], t[2], n))
        pents.append(idxs)
    n_pents = len(pents)

    tri_quads = [[] for _ in range(m)]
    for qi, idxs in enumerate(quads):
        for i in idxs:
            tri_quads[i].append(qi)

    tri_pents = [[] for _ in range(m)]
    for pi, idxs in enumerate(pents):
        for i in idxs:
            tri_pents[i].append(pi)

    return triples, m, quads, pents, tri_quads, tri_pents


def _to_colors(masks, n):
    colors = [0] * (n * (n - 1) * (n - 2) // 6)
    for a in range(n):
        ma = masks[a]
        la = n - a - 1
        bpos = 0
        for b in range(a + 1, n):
            mb = masks[b]
            lb = n - b - 1
            cb = 0
            for c in range(b + 1, n):
                bit = (ma >> (bpos + cb)) & 1
                colors[_t_idx(a, b, c, n)] = bit
                cb += 1
            bpos += lb
    return colors


def _count_violations(masks, n, quads, pents):
    # red K4: 4-set with all 4 triples red
    red_k4 = 0
    for a in range(n - 3):
        ma = masks[a]
        la = n - a - 1
        bpos = 0
        for b in range(a + 1, n - 2):
            mb = masks[b]
            lb = n - b - 1
            bc_bit = (ma >> bpos) & 1
            cb = 0
            for c in range(b + 1, n - 1):
                mc = masks[c]
                lc = n - c - 1
                ac_bit = (ma >> (bpos + cb)) & 1
                bc_bit2 = (mb >> cb) & 1
                cb2 = 0
                for d in range(c + 1, n):
                    ad_bit = (ma >> (bpos + cb + cb2)) & 1
                    cd_bit = (mc >> cb2) & 1
                    if bc_bit and ac_bit and ad_bit and cd_bit:
                        red_k4 += 1
                    cb2 += 1
                cb += 1
            bpos += lb
    # blue K5: 5-set with no red triple
    blue_k5 = 0
    for a in range(n - 4):
        ma = masks[a]
        la = n - a - 1
        bpos = 0
        for b in range(a + 1, n - 3):
            mb = masks[b]
            lb = n - b - 1
            ab = (ma >> bpos) & 1
            cb = 0
            for c in range(b + 1, n - 2):
                mc = masks[c]
                lc = n - c - 1
                ac = (ma >> (bpos + cb)) & 1
                bc = (mb >> cb) & 1
                cb2 = 0
                for d in range(c + 1, n - 1):
                    md = masks[d]
                    ld = n - d - 1
                    ad = (ma >> (bpos + cb + cb2)) & 1
                    bd = (mb >> (cb + cb2)) & 1
                    cd = (mc >> cb2) & 1
                    if (ab | ac | bc | ad | bd | cd) == 0:
                        # need e with all blue to a,b,c,d
                        cb3 = 0
                        for e in range(d + 1, n):
                            ae = (ma >> (bpos + cb + cb2 + cb3)) & 1
                            be = (mb >> (cb + cb2 + cb3)) & 1
                            ce = (mc >> (cb2 + cb3)) & 1
                            de = (md >> cb3) & 1
                            if (ae | be | ce | de) == 0:
                                blue_k5 += 1
                            cb3 += 1
                    cb2 += 1
                cb += 1
            bpos += lb
    return red_k4, blue_k5


def _random_masks(n, p_red, rng):
    masks = [0] * n
    for a in range(n - 2):
        la = n - a - 1
        bits = 0
        bpos = 0
        for b in range(a + 1, n):
            lb = n - b - 1
            for c in range(b + 1, n):
                if rng.random() < p_red:
                    bits |= (1 << (bpos + (c - b - 1)))
            bpos += lb
        masks[a] = bits
    return masks


def _walksat(n, quads, pents, time_budget, seed):
    rng = random.Random(seed)
    t_end = time.time() + time_budget
    best_energy = None
    best_masks = None
    restart = 0
    while time.time() < t_end:
        p_red = 0.22 + 0.18 * rng.random()
        masks = _random_masks(n, p_red, rng)
        red_k4, blue_k5 = _count_violations(masks, n, quads, pents)
        energy = red_k4 + blue_k5
        if energy == 0:
            return _to_colors(masks, n)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_masks = masks[:]
        steps = 0
        MAX_STEPS = 4000
        while steps < MAX_STEPS and time.time() < t_end:
            if red_k4 == 0 and blue_k5 == 0:
                return _to_colors(masks, n)
            steps += 1
            # pick a violating structure
            if red_k4 > 0 and (blue_k5 == 0 or rng.random() < red_k4 / (red_k4 + blue_k5)):
                # find a red K4
                a = rng.randrange(n - 3)
                ma = masks[a]
                la = n - a - 1
                bpos = 0
                found = None
                for b in range(a + 1, n - 2):
                    mb = masks[b]
                    lb = n - b - 1
                    if not ((ma >> bpos) & 1):
                        bpos += lb
                        continue
                    cb = 0
                    for c in range(b + 1, n - 1):
                        mc = masks[c]
                        lc = n - c - 1
                        if not ((ma >> (bpos + cb)) & 1) or not ((mb >> cb) & 1):
                            cb += 1
                            continue
                        cb2 = 0
                        for d in range(c + 1, n):
                            if ((ma >> (bpos + cb + cb2)) & 1) and ((mc >> cb2) & 1):
                                found = (a, b, c, d)
                                break
                            cb2 += 1
                        if found:
                            break
                        cb += 1
                    if found:
                        break
                    bpos += lb
                if found is None:
                    # fallback: random flip
                    a = rng.randrange(n - 2)
                    b = rng.randrange(a + 2, n)
                    c = rng.randrange(b + 1, n)
                    pos = (b - a - 1) + (c - b - 1)
                    masks[a] ^= (1 << pos)
                else:
                    a, b, c, d = found
                    # flip one of the 4 red triples to blue
                    t = rng.randrange(4)
                    if t == 0:
                        x, y, z = a, b, c
                    elif t == 1:
                        x, y, z = a, b, d
                    elif t == 2:
                        x, y, z = a, c, d
                    else:
                        x, y, z = b, c, d
                    pos = (y - x - 1) + (z - y - 1)
                    masks[x] ^= (1 << pos)
            else:
                # find a blue K5
                a = rng.randrange(n - 4)
                ma = masks[a]
                la = n - a - 1
                bpos = 0
                found = None
                for b in range(a + 1, n - 3):
                    mb = masks[b]
                    lb = n - b - 1
                    if (ma >> bpos) & 1:
                        bpos += lb
                        continue
                    cb = 0
                    for c in range(b + 1, n - 2):
                        mc = masks[c]
                        lc = n - c - 1
                        if (ma >> (bpos + cb)) & 1 or (mb >> cb) & 1:
                            cb += 1
                            continue
                        cb2 = 0
                        for d in range(c + 1, n - 1):
                            md = masks[d]
                            ld = n - d - 1
                            if ((ma >> (bpos + cb + cb2)) & 1) or ((mb >> (cb + cb2)) & 1) or ((mc >> cb2) & 1):
                                cb2 += 1
                                continue
                            cb3 = 0
                            for e in range(d + 1, n):
                                if (((ma >> (bpos + cb + cb2 + cb3)) & 1) or
                                        ((mb >> (cb + cb2 + cb3)) & 1) or
                                        ((mc >> (cb2 + cb3)) & 1) or
                                        ((md >> cb3) & 1)):
                                    cb3 += 1
                                    continue
                                found = (a, b, c, d, e)
                                break
                            if found:
                                break
                            cb2 += 1
                        if found:
                            break
                        cb += 1
                    if found:
                        break
                    bpos += lb
                if found is None:
                    a = rng.randrange(n - 2)
                    b = rng.randrange(a + 2, n)
                    c = rng.randrange(b + 1, n)
                    pos = (b - a - 1) + (c - b - 1)
                    masks[a] ^= (1 << pos)
                else:
                    a, b, c, d, e = found
                    # flip one of the 10 blue triples to red
                    tris = [(a, b, c), (a, b, d), (a, b, e), (a, c, d),
                            (a, c, e), (a, d, e), (b, c, d), (b, c, e),
                            (b, d, e), (c, d, e)]
                    x, y, z = tris[rng.randrange(10)]
                    pos = (y - x - 1) + (z - y - 1)
                    masks[x] ^= (1 << pos)
            # recompute energy (exact) periodically / on acceptance
            red_k4, blue_k5 = _count_violations(masks, n, quads, pents)
            energy = red_k4 + blue_k5
            if energy < best_energy:
                best_energy = energy
                best_masks = masks[:]
        restart += 1
    if best_energy == 0 and best_masks is not None:
        return _to_colors(best_masks, n)
    return None


def construct():
    deadline = _START + _DEADLINE - 5.0
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    for n in range(9, 35):
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if remaining < 4.0:
            break
        budget = min(remaining, 35.0)
        try:
            triples, m, quads, pents, tri_quads, tri_pents = _build_data(n)
        except Exception:
            break
        colors = None
        for s in range(4):
            if time.time() >= deadline - 3.0:
                break
            rem = deadline - 3.0 - time.time()
            if rem < 2.0:
                break
            colors = _walksat(n, quads, pents, min(rem, budget / 4.0), seed=1000 + n * 10 + s)
            if colors is not None:
                break
        if colors is not None:
            best_n = n
            best_colors = colors
            # try to push a little further next iteration
    return {'n': best_n, 'colors': best_colors}
