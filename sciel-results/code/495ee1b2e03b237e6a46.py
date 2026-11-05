"""Minimize independence number of a linear 3-uniform hypergraph on 16 points."""
import random
import time
import numpy as np


def _alpha(triples, n=16):
    if not triples:
        return n
    tm = np.array([(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in triples], dtype=np.uint32)
    masks = np.arange(1 << n, dtype=np.uint32)
    best = 0
    cs = 16384
    for s in range(0, 1 << n, cs):
        ch = masks[s:s + cs]
        cont = (ch[:, None] & tm[None, :]) == tm[None, :]
        indep = ~cont.any(axis=1)
        if indep.any():
            im = ch[indep]
            cnts = np.array([bin(int(x)).count('1') for x in im])
            if cnts.max() > best:
                best = int(cnts.max())
    return best


def _largest_indep(triples, n=16):
    if not triples:
        return set(range(n))
    tm = np.array([(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in triples], dtype=np.uint32)
    masks = np.arange(1 << n, dtype=np.uint32)
    best_mask = 0
    best_cnt = 0
    cs = 16384
    for s in range(0, 1 << n, cs):
        ch = masks[s:s + cs]
        cont = (ch[:, None] & tm[None, :]) == tm[None, :]
        indep = ~cont.any(axis=1)
        if indep.any():
            im = ch[indep]
            cnts = np.array([bin(int(x)).count('1') for x in im])
            idx = int(np.argmax(cnts))
            if cnts[idx] > best_cnt:
                best_cnt = int(cnts[idx])
                best_mask = int(im[idx])
    return set(i for i in range(n) if (best_mask >> i) & 1)


def construct():
    deadline = time.time() + 140
    random.seed(777)
    n = 16
    all_triples = [(a, b, c) for a in range(n) for b in range(a + 1, n) for c in range(b + 1, n)]

    def greedy():
        used = set()
        tr = []
        cand = all_triples[:]
        random.shuffle(cand)
        for t in cand:
            a, b, c = t
            ps = [(a, b), (a, c), (b, c)]
            if all(p not in used for p in ps):
                used.update(ps)
                tr.append(list(t))
        return tr

    def hit(triples, limit):
        used = set()
        for t in triples:
            for i in range(3):
                for j in range(i + 1, 3):
                    used.add((min(t[i], t[j]), max(t[i], t[j])))
        for _ in range(80):
            if time.time() > limit:
                break
            indep = _largest_indep(triples, n)
            if len(indep) <= 4:
                break
            cands = []
            for t in all_triples:
                a, b, c = t
                ps = [(a, b), (a, c), (b, c)]
                if any(p in used for p in ps):
                    continue
                if not (set(t) & indep):
                    continue
                cands.append(t)
            if not cands:
                break
            best = max(cands, key=lambda t: len(set(t) & indep))
            a, b, c = best
            triples.append(list(best))
            used.update([(a, b), (a, c), (b, c)])
        return triples

    best_tr = None
    best_a = 16
    for _ in range(40):
        if time.time() > deadline:
            break
        tr = greedy()
        tr = hit(tr, deadline)
        a = _alpha(tr, n)
        if a < best_a:
            best_a = a
            best_tr = tr
            if a <= 4:
                break

    if best_tr is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    return best_tr
