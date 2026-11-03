"""Covering code K_q(n, radius): minimize number of centers covering {0..q-1}^n.
Combines direct-sum seed + bitset greedy + safety-margin packing + 2-opt local search."""
import time
import random
import itertools
import numpy as np


def _build_word_tables(q, n):
    N = q ** n
    digits = np.empty((N, n), dtype=np.int32)
    for i in range(N):
        x = i
        for j in range(n - 1, -1, -1):
            digits[i, j] = x % q
            x //= q
    return N, digits


def _build_neighbors(N, n, q, radius, digits):
    """neighbor_lists[c] = list of word-indices within Hamming distance <= radius of c."""
    neighbor_lists = [[] for _ in range(N)]
    for c in range(N):
        lst = [c]
        if radius >= 1:
            for i in range(n):
                orig = digits[c, i]
                for nv in range(q):
                    if nv != orig:
                        lst.append(c + (nv - orig) * (q ** (n - 1 - i)))
        if radius >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    oi = digits[c, i]
                    oj = digits[c, j]
                    pi = q ** (n - 1 - i)
                    pj = q ** (n - 1 - j)
                    for ni in range(q):
                        di = ni - oi
                        for nj in range(q):
                            dj = nj - oj
                            if di == 0 and dj == 0:
                                continue
                            lst.append(c + di * pi + dj * pj)
        neighbor_lists[c] = lst
    return neighbor_lists


def _build_masks(neighbor_lists):
    masks = []
    for lst in neighbor_lists:
        m = 0
        for w in lst:
            m |= (1 << w)
        masks.append(m)
    return masks


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 35s safety margin before 150s kill

    N = q ** n
    N, digits = _build_word_tables(q, n)
    neighbor_lists = _build_neighbors(N, n, q, radius, digits)
    masks = _build_masks(neighbor_lists)

    # ---------- Initial feasible code ----------
    # Direct-sum seed for the known K_3(8,1) structure, else criticality greedy.
    if q == 3 and n == 8 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = []
        for t in tetracode:
            base = 0
            for i in range(4):
                base = base * 3 + t[i]
            for tail in itertools.product(range(3), repeat=4):
                idx = base
                for v in tail:
                    idx = idx * 3 + v
                code.append(idx)
    else:
        code = []
        code_set = set()
        uncovered = (1 << N) - 1
        # criticality: number of codewords that can cover each word
        word_to_coders = [[] for _ in range(N)]
        for c in range(N):
            for w in neighbor_lists[c]:
                word_to_coders[w].append(c)
        while uncovered:
            if time.time() > deadline - 20.0:
                break
            # pick uncovered word with fewest potential coders
            min_crit = N + 1
            target = -1
            temp = uncovered
            while temp:
                lsb = temp & (-temp)
                w = lsb.bit_length() - 1
                crit = len(word_to_coders[w])
                if crit < min_crit:
                    min_crit = crit
                    target = w
                    if min_crit == 1:
                        break
                temp ^= lsb
            if target < 0:
                break
            best_c = -1
            best_gain = -1
            for c in word_to_coders[target]:
                if c in code_set:
                    continue
                gain = (uncovered & masks[c]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c < 0:
                break
            code.append(best_c)
            code_set.add(best_c)
            uncovered &= ~masks[best_c]

    # Ensure feasibility (fill any gaps)
    if time.time() < deadline - 10.0:
        covered = 0
        for c in code:
            covered |= masks[c]
        if covered != (1 << N) - 1:
            uncovered = ((1 << N) - 1) & ~covered
            while uncovered:
                if time.time() > deadline - 10.0:
                    break
                lsb = uncovered & (-uncovered)
                w = lsb.bit_length() - 1
                code.append(w)
                covered |= masks[w]
                uncovered &= ~masks[w]

    # ---------- Safety-margin (packing) refinement ----------
    # Iteratively place the uncovered word farthest from any existing center.
    if time.time() < deadline - 15.0:
        covered = 0
        for c in code:
            covered |= masks[c]
        uncovered = ((1 << N) - 1) & ~covered
        # min distance from each word to nearest center (Hamming), radius>=1
        # For radius 1, "uncovered" words are at distance >=2 from every center.
        # We add centers that maximize min-distance to existing set (uniform density).
        placed = set(code)
        # distance to nearest placed center (only need up to some cap)
        min_dist = np.full(N, 1000, dtype=np.int32)
        for c in code:
            # words at distance 0
            min_dist[c] = 0
            if radius >= 1:
                for w in neighbor_lists[c]:
                    if min_dist[w] > 1:
                        min_dist[w] = 1
        # distance 2 words
        if radius >= 1:
            for c in code:
                for w1 in neighbor_lists[c]:
                    if w1 == c:
                        continue
                    for w2 in neighbor_lists[w1]:
                        if w2 != c and w2 != w1 and min_dist[w2] > 2:
                            min_dist[w2] = 2
        guard = 0
        while uncovered and time.time() < deadline - 12.0:
            guard += 1
            if guard > 200:
                break
            # pick uncovered word with max min_dist (safety margin)
            best_w = -1
            best_d = -1
            temp = uncovered
            while temp:
                lsb = temp & (-temp)
                w = lsb.bit_length() - 1
                d = min_dist[w]
                if d > best_d:
                    best_d = d
                    best_w = w
                temp ^= lsb
            if best_w < 0:
                break
            code.append(best_w)
            placed.add(best_w)
            covered |= masks[best_w]
            uncovered &= ~masks[best_w]
            # update min_dist around new center
            min_dist[best_w] = 0
            for w1 in neighbor_lists[best_w]:
                if min_dist[w1] > 1:
                    min_dist[w1] = 1
            for w1 in neighbor_lists[best_w]:
                if w1 == best_w:
                    continue
                for w2 in neighbor_lists[w1]:
                    if w2 != best_w and w2 != w1 and min_dist[w2] > 2:
                        min_dist[w2] = 2

    # ---------- Local search ----------
    def _coverage_counts(code_list):
        cc = np.zeros(N, dtype=np.int32)
        for c in code_list:
            for w in neighbor_lists[c]:
                cc[w] += 1
        return cc

    def _is_covered(code_list):
        cc = _coverage_counts(code_list)
        return not np.any(cc == 0)

    # 1) remove redundant centers
    if time.time() < deadline - 8.0:
        cc = _coverage_counts(code)
        changed = True
        while changed and time.time() < deadline - 8.0:
            changed = False
            for i in range(len(code)):
                c = code[i]
                redundant = True
                for w in neighbor_lists[c]:
                    if cc[w] == 1:
                        redundant = False
                        break
                if redundant:
                    for w in neighbor_lists[c]:
                        cc[w] -= 1
                    code.pop(i)
                    changed = True
                    break

    # 2) 2-opt: replace a center to enable removing another
    if time.time() < deadline - 5.0:
        code_set = set(code)
        cc = _coverage_counts(code)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(code)):
                if time.time() > deadline - 5.0:
                    break
                c_rm = code[i]
                newly_uncovered = [w for w in neighbor_lists[c_rm] if cc[w] == 1]
                if not newly_uncovered:
                    for w in neighbor_lists[c_rm]:
                        cc[w] -= 1
                    code.pop(i)
                    code_set.discard(c_rm)
                    improved = True
                    break
                # try to find one center that covers all newly_uncovered, not in code
                best_add = -1
                best_extra = -1
                nu_set = set(newly_uncovered)
                # candidate centers: any word covering all newly_uncovered
                # sample: for each newly_uncovered word, its coders
                cand = set()
                for w in newly_uncovered:
                    for c2 in word_to_coders_all(w):
                        cand.add(c2)
                for c2 in cand:
                    if c2 in code_set:
                        continue
                    if not nu_set.issubset(set(neighbor_lists[c2])):
                        continue
                    extra = sum(1 for w in neighbor_lists[c2] if cc[w] == 0)
                    if extra > best_extra:
                        best_extra = extra
                        best_add = c2
                if best_add >= 0:
                    # add best_add, remove c_rm
                    for w in neighbor_lists[best_add]:
                        cc[w] += 1
                    for w in neighbor_lists[c_rm]:
                        cc[w] -= 1
                    code[i] = best_add
                    code_set.discard(c_rm)
                    code_set.add(best_add)
                    improved = True
                    break

    # Final feasibility guarantee
    cc = _coverage_counts(code)
    if np.any(cc == 0):
        uncovered = ((1 << N) - 1) & ~masks_reduce(code)
        while uncovered and time.time() < deadline:
            lsb = uncovered & (-uncovered)
            w = lsb.bit_length() - 1
            code.append(w)
            uncovered &= ~masks[w]

    # Deduplicate
    seen = set()
    out = []
    for c in code:
        if c not in seen:
            seen.add(c)
            word = [0] * n
            x = c
            for j in range(n - 1, -1, -1):
                word[j] = x % q
                x //= q
            out.append(word)
    return out


# Global helper cache for word_to_coders (built lazily)
_WORD_TO_CODERS_CACHE = {}


def _build_word_to_coders(N, neighbor_lists):
    wtc = [[] for _ in range(N)]
    for c in range(N):
        for w in neighbor_lists[c]:
            wtc[w].append(c)
    return wtc


def word_to_coders_all(w):
    # placeholder to keep signature; real data attached via closure below
    return _GLOBAL_WTC[w]


_GLOBAL_WTC = []


def construct(q, n, radius):
    global _GLOBAL_WTC
    start_time = time.time()
    deadline = start_time + 115.0

    N = q ** n
    N, digits = _build_word_tables(q, n)
    neighbor_lists = _build_neighbors(N, n, q, radius, digits)
    masks = _build_masks(neighbor_lists)
    _GLOBAL_WTC = _build_word_to_coders(N, neighbor_lists)

    # ---------- Initial feasible code ----------
    if q == 3 and n == 8 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = []
        for t in tetracode:
            base = 0
            for i in range(4):
                base = base * 3 + t[i]
            for tail in itertools.product(range(3), repeat=4):
                idx = base
                for v in tail:
                    idx = idx * 3 + v
                code.append(idx)
    else:
        code = []
        code_set = set()
        uncovered = (1 << N) - 1
        while uncovered:
            if time.time() > deadline - 20.0:
                break
            min_crit = N + 1
            target = -1
            temp = uncovered
            while temp:
                lsb = temp & (-temp)
                w = lsb.bit_length() - 1
                crit = len(_GLOBAL_WTC[w])
                if crit < min_crit:
                    min_crit = crit
                    target = w
                    if min_crit == 1:
                        break
                temp ^= lsb
            if target < 0:
                break
            best_c = -1
            best_gain = -1
            for c in _GLOBAL_WTC[target]:
                if c in code_set:
                    continue
                gain = (uncovered & masks[c]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c < 0:
                break
            code.append(best_c)
            code_set.add(best_c)
            uncovered &= ~masks[best_c]

    # ensure feasibility
    if time.time() < deadline - 10.0:
        covered = 0
        for c in code:
            covered |= masks[c]
        if covered != (1 << N) - 1:
            uncovered = ((1 << N) - 1) & ~covered
            while uncovered and time.time() < deadline - 10.0:
                lsb = uncovered & (-uncovered)
                w = lsb.bit_length() - 1
                code.append(w)
                covered |= masks[w]
                uncovered &= ~masks[w]

    # ---------- Safety-margin (packing) refinement ----------
    if time.time() < deadline - 15.0:
        covered = 0
        for c in code:
            covered |= masks[c]
        uncovered = ((1 << N) - 1) & ~covered
        min_dist = np.full(N, 1000, dtype=np.int32)
        for c in code:
            min_dist[c] = 0
            for w in neighbor_lists[c]:
                if min_dist[w] > 1:
                    min_dist[w] = 1
        for c in code:
            for w1 in neighbor_lists[c]:
                if w1 == c:
                    continue
                for w2 in neighbor_lists[w1]:
                    if w2 != c and w2 != w1 and min_dist[w2] > 2:
                        min_dist[w2] = 2
        guard = 0
        while uncovered and time.time() < deadline - 12.0:
            guard += 1
            if guard > 300:
                break
            best_w = -1
            best_d = -1
            temp = uncovered
            while temp:
                lsb = temp & (-temp)
                w = lsb.bit_length() - 1
                d = min_dist[w]
                if d > best_d:
                    best_d = d
                    best_w = w
                temp ^= lsb
            if best_w < 0:
                break
            code.append(best_w)
            covered |= masks[best_w]
            uncovered &= ~masks[best_w]
            min_dist[best_w] = 0
            for w1 in neighbor_lists[best_w]:
                if min_dist[w1] > 1:
                    min_dist[w1] = 1
            for w1 in neighbor_lists[best_w]:
                if w1 == best_w:
                    continue
                for w2 in neighbor_lists[w1]:
                    if w2 != best_w and w2 != w1 and min_dist[w2] > 2:
                        min_dist[w2] = 2

    # ---------- Local search ----------
    def _coverage_counts(cl):
        cc = np.zeros(N, dtype=np.int32)
        for c in cl:
            for w in neighbor_lists[c]:
                cc[w] += 1
        return cc

    # 1) remove redundant centers
    if time.time() < deadline - 8.0:
        cc = _coverage_counts(code)
        changed = True
        while changed and time.time() < deadline - 8.0:
            changed = False
            for i in range(len(code)):
                c = code[i]
                redundant = True
                for w in neighbor_lists[c]:
                    if cc[w] == 1:
                        redundant = False
                        break
                if redundant:
                    for w in neighbor_lists[c]:
                        cc[w] -= 1
                    code.pop(i)
                    changed = True
                    break

    # 2) 2-opt replace
    if time.time() < deadline - 5.0:
        code_set = set(code)
        cc = _coverage_counts(code)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(code)):
                if time.time() > deadline - 5.0:
                    break
                c_rm = code[i]
                newly_uncovered = [w for w in neighbor_lists[c_rm] if cc[w] == 1]
                if not newly_uncovered:
                    for w in neighbor_lists[c_rm]:
                        cc[w] -= 1
                    code.pop(i)
                    code_set.discard(c_rm)
                    improved = True
                    break
                nu_set = set(newly_uncovered)
                best_add = -1
                best_extra = -1
                cand = set()
                for w in newly_uncovered:
                    for c2 in _GLOBAL_WTC[w]:
                        cand.add(c2)
                for c2 in cand:
                    if c2 in code_set:
                        continue
                    ok = True
                    for w in newly_uncovered:
                        if cc[w] > 0 or True:
                            pass
                    # check c2 covers all newly_uncovered
                    nb2 = set(neighbor_lists[c2])
                    if not nu_set.issubset(nb2):
                        continue
                    extra = 0
                    for w in neighbor_lists[c2]:
                        if cc[w] == 0:
                            extra += 1
                    if extra > best_extra:
                        best_extra = extra
                        best_add = c2
                if best_add >= 0:
                    for w in neighbor_lists[best_add]:
                        cc[w] += 1
                    for w in neighbor_lists[c_rm]:
                        cc[w] -= 1
                    code[i] = best_add
                    code_set.discard(c_rm)
                    code_set.add(best_add)
                    improved = True
                    break

    # Final feasibility guarantee
    cc = _coverage_counts(code)
    if np.any(cc == 0):
        covered = 0
        for c in code:
            covered |= masks[c]
        uncovered = ((1 << N) - 1) & ~covered
        while uncovered and time.time() < deadline:
            lsb = uncovered & (-uncovered)
            w = lsb.bit_length() - 1
            code.append(w)
            covered |= masks[w]
            uncovered &= ~masks[w]

    # Deduplicate + convert to word lists
    seen = set()
    out = []
    for c in code:
        if c not in seen:
            seen.add(c)
            word = [0] * n
            x = c
            for j in range(n - 1, -1, -1):
                word[j] = x % q
                x //= q
            out.append(word)
    return out
