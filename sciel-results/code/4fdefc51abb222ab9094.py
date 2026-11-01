import time
import itertools
import numpy as np

_START = None


def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START


def _deadline_left(margin=2.0):
    return 145.0 - _now() - margin


def _all_words(q, n):
    # list of tuples, 0..q-1
    return list(itertools.product(range(q), repeat=n))


def _hamming(a, b):
    d = 0
    for x, y in zip(a, b):
        if x != y:
            d += 1
    return d


def _hamming_fast(a, b):
    # a,b are tuples of ints; b is a list of chosen codewords (tuples)
    # returns min distance
    best = n = len(a)
    for c in b:
        d = 0
        for i in range(n):
            if a[i] != c[i]:
                d += 1
                if d >= best:
                    break
        if d < best:
            best = d
            if best == 0:
                break
    return best


def _covered(a, code):
    # code: list of tuples
    for c in code:
        d = 0
        for i in range(len(a)):
            if a[i] != c[i]:
                d += 1
                if d > 1:
                    break
        if d <= 1:
            return True
    return False


def _max_gap(a, code):
    # return max distance from a to any codeword
    best = 0
    for c in code:
        d = 0
        for i in range(len(a)):
            if a[i] != c[i]:
                d += 1
        if d > best:
            best = d
    return best


def _symmetry_score(word, code):
    # symmetry-guided: prefer words that are "balanced" relative to existing code
    # count how many codewords are at distance 1 from word (overlap of spheres)
    # a word with fewer distance-1 neighbors in code is more "central"/spread
    n = len(word)
    cnt = 0
    for c in code:
        d = 0
        for i in range(n):
            if word[i] != c[i]:
                d += 1
        if d == 1:
            cnt += 1
    return cnt


def construct(q, n, radius):
    # Greedy covering-code construction with local-search swaps.
    words = _all_words(q, n)
    N = len(words)

    # Initialize with a small seed (tetracode product) to get a valid-ish start,
    # then greedily add codewords to cover uncovered words, then local-search swaps.
    # For n>=4 use tetracode product; else brute force.
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(q) for b in range(q)]
        # take first min(9, ...) of tetracode scaled to q (works for q=3)
        base = tetracode[:min(len(tetracode), 9)]
        # extend to length n by padding with 0s
        code = [tuple(c + [0] * (n - len(c))) for c in base]
    else:
        code = []

    # Ensure code is a list of tuples, distinct
    seen = set()
    clean = []
    for c in code:
        if c not in seen:
            seen.add(c)
            clean.append(c)
    code = clean

    # Greedy add: repeatedly add the word that covers the most uncovered words
    # (with symmetry-guided tie-breaking: among ties, pick the one with fewer
    # distance-1 neighbors in code, i.e., more spread out).
    covered_mask = np.zeros(N, dtype=bool)

    def update_covered(c):
        # mark words within radius of c as covered
        for i, w in enumerate(words):
            if not covered_mask[i]:
                d = 0
                for j in range(n):
                    if w[j] != c[j]:
                        d += 1
                        if d > radius:
                            break
                if d <= radius:
                    covered_mask[i] = True

    for c in code:
        update_covered(c)

    deadline = 145.0

    # Greedy addition loop
    while True:
        if time.time() > deadline - 30:
            break
        if covered_mask.all():
            break
        # find uncovered words
        unc_idx = np.where(~covered_mask)[0]
        if len(unc_idx) == 0:
            break
        # candidate words: all words (or a sample if large)
        # For each candidate, compute how many currently-uncovered words it would cover
        # Use symmetry-guided tie-breaking.
        # To keep it fast, sample candidates.
        cand_sample = words if N <= 2000 else words[::max(1, N // 2000)]
        best_gain = -1
        best_word = None
        best_sym = None
        for c in cand_sample:
            # skip if already in code
            if c in seen:
                continue
            gain = 0
            for i in unc_idx:
                w = words[i]
                d = 0
                for j in range(n):
                    if w[j] != c[j]:
                        d += 1
                        if d > radius:
                            break
                if d <= radius:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_word = c
                best_sym = _symmetry_score(c, code)
            elif gain == best_gain and gain > 0:
                s = _symmetry_score(c, code)
                if best_sym is None or s < best_sym:
                    best_word = c
                    best_sym = s
        if best_word is None or best_gain <= 0:
            break
        code.append(best_word)
        seen.add(best_word)
        update_covered(best_word)

    # Local search: try to swap a codeword with another to reduce size (remove one)
    # or to improve coverage (not needed since we require coverage).
    # We aim to REDUCE the number of codewords: try removing a codeword and
    # see if coverage is still complete; if yes, remove it.
    # Repeat until no removal is possible or deadline.
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for k in range(len(code)):
            if time.time() > deadline - 10:
                break
            # try removing code[k]
            code_without = code[:k] + code[k + 1:]
            # check coverage
            ok = True
            for i, w in enumerate(words):
                d = 0
                for c in code_without:
                    d = 0
                    for j in range(n):
                        if w[j] != c[j]:
                            d += 1
                            if d > radius:
                                break
                    if d <= radius:
                        break
                else:
                    ok = False
                    break
            if ok:
                code = code_without
                seen.discard(code[k])
                improved = True

    # Final local-search swaps: try replacing a codeword with another to allow
    # subsequent removals.
    if time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            found = False
            for k in range(len(code)):
                if time.time() > deadline - 5:
                    break
                for c in words:
                    if c in seen:
                        continue
                    new_code = code[:k] + [c] + code[k + 1:]
                    # check coverage
                    ok = True
                    for w in words:
                        d = 0
                        for cc in new_code:
                            d = 0
                            for j in range(n):
                                if w[j] != cc[j]:
                                    d += 1
                                    if d > radius:
                                        break
                            if d <= radius:
                                break
                        else:
                            ok = False
                            break
                    if ok:
                        # this swap keeps coverage; try to then remove a codeword
                        for m in range(len(new_code)):
                            code_without = new_code[:m] + new_code[m + 1:]
                            ok2 = True
                            for w in words:
                                d = 0
                                for cc in code_without:
                                    d = 0
                                    for j in range(n):
                                        if w[j] != cc[j]:
                                            d += 1
                                            if d > radius:
                                                break
                                    if d <= radius:
                                        break
                                else:
                                    ok2 = False
                                    break
                            if ok2:
                                code = code_without
                                seen.discard(code[k])
                                seen.add(c)
                                found = True
                                break
                        if found:
                            break
                if found:
                    break
            if not found:
                break

    # Convert to list of lists
    return [list(c) for c in code]
