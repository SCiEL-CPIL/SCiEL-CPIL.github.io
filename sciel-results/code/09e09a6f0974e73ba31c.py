import time
import random
import itertools


def construct(q, n, radius):
    if (q, n, radius) != (3, 6, 1):
        # Fallback: direct-sum of tetracode (valid for q=3, n>=4, radius=1)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]

    start = time.time()
    deadline = start + 145.0  # safety margin under the 150s kill

    Q = q
    N = n
    R = radius
    total = Q ** N
    # 1 = 3^0, 3 = 3^1, 9 = 3^2, 27 = 3^3, 81 = 3^4, 243 = 3^5
    pow3 = [1]
    for _ in range(N):
        pow3.append(pow3[-1] * Q)
    # pow3[i] = 3^i

    def idx(w):
        v = 0
        for i in range(N):
            v += w[i] * pow3[i]
        return v

    def unidx(v):
        w = [0] * N
        for i in range(N):
            w[i] = v % Q
            v //= Q
        return w

    # Precompute neighbor lists (Hamming distance <= radius)
    neighbors = []
    for v in range(total):
        w = unidx(v)
        nbrs = [v]
        for i in range(N):
            orig = w[i]
            for c in range(Q):
                if c == orig:
                    continue
                w[i] = c
                nbrs.append(idx(w))
            w[i] = orig
        neighbors.append(nbrs)

    def ball(v):
        return neighbors[v]

    # Initial code: tetracode direct sum (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init = [idx(h + list(tail))
            for h in tetracode
            for tail in itertools.product(range(Q), repeat=N - 4)]
    init_set = set(init)

    # covered bitset (list of bool)
    covered = [False] * total
    for v in init:
        for u in ball(v):
            covered[u] = True

    def is_full(covered):
        # check all covered
        for u in range(total):
            if not covered[u]:
                return False
        return True

    def greedy_build():
        cov = [False] * total
        chosen = []
        chosen_set = set()
        # precompute uncovered count per candidate (only over not-yet-covered)
        # We'll recompute gain on the fly using cov
        # candidate pool = all words
        for _ in range(total):
            best = -1
            best_gain = -1
            for c in range(total):
                if c in chosen_set:
                    continue
                gain = 0
                for u in ball(c):
                    if not cov[u]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best = c
                    if gain == 0:
                        break
            if best < 0 or best_gain == 0:
                break
            chosen.append(best)
            chosen_set.add(best)
            for u in ball(best):
                cov[u] = True
        return chosen, cov, chosen_set

    # Greedy from scratch (baseline), then improve
    best_code, best_cov, best_set = greedy_build()
    best_size = len(best_code)

    # Local improvement: deletions + swaps
    def try_deletions(code, cov, cset):
        # returns (code, cov, cset, improved)
        improved = True
        while improved:
            improved = False
            for c in list(code):
                # check if removing c leaves everything covered
                ok = True
                for u in ball(c):
                    if cov[u]:
                        # need another codeword covering u
                        # quick: count coverage
                        pass
                # compute: for each u in ball(c), is there another codeword v != c with u in ball(v)?
                for u in ball(c):
                    # find if covered by another
                    found = False
                    for v in code:
                        if v == c:
                            continue
                        if u in neighbors[v]:
                            found = True
                            break
                    if not found:
                        ok = False
                        break
                if ok:
                    # remove c
                    code.remove(c)
                    cset.discard(c)
                    # recompute cov
                    newcov = [False] * total
                    for v in code:
                        for u in ball(v):
                            newcov[u] = True
                    cov[:] = newcov
                    improved = True
        return code, cov, cset

    def recompute_cov(code):
        cov = [False] * total
        for v in code:
            for u in ball(v):
                cov[u] = True
        return cov

    def try_swaps(code, cov, cset):
        # 1-for-1: replace c with x if it reduces size? no, keeps size. Try to enable a later deletion.
        # We'll do: for each c, for each x not in code, if replacing c with x keeps full coverage
        # AND then a deletion becomes possible, accept.
        improved = False
        # To limit cost, only consider x that cover currently poorly-covered words.
        # Build for each u the set of codewords covering it (count)
        # We'll attempt a bounded number of swaps per call.
        cand_pool = list(range(total))
        random.shuffle(cand_pool)
        for x in cand_pool:
            if x in cset:
                continue
            # try replacing each c
            for c in list(code):
                if c == x:
                    continue
                # new code = code - {c} + {x}
                # check coverage
                ok = True
                # words only covered by c (and not x) would break
                # compute coverage of (code - {c}) and see if x covers the gap
                # For each u in ball(c): is u covered by x or by another in code?
                for u in ball(c):
                    covered_else = False
                    if u in neighbors[x]:
                        covered_else = True
                    else:
                        for v in code:
                            if v == c:
                                continue
                            if u in neighbors[v]:
                                covered_else = True
                                break
                    if not covered_else:
                        ok = False
                        break
                if ok:
                    # replace
                    code.remove(c)
                    cset.discard(c)
                    code.append(x)
                    cset.add(x)
                    cov[:] = recompute_cov(code)
                    improved = True
                    # after swap, try deletions
                    code, cov, cset = try_deletions(code, cov, cset)
                    break
            if time.time() > deadline:
                break
        return code, cov, cset, improved

    # Run improvement loop with time guard
    code = list(best_code)
    cov = list(best_cov)
    cset = set(best_set)

    # First, deletions
    code, cov, cset = try_deletions(code, cov, cset)
    if len(code) < best_size:
        best_size = len(code)
        best_code = list(code)
        best_cov = list(cov)
        best_set = set(cset)

    # Swaps
    improved = True
    while improved and time.time() < deadline:
        code, cov, cset, improved = try_swaps(code, cov, cset)
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            best_cov = list(cov)
            best_set = set(cset)

    # Return best found
    result = [unidx(v) for v in best_code]
    # ensure distinct (they are, by set)
    return result
