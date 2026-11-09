import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start = time.time()
    deadline = start + 120
    total = q ** n
    random.seed(12345)

    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    idx = {w: i for i, w in enumerate(all_words)}
    words = all_words

    # Precompute per-position neighbor lists for each word (radius 1 ball minus self)
    neigh = [[] for _ in range(total)]
    for w in words:
        lst = []
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                lst.append((pos, idx[tuple(nw)]))
        neigh[w and idx[w] or idx[w]] = lst
    # fix: index by position in list
    neigh = [None] * total
    for w in words:
        i = idx[w]
        lst = []
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                lst.append((pos, idx[tuple(nw)]))
        neigh[i] = lst

    def ball_members(i):
        """list of indices covered by codeword i (including self)"""
        res = [i]
        res.extend(j for (_p, j) in neigh[i])
        return res

    # ---- Initial code: direct sum of tetracode (known 486) ----
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = []
    seen = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            cw = tuple(h + list(tail))
            if cw not in seen:
                seen.add(cw)
                code.append(idx[cw])
    code = list(dict.fromkeys(code))

    # ---- Build cover counts ----
    def build_counts(code):
        counts = np.zeros(total, dtype=np.int32)
        for i in code:
            counts[i] += 1
            for (_p, j) in neigh[i]:
                counts[j] += 1
        return counts

    counts = build_counts(code)

    def is_valid(counts):
        return np.min(counts) >= 1

    # ---- 1-opt: remove redundant codewords ----
    def try_remove(code, counts):
        changed = True
        while changed:
            changed = False
            # order by smallest coverage of uniquely-needed words
            for k in range(len(code) - 1, -1, -1):
                i = code[k]
                # check if any member has count==1
                if counts[i] == 1:
                    continue
                ok = True
                for (_p, j) in neigh[i]:
                    if counts[j] == 1:
                        ok = False
                        break
                if ok:
                    counts[i] -= 1
                    for (_p, j) in neigh[i]:
                        counts[j] -= 1
                    code.pop(k)
                    changed = True
                    break
        return code, counts

    code, counts = try_remove(code, counts)

    # ---- Greedy add if not valid (shouldn't happen) ----
    if not is_valid(counts):
        while not is_valid(counts) and (time.time() - start) < deadline - 20:
            # find uncovered word with most... just add a codeword covering many uncovered
            uncovered = np.where(counts == 0)[0]
            best = -1
            bestc = -1
            for cand in range(total):
                c = np.sum(counts[ball_members(cand)] == 0)
                if c > bestc:
                    bestc = c
                    best = cand
            if best == -1:
                break
            code.append(best)
            counts[best] += 1
            for (_p, j) in neigh[best]:
                counts[j] += 1
        code, counts = try_remove(code, counts)

    # ---- 2-opt: replace codeword A with B to allow removal of C ----
    def two_opt(code, counts, budget):
        improved = True
        while improved and (time.time() - start) < budget:
            improved = False
            # random order of A
            order = list(range(len(code)))
            random.shuffle(order)
            for ai in order:
                A = code[ai]
                # candidate B: pick from uncovered-ish region; sample
                # We'll sample B from words far from A
                for _ in range(40):
                    B = random.randrange(total)
                    if B == A:
                        continue
                    # tentatively remove A
                    counts[A] -= 1
                    for (_p, j) in neigh[A]:
                        counts[j] -= 1
                    # find a codeword C (other than A) that becomes removable and whose removal is "enabled"
                    # Actually we want: after removing A and adding B, can we remove some C?
                    # Simpler: after removing A, is code still valid? If yes, we already reduced -> handle in 1-opt.
                    # For 2-opt: remove A, add B, then try to remove a C.
                    # Check validity after removing A
                    if np.min(counts) < 1:
                        # adding B may fix
                        counts[B] += 1
                        for (_p, j) in neigh[B]:
                            counts[j] += 1
                        # now try to remove some C
                        removed = False
                        for k in range(len(code) - 1, -1, -1):
                            if code[k] == B:
                                continue
                            C = code[k]
                            if counts[C] == 1:
                                continue
                            ok = True
                            for (_p, j) in neigh[C]:
                                if counts[j] == 1:
                                    ok = False
                                    break
                            if ok:
                                counts[C] -= 1
                                for (_p, j) in neigh[C]:
                                    counts[j] -= 1
                                code.pop(k)
                                removed = True
                                break
                        if not removed:
                            # revert B
                            counts[B] -= 1
                            for (_p, j) in neigh[B]:
                                counts[j] -= 1
                            # revert A
                            counts[A] += 1
                            for (_p, j) in neigh[A]:
                                counts[j] += 1
                            continue
                        else:
                            # keep: A removed, B added, C removed
                            # remove A from code list
                            code = [x for x in code if x != A]
                            if B not in code:
                                code.append(B)
                            improved = True
                            break
                    else:
                        # removing A alone kept valid -> revert and let 1-opt handle
                        counts[A] += 1
                        for (_p, j) in neigh[A]:
                            counts[j] += 1
                if improved:
                    break
        return code, counts

    # Interleave 1-opt and 2-opt
    while (time.time() - start) < deadline - 15:
        before = len(code)
        code, counts = try_remove(code, counts)
        if len(code) != before:
            continue
        code, counts = two_opt(code, counts, deadline - 15)
        if len(code) == before:
            # try a few random restarts of 2-opt with different seeds
            break

    # Final 1-opt
    code, counts = try_remove(code, counts)

    # Ensure validity
    if not is_valid(counts):
        # rebuild safety: add until valid
        while not is_valid(counts) and (time.time() - start) < deadline - 2:
            uncovered = np.where(counts == 0)[0]
            if len(uncovered) == 0:
                break
            # add codeword covering most uncovered
            best = -1
            bestc = -1
            for cand in uncovered[:200]:
                c = np.sum(counts[ball_members(cand)] == 0)
                if c > bestc:
                    bestc = c
                    best = cand
            if best == -1:
                best = int(uncovered[0])
            code.append(best)
            counts[best] += 1
            for (_p, j) in neigh[best]:
                counts[j] += 1
        code, counts = try_remove(code, counts)

    # Deduplicate
    uniq = list(dict.fromkeys(code))
    result = [list(words[i]) for i in uniq]
    return result
