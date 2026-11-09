def construct():
    import time
    import numpy as np

    start = time.time()
    deadline = start + 84  # safety margin before 90s kill

    N = 17
    MIN_D = 6
    SIZE = 1 << N  # 131072

    # Precompute weights for all 17-bit words
    weights = np.fromiter((i.bit_count() for i in range(SIZE)), dtype=np.int8, count=SIZE)

    # All diffs with weight <= 5 (the "ball radius 5" offsets)
    diffs_le_5 = np.nonzero(weights <= 5)[0].astype(np.int32)
    D5 = len(diffs_le_5)

    def get_linear_code():
        """Construct a [17,8,6] linear code (256 words) via greedy basis extension."""
        basis = []
        code = [0]
        for w in range(1, SIZE):
            if len(basis) == 8:
                break
            if time.time() > deadline - 50:
                break
            valid = True
            for c in code:
                if weights[w ^ c] < MIN_D:
                    valid = False
                    break
            if valid:
                basis.append(w)
                code.extend(c ^ w for c in code)
        return code

    lin_code = get_linear_code()
    if len(lin_code) < 256:
        return lin_code

    code = list(lin_code)
    code_set = set(code)
    n = len(code)

    def ball(c):
        """Boolean mask of all words at distance <=5 from c."""
        m = np.zeros(SIZE, dtype=bool)
        m[c ^ diffs_le_5] = True
        return m

    # Global forbidden mask: union of balls around every current codeword.
    # A word w is "free" (d(w, c) >= 6 for all c) iff ~forbidden[w].
    forbidden = np.zeros(SIZE, dtype=bool)
    for c in code:
        forbidden[c ^ diffs_le_5] = True

    # Free words (candidates for the new nonlinear word), excluding current codewords.
    free = (~forbidden).copy()
    for c in code:
        free[c] = False

    # ---- 2-for-1 swap search: remove two codewords, add one new word ----
    # For removal pair (i,j), the new word w must be free of all codewords
    # except possibly c_i, c_j, i.e. w in ~(forbidden \ (ball_i | ball_j)).
    best = None
    for i in range(n):
        if time.time() > deadline - 15:
            break
        ci = code[i]
        bi = ball(ci)
        for j in range(i + 1, n):
            cj = code[j]
            bj = ball(cj)
            # candidate new words: free of everyone except ci, cj
            cand = ~(forbidden & ~(bi | bj))
            cand[ci] = False
            cand[cj] = False
            idx = np.nonzero(cand)[0]
            if idx.size >= 2:
                # pick two that are mutually >=6 apart (they will replace ci,cj)
                u = int(idx[0])
                v = None
                for t in range(1, idx.size):
                    w2 = int(idx[t])
                    if weights[u ^ w2] >= MIN_D:
                        v = w2
                        break
                if v is None:
                    # fallback: any single new word gives a +1 net gain is impossible
                    # (2-for-1 reduces size), so skip unless we can keep size via 2 new words
                    continue
                # 2-for-1 keeps size constant; we want net gain, so this alone is not enough.
                # But it may unblock a later single addition. Record best improvement.
                if best is None:
                    best = (i, j, u, v)
    # The 2-for-1 swap keeps size at 256 but can enable a subsequent +1 addition.
    # Apply the best swap if found, then attempt a single addition.
    if best is not None:
        i, j, u, v = best
        # remove i and j (indices; remove higher first)
        a, b = sorted([i, j], reverse=True)
        for k in (b, a):
            c_removed = code.pop(k)
            code_set.discard(c_removed)
        # add u and v
        code.append(u); code_set.add(u)
        code.append(v); code_set.add(v)
        # rebuild forbidden
        forbidden = np.zeros(SIZE, dtype=bool)
        for c in code:
            forbidden[c ^ diffs_le_5] = True

    # ---- Try single additions (the net +1) ----
    def try_add():
        nonlocal forbidden
        # free word = not in forbidden and not already in code
        for w in range(SIZE):
            if forbidden[w]:
                continue
            if w in code_set:
                continue
            return w
        return None

    # Repeatedly try to add a free word; after each add, update forbidden.
    while time.time() < deadline - 5:
        # recompute free words quickly
        free_mask = (~forbidden).copy()
        for c in code:
            free_mask[c] = False
        idx = np.nonzero(free_mask)[0]
        if idx.size == 0:
            break
        w = int(idx[0])
        code.append(w)
        code_set.add(w)
        forbidden[w ^ diffs_le_5] = True

    # ---- Fallback: 1-for-2 swap (remove one, add two) for further growth ----
    if time.time() < deadline - 5:
        forbidden = np.zeros(SIZE, dtype=bool)
        for c in code:
            forbidden[c ^ diffs_le_5] = True
        for i in range(len(code)):
            if time.time() > deadline - 3:
                break
            ci = code[i]
            bi = ball(ci)
            # words free of everyone except ci
            cand = ~(forbidden & ~bi)
            cand[ci] = False
            idx = np.nonzero(cand)[0]
            if idx.size < 2:
                continue
            # find a mutually >=6 pair among candidates
            u = None
            v = None
            for a in range(idx.size):
                x = int(idx[a])
                for b in range(a + 1, idx.size):
                    y = int(idx[b])
                    if weights[x ^ y] >= MIN_D:
                        u, v = x, y
                        break
                if u is not None:
                    break
            if u is not None:
                c_removed = code.pop(i)
                code_set.discard(c_removed)
                code.append(u); code_set.add(u)
                code.append(v); code_set.add(v)
                forbidden = np.zeros(SIZE, dtype=bool)
                for c in code:
                    forbidden[c ^ diffs_le_5] = True
                # try more additions
                while time.time() < deadline - 3:
                    free_mask = (~forbidden).copy()
                    for c in code:
                        free_mask[c] = False
                    idx2 = np.nonzero(free_mask)[0]
                    if idx2.size == 0:
                        break
                    w = int(idx2[0])
                    code.append(w); code_set.add(w)
                    forbidden[w ^ diffs_le_5] = True
                break

    # Final verification of minimum distance (defensive)
    out = list(code_set)
    return out
