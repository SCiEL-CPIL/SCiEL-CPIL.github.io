import numpy as np
import time
from itertools import combinations
import random

def construct():
    N = 17
    MIN_DIST = 6
    FULL = 1 << N
    start_time = time.time()
    DEADLINE = start_time + 85.0  # safety margin under 90s

    def time_ok():
        return time.time() < DEADLINE

    # Precompute neighbor lists: for each word, all words within distance MIN_DIST-1
    # (i.e. distance 0..5). Used for O(1) blocked updates.
    neighbors = [[] for _ in range(FULL)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            for w in range(FULL):
                neighbors[w].append(w ^ mask)
    # Each neighbors[w] now contains w itself (d=0) and all words within distance 5.

    # ---- Step 1: Construct the [17,9,6] QR code (256 words) deterministically ----
    residues = [0, 1, 2, 4, 8, 9, 13, 15, 16]
    gen_row = [0] * N
    for r in residues:
        gen_row[r] = 1
    circ = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        for j in range(N):
            circ[i, j] = gen_row[(j - i) % N]
    G = circ[:9, :]
    g_ints = []
    for i in range(9):
        val = 0
        for j in range(N):
            if G[i, j] == 1:
                val |= (1 << j)
        g_ints.append(val)

    code = []
    for mask in range(256):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                w ^= g_ints[i]
        code.append(w)

    # Verify min weight (linear code min weight == min distance)
    min_w = N + 1
    for w in code:
        if w == 0:
            continue
        wt = w.bit_count()
        if wt < min_w:
            min_w = wt
    if min_w < MIN_DIST:
        # Fallback: greedy from 0
        blocked = bytearray(FULL)
        code = [0]
        for nb in neighbors[0]:
            blocked[nb] = 1
        for w in range(1, FULL):
            if not time_ok():
                break
            if not blocked[w]:
                code.append(w)
                for nb in neighbors[w]:
                    blocked[nb] = 1
        return code

    # ---- Build blocked array from the 256-word linear code ----
    blocked = bytearray(FULL)
    for w in code:
        for nb in neighbors[w]:
            blocked[nb] = 1

    # ---- Step 2: Greedy scan for extra words, weight-ordered ----
    all_words = list(range(FULL))
    all_words.sort(key=lambda x: x.bit_count())
    for w in all_words:
        if not time_ok():
            break
        if not blocked[w]:
            code.append(w)
            for nb in neighbors[w]:
                blocked[nb] = 1

    # ---- Step 3: Local search ----
    code_set = set(code)

    def try_add(w):
        if blocked[w]:
            return False
        code.append(w)
        code_set.add(w)
        for nb in neighbors[w]:
            blocked[nb] = 1
        return True

    def remove(w):
        code.remove(w)
        code_set.discard(w)
        for nb in neighbors[w]:
            blocked[nb] = 0

    def scan_add():
        for w in all_words:
            if not time_ok():
                break
            if not blocked[w]:
                code.append(w)
                code_set.add(w)
                for nb in neighbors[w]:
                    blocked[nb] = 1

    # Pass A: drop-and-re-add in varied orders to escape local optima
    for iteration in range(30):
        if not time_ok():
            break
        if iteration % 3 == 0:
            order = sorted(range(len(code)), key=lambda i: -code[i].bit_count())
        elif iteration % 3 == 1:
            order = list(range(len(code)))
            random.shuffle(order)
        else:
            order = sorted(range(len(code)), key=lambda i: code[i].bit_count())

        removed = []
        for idx in order:
            if not time_ok():
                break
            w = code.pop(idx)
            code_set.discard(w)
            for nb in neighbors[w]:
                blocked[nb] = 0
            removed.append(w)

        for w in removed:
            if not time_ok():
                break
            if not blocked[w]:
                code.append(w)
                code_set.add(w)
                for nb in neighbors[w]:
                    blocked[nb] = 1

        scan_add()

    # Pass B: replace-one-with-two (from inspiration) — split a well-separated word
    for _ in range(30):
        if not time_ok():
            break
        n = len(code)
        if n < 2:
            break
        improved = False
        i = random.randrange(n)
        w_i = code[i]

        # Find closest neighbor distance
        closest_dist = N + 1
        for j in range(n):
            if j == i:
                continue
            d = (w_i ^ code[j]).bit_count()
            if d < closest_dist:
                closest_dist = d
        if closest_dist >= MIN_DIST + 1:
            # Try to split w_i into two words that are mutually far and far from others
            for _ in range(30):
                if not time_ok():
                    break
                pm1 = random.getrandbits(N)
                if pm1.bit_count() > 4:
                    continue
                cand1 = w_i ^ pm1
                if cand1 in code_set or cand1 == w_i:
                    continue
                valid1 = True
                for j in range(n):
                    if j == i:
                        continue
                    if (cand1 ^ code[j]).bit_count() < MIN_DIST:
                        valid1 = False
                        break
                if not valid1:
                    continue
                for _ in range(15):
                    if not time_ok():
                        break
                    pm2 = random.getrandbits(N)
                    if pm2.bit_count() > 4:
                        continue
                    cand2 = w_i ^ pm2
                    if cand2 == cand1 or cand2 in code_set or cand2 == w_i:
                        continue
                    if (cand1 ^ cand2).bit_count() < MIN_DIST:
                        continue
                    valid2 = True
                    for j in range(n):
                        if j == i:
                            continue
                        if (cand2 ^ code[j]).bit_count() < MIN_DIST:
                            valid2 = False
                            break
                    if not valid2:
                        continue
                    # Do the replacement: remove w_i, add cand1 and cand2
                    code[i] = cand1
                    code_set.discard(w_i)
                    code_set.add(cand1)
                    code.append(cand2)
                    code_set.add(cand2)
                    for nb in neighbors[w_i]:
                        blocked[nb] = 0
                    for nb in neighbors[cand1]:
                        blocked[nb] = 1
                    for nb in neighbors[cand2]:
                        blocked[nb] = 1
                    improved = True
                    break
                if improved:
                    break
        if improved:
            scan_add()

    # Final scan to pick up anything newly freed
    scan_add()

    # ---- Final validity check (defensive) ----
    code = list(dict.fromkeys(code))
    n = len(code)
    valid = True
    for a in range(n):
        for b in range(a + 1, n):
            if (code[a] ^ code[b]).bit_count() < MIN_DIST:
                valid = False
                break
        if not valid:
            break
    if not valid:
        # Rebuild a guaranteed-valid subset greedily
        blocked2 = bytearray(FULL)
        clean = []
        for w in code:
            if not blocked2[w]:
                clean.append(w)
                for nb in neighbors[w]:
                    blocked2[nb] = 1
        code = clean

    return code
