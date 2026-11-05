import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130  # safety margin under 150s kill

    N = q ** n
    n2 = n * n

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute masks: mask[i][p] = q^p so that idx + mask[i][p]*d shifts digit i by d.
    mask = np.empty((n, n), dtype=np.int64)
    for i in range(n):
        for p in range(n):
            mask[i][p] = (q ** p) * (q ** (n - 1 - i))

    # Precompute neighbor index lists for all words.
    neighbors = [None] * N
    for i in range(N):
        lst = [i]
        for p in range(n):
            digit = (i // mask[i][p]) % q
            for d in range(q):
                if d != digit:
                    lst.append(i + mask[i][p] * (d - digit))
        neighbors[i] = lst

    def greedy_cover(order2=False, randomize=False, rng=None):
        """Greedy set cover. Returns list of chosen indices."""
        gain = np.empty(N, dtype=np.int64)
        gain[:] = len(neighbors[0])
        code = []
        while True:
            if order2:
                # gain2[i] = number of uncovered words at Hamming distance 2 from i
                gain2 = np.zeros(N, dtype=np.int64)
                for v in range(N):
                    if gain[v] > 0:
                        for u in neighbors[v]:
                            gain2[u] += 1
                # score = 10000*gain + gain2 to prioritize 1st-neighborhood coverage
                score = gain * 10000 + gain2
            else:
                score = gain
            if randomize:
                # add small random noise to break ties / explore
                noise = rng.random(N) * 0.5
                score = score + noise
            best_idx = int(np.argmax(score))
            if score[best_idx] <= 0:
                break
            code.append(best_idx)
            for v in neighbors[best_idx]:
                if gain[v] > 0:
                    gain[v] = 0
                    for u in neighbors[v]:
                        gain[u] -= 1
            if time.time() > deadline:
                return code
        return code

    def peel(code):
        """Remove non-essential codewords (those whose removal doesn't break coverage)."""
        cover_count = np.zeros(N, dtype=np.int64)
        for c in code:
            for v in neighbors[c]:
                cover_count[v] += 1
        changed = True
        while changed:
            changed = False
            to_remove = []
            for c in code:
                # c is essential if it's the only codeword covering some word
                essential = False
                for v in neighbors[c]:
                    if cover_count[v] == 1:
                        essential = True
                        break
                if not essential:
                    to_remove.append(c)
            if to_remove:
                for c in to_remove:
                    code.remove(c)
                    for v in neighbors[c]:
                        cover_count[v] -= 1
                changed = True
        return code

    # Phase 1: deterministic greedy (order-1)
    code1 = greedy_cover(order2=False)
    code1 = peel(code1)
    best_code = code1
    best_len = len(code1)

    # Phase 2: deterministic greedy (order-2)
    if time.time() < deadline - 30:
        code2 = greedy_cover(order2=True)
        code2 = peel(code2)
        if len(code2) < best_len:
            best_code = code2
            best_len = len(code2)

    # Phase 3: randomized restarts
    rng = random.Random(42)
    while time.time() < deadline - 15:
        code_r = greedy_cover(order2=False, randomize=True, rng=rng)
        code_r = peel(code_r)
        if len(code_r) < best_len:
            best_code = code_r
            best_len = len(code_r)

    # Phase 4: try order-2 randomized
    while time.time() < deadline - 5:
        code_r = greedy_cover(order2=True, randomize=True, rng=rng)
        code_r = peel(code_r)
        if len(code_r) < best_len:
            best_code = code_r
            best_len = len(code_r)

    return [idx_to_word(idx) for idx in best_code]
