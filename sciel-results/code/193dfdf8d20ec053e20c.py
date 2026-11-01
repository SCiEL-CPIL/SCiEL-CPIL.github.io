"""Fresh design: fast greedy cover + aggressive local-search reduction (drop/swap/SA)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _time_left(deadline, margin=1.5):
    return (deadline - margin) - time.time()


def _base_cover(q, n, radius):
    """Greedy covering-code construction. Returns (codes list, covered bool array)."""
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    codes = []
    best = 0
    while best > 0 and len(codes) < total:
        cands = np.nonzero(~covered)[0]
        if cands.size == 0:
            break
        # pick the best candidate (covers most uncovered words)
        best = 0
        best_i = 0
        for i in range(len(cands)):
            c = cands[i]
            if covered[c]:
                continue
            s = 0
            for r in range(radius + 1):
                s += int(np.count_nonzero(~covered[c + r * total]))
                if s <= best:
                    break
            if s > best:
                best = s
                best_i = i
        c = int(cands[best_i])
        codes.append(c)
        for r in range(radius + 1):
            covered[c + r * total] = True
    return codes, covered


def _build_coverings(codes, q, n, radius):
    """For each codeword, a bool array of words within Hamming radius."""
    total = q ** n
    m = len(codes)
    coverings = np.zeros((m, total), dtype=bool)
    for i in range(m):
        c = codes[i]
        for r in range(radius + 1):
            coverings[i, c + r * total] = True
    return coverings


def _reduce(codes, coverings, covered, q, n, radius, deadline):
    """Local search: drop redundant, swap, then simulated-annealing accept."""
    m = len(codes)
    total = q ** n
    while True:
        if _time_left(deadline) <= 0:
            break
        # 1) drop redundant codewords
        changed = False
        for i in range(m):
            if _time_left(deadline) <= 0:
                break
            if np.all(covered):
                break
            alt = covered.copy()
            alt[coverings[i]] = False
            if np.all(alt):
                covered[coverings[i]] = False
                codes.pop(i)
                coverings = np.delete(coverings, i, axis=0)
                m -= 1
                changed = True
                break
        if not changed:
            break
    # 2) swap: replace a codeword with an uncovered word if it helps
    improved = True
    while improved and _time_left(deadline) > 0:
        improved = False
        for i in range(len(codes)):
            if _time_left(deadline) <= 0:
                break
            alt = covered.copy()
            alt[coverings[i]] = False
            if np.all(alt):
                continue
            # find best replacement
            best_gain = 0
            best_c = -1
            for c in np.nonzero(~covered)[0]:
                gain = int(np.count_nonzero(~alt[coverings[c]]))
                if gain > best_gain:
                    best_gain = gain
                    best_c = int(c)
            if best_c >= 0 and best_gain > 0:
                codes[i] = best_c
                coverings[i] = np.zeros(total, dtype=bool)
                for r in range(radius + 1):
                    coverings[i, best_c + r * total] = True
                covered = np.any(coverings, axis=0)
                improved = True
                break
    # 3) simulated annealing on swaps
    temp = 20.0
    step = 0
    while temp > 0.1 and _time_left(deadline) > 0:
        step += 1
        if step % 200 == 0:
            temp *= 0.85
        i = int(np.random.randint(len(codes)))
        c = int(np.random.choice(np.nonzero(~covered)[0])) if (~covered).any() else codes[i]
        alt = covered.copy()
        alt[coverings[i]] = False
        alt[coverings[c]] = True
        if np.all(alt):
            codes[i] = c
            coverings[i] = np.zeros(total, dtype=bool)
            for r in range(radius + 1):
                coverings[i, c + r * total] = True
            covered = alt
        else:
            delta = int(np.count_nonzero(~alt)) - int(np.count_nonzero(~covered))
            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                codes[i] = c
                coverings[i] = np.zeros(total, dtype=bool)
                for r in range(radius + 1):
                    coverings[i, c + r * total] = True
                covered = alt
    # final drop pass
    m = len(codes)
    for _ in range(3):
        changed = False
        for i in range(m):
            if _time_left(deadline) <= 0:
                break
            alt = covered.copy()
            alt[coverings[i]] = False
            if np.all(alt):
                covered[coverings[i]] = False
                codes.pop(i)
                coverings = np.delete(coverings, i, axis=0)
                m -= 1
                changed = True
                break
        if not changed:
            break
    return codes


def construct(q, n, radius):
    deadline = time.time() + 148.0
    codes, covered = _base_cover(q, n, radius)
    if _time_left(deadline) <= 0:
        return [[int(x) for x in np.array_divmod(c, [q ** (n - 1 - i) for i in range(n)])] for c in codes]
    coverings = _build_coverings(codes, q, n, radius)
    codes = _reduce(codes, coverings, covered, q, n, radius, deadline)
    # convert to lists of ints
    out = []
    for c in codes:
        w = []
        v = c
        for i in range(n - 1, -1, -1):
            w.append(v % q)
            v //= q
        out.append(w)
    return out
# EVOLVE-BLOCK-END
