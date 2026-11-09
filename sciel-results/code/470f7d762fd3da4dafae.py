import time
import itertools
import numpy as np

def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0          # safety margin vs 120s kill

    def remaining_budget():
        return max(0.0, deadline - time.time())

    # ---- Fast coverage tracking over the full q**n space ----
    total = q ** n
    covered = np.zeros(total, dtype=np.uint8)

    def add_cover(word, delta):
        for d in range(delta + 1):
            if d == 0:
                idx = 0
                for i, v in enumerate(word):
                    idx = idx * q + v
                covered[idx] = 1
                continue
            for pos in range(n):
                for newv in range(q):
                    if newv == word[pos]:
                        continue
                    idx = 0
                    for i in range(n):
                        idx = idx * q + (newv if i == pos else word[i])
                    covered[idx] = 1

    # ---- Seed: direct sum of the ternary tetracode ----
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed = [h + list(tail) for h in tetracode
            for tail in itertools.product(range(3), repeat=n - 4)]
    for w in seed:
        add_cover(w, radius)
    best = [list(w) for w in seed]
    best_size = len(seed)

    # ---- Greedy set-cover: add the word that covers the most uncovered words ----
    while remaining_budget() > 1.0:
        best_gain = 0
        best_word = None
        for a in range(q):
            for b in range(q):
                w = [a, b, (a + b) % q, (a + 2 * b) % q] + [0] * (n - 4)
                for tail in itertools.product(range(q), repeat=n - 4):
                    w[4:] = tail
                    # number of currently uncovered words within radius
                    gain = 0
                    for d in range(radius + 1):
                        if d == 0:
                            idx = 0
                            for i, v in enumerate(w):
                                idx = idx * q + v
                            gain += (1 - covered[idx])
                            continue
                        for pos in range(n):
                            for newv in range(q):
                                if newv == w[pos]:
                                    continue
                                idx = 0
                                for i in range(n):
                                    idx = idx * q + (newv if i == pos else w[i])
                                gain += (1 - covered[idx])
                    if gain > best_gain:
                        best_gain = gain
                        best_word = list(w)
        if best_word is None or best_gain == 0:
            break
        add_cover(best_word, radius)
        best.append(best_word)
        best_size = len(best)
        if covered.all():
            break

    # ---- Local search: drop redundant codewords, then swap one for a better one ----
    def is_covered_by_others(idx, skip):
        for j, w in enumerate(best):
            if j == skip:
                continue
            for d in range(radius + 1):
                if d == 0:
                    wi = 0
                    for i, v in enumerate(w):
                        wi = wi * q + v
                    if wi == idx:
                        return True
                    continue
                for pos in range(n):
                    for newv in range(q):
                        if newv == w[pos]:
                            continue
                        wi = 0
                        for i in range(n):
                            wi = wi * q + (newv if i == pos else w[i])
                        if wi == idx:
                            return True
        return False

    def drop_redundant():
        nonlocal best, best_size
        changed = True
        while changed and remaining_budget() > 0.5:
            changed = False
            for i in range(len(best)):
                w = best[i]
                redundant = True
                for d in range(radius + 1):
                    if d == 0:
                        idx = 0
                        for k, v in enumerate(w):
                            idx = idx * q + v
                        if not is_covered_by_others(idx, i):
                            redundant = False
                            break
                        continue
                    for pos in range(n):
                        for newv in range(q):
                            if newv == w[pos]:
                                continue
                            idx = 0
                            for k in range(n):
                                idx = idx * q + (newv if k == pos else w[k])
                            if not is_covered_by_others(idx, i):
                                redundant = False
                                break
                            if not redundant:
                                break
                        if not redundant:
                            break
                    if not redundant:
                        break
                if redundant:
                    best.pop(i)
                    best_size = len(best)
                    changed = True
                    break

    drop_redundant()

    # ---- Swap: replace a redundant/low-value codeword with a higher-gain one ----
    while remaining_budget() > 0.5:
        improved = False
        for i in range(len(best)):
            # find a candidate word that adds more uncovered coverage than best[i] holds
            for a in range(q):
                for b in range(q):
                    w = [a, b, (a + b) % q, (a + 2 * b) % q] + [0] * (n - 4)
                    for tail in itertools.product(range(q), repeat=n - 4):
                        w[4:] = tail
                        if w == best[i]:
                            continue
                        gain = 0
                        for d in range(radius + 1):
                            if d == 0:
                                idx = 0
                                for k, v in enumerate(w):
                                    idx = idx * q + v
                                gain += (1 - covered[idx])
                                continue
                            for pos in range(n):
                                for newv in range(q):
                                    if newv == w[pos]:
                                        continue
                                    idx = 0
                                    for k in range(n):
                                        idx = idx * q + (newv if k == pos else w[k])
                                    gain += (1 - covered[idx])
                        if gain > 0:
                            # swap: remove i, add w
                            best[i] = w
                            improved = True
                            break
                        if not improved:
                            continue
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
        drop_redundant()

    # ---- Guarantee validity: if anything is uncovered, re-cover greedily ----
    if not covered.all():
        while remaining_budget() > 0.2 and not covered.all():
            best_gain = 0
            best_word = None
            for a in range(q):
                for b in range(q):
                    w = [a, b, (a + b) % q, (a + 2 * b) % q] + [0] * (n - 4)
                    for tail in itertools.product(range(q), repeat=n - 4):
                        w[4:] = tail
                        gain = 0
                        for d in range(radius + 1):
                            if d == 0:
                                idx = 0
                                for k, v in enumerate(w):
                                    idx = idx * q + v
                                gain += (1 - covered[idx])
                                continue
                            for pos in range(n):
                                for newv in range(q):
                                    if newv == w[pos]:
                                        continue
                                    idx = 0
                                    for k in range(n):
                                        idx = idx * q + (newv if k == pos else w[k])
                                    gain += (1 - covered[idx])
                        if gain > best_gain:
                            best_gain = gain
                            best_word = list(w)
            if best_word is None or best_gain == 0:
                break
            add_cover(best_word, radius)
            best.append(best_word)
            best_size = len(best)

    return [list(w) for w in best]
