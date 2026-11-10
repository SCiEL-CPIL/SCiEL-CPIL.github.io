"""Greedy covering-code search for K_q(n, r) with an explicit deadline guard."""
import itertools
import time


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin below the 150s kill

    # ---- small / trivial cases handled exactly ----
    total = q ** n
    if n == 0:
        return [[]]
    if radius >= n:
        return [[0] * n]
    if n == 1:
        return [[i] for i in range(q)]

    # ---- build the universe and the coverage sets ----
    # Universe = all words 0..total-1.  For each candidate word c, its coverage
    # set is {c} union {words at Hamming distance <= radius from c}.
    # For radius == 1: coverage = {c} + {c with one coordinate changed}.
    if radius == 1:
        covers = []
        for c in range(total):
            s = {c}
            for pos in range(n):
                base = c - (c % q) ** pos  # value at pos
                for v in range(q):
                    if v == base:
                        continue
                    s.add(c + (v - base) * (q ** pos))
            covers.append(s)
    else:
        # generic (slower) coverage via Hamming distance
        words = list(itertools.product(range(q), repeat=n))
        idx = {w: i for i, w in enumerate(words)}
        covers = []
        for c in range(total):
            cw = words[c]
            s = {c}
            for pos in range(n):
                for v in range(q):
                    if v == cw[pos]:
                        continue
                    lst = list(cw)
                    lst[pos] = v
                    s.add(idx[tuple(lst)])
            covers.append(s)

    # ---- greedy set cover, with an order that prefers low-weight words ----
    # (low-weight words are more "central" and often cover more uncovered words)
    weight = [sum(w) for w in (
        [0] * 0 if radius != 1 else
        [sum(cw) for cw in itertools.product(range(q), repeat=n)]
    )] if radius == 1 else [0] * total

    order = list(range(total))
    order.sort(key=lambda c: (weight[c], c))

    remaining = total
    chosen = []
    covered = bytearray(total)  # 1 if word is covered

    # precompute the candidate list once
    cand_list = order

    while remaining > 0:
        # pick the candidate covering the most currently-uncovered words
        best = -1
        best_gain = 0
        for c in cand_list:
            cov = covers[c]
            gain = 0
            for w in cov:
                if not covered[w]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best = c
                if best_gain == remaining:
                    break
        if best_gain == 0:
            # nothing left is uncovered (shouldn't happen), bail out
            break
        chosen.append(best)
        cov = covers[best]
        for w in cov:
            if not covered[w]:
                covered[w] = 1
                remaining -= 1
        # remove best from candidates to avoid re-evaluating
        try:
            cand_list.remove(best)
        except ValueError:
            pass
        if time.time() > deadline:
            break

    # ---- convert chosen indices back to codewords ----
    result = []
    for c in chosen:
        word = []
        t = c
        for pos in range(n):
            word.append(t % q)
            t //= q
        word.reverse()
        result.append(word)

    # ---- verify coverage; if incomplete (deadline), extend greedily ----
    if remaining > 0:
        # add words until covered (worst case, add the uncovered words themselves)
        for w in range(total):
            if not covered[w]:
                # add w as a codeword and mark its coverage
                c = w
                cov = covers[c]
                for x in cov:
                    if not covered[x]:
                        covered[x] = 1
                        remaining -= 1
                result.append([
                    (w // q ** pos) % q for pos in range(n)
                ])
                if remaining == 0:
                    break

    return result
