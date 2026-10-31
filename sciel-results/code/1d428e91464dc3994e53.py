"""Greedy set-cover construction for covering codes K_q(n, radius)."""
import itertools


def _gen_radius_ball(q, n, radius):
    """Yield all words of length n over {0..q-1} within Hamming distance <= radius
    of the all-zeros word."""
    if radius >= n:
        for word in itertools.product(range(q), repeat=n):
            yield word
        return
    for d in range(radius + 1):
        for pos in itertools.combinations(range(n), d):
            for vals in itertools.product(range(1, q), repeat=d):
                w = [0] * n
                for p, v in zip(pos, vals):
                    w[p] = v
                yield tuple(w)


def construct(q, n, radius):
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_index = {w: i for i, w in enumerate(all_words)}

    offsets = list(_gen_radius_ball(q, n, radius))

    # cov[c] = list of word-indices covered by candidate c.
    cov = []
    for c in all_words:
        lst = []
        for o in offsets:
            w = tuple((c[i] + o[i]) % q for i in range(n))
            lst.append(word_index[w])
        cov.append(lst)

    # inverted[wi] = list of candidate indices that cover word wi.
    inverted = [[] for _ in range(total)]
    for c in range(total):
        for wi in cov[c]:
            inverted[wi].append(c)

    uncovered = [1] * total
    gain = [0] * total
    for c in range(total):
        s = 0
        for wi in cov[c]:
            s += uncovered[wi]
        gain[c] = s

    code = []
    for _ in range(total):
        best = -1
        best_gain = -1
        for c in range(total):
            if gain[c] > best_gain:
                best_gain = gain[c]
                best = c
        if best_gain <= 0:
            break
        code.append(list(all_words[best]))
        for wi in cov[best]:
            if uncovered[wi]:
                uncovered[wi] = 0
                for c in inverted[wi]:
                    gain[c] -= 1
    return code
