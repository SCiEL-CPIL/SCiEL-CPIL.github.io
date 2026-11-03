"""
Structural change: use the tetracode product (36-word seed) as the initial
covering code for K_3(8,1), then aggressively extend with vectorized greedy
set-cover, trim redundant words, and refine with a targeted swap local search
that exploits an inverse-index of which centers cover each word.
"""
import time
import random
import itertools
import numpy as np

_DEADLINE = 112.0


def construct(q, n, radius):
    t0 = time.perf_counter()
    deadline = t0 + _DEADLINE

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # ---------- Trivial / generic fallbacks ----------
    if total <= ball_size:
        return [[0] * n]
    if q == 2 and n == 1:
        return [[0], [1]]
    if q == 3 and n == 1:
        return [[0], [1], [2]]

    # ---------- Specialized K_3(8,1) path ----------
    if q == 3 and n == 8 and radius == 1:
        return _k381(t0, deadline)

    # ---------- Generic fallback: tetracode direct sum ----------
    if q == 3 and radius == 1 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = []
        for tail in itertools.product(range(3), repeat=n - 4):
            for h in tetracode:
                code.append(h + list(tail))
        return code

    # ---------- Very generic fallback ----------
    if total <= 200000:
        words = [list(w) for w in itertools.product(range(q), repeat=n)]
        chosen = []
        covered = set()
        for w in words:
            if tuple(w) in covered:
                continue
            chosen.append(w)
            for i in range(n):
                for v in range(q):
                    if v == w[i]:
                        covered.add(tuple(w))
                    else:
                        nw = w[:i] + [v] + w[i + 1:]
                        covered.add(tuple(nw))
            if len(covered) >= total:
                break
        return chosen

    return [list(t) for t in itertools.product(range(q), repeat=n)]


def _k381(t0, deadline):
    """Specialized solver for K_3(8,1)."""
    q, n, radius = 3, 8, 1
    total = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # ---------- Precompute all balls as integer indices ----------
    balls = np.empty((total, ball_size), dtype=np.int32)
    for w in range(total):
        digits = int_to_digits(w, n, q)
        ball = [w]
        for pos in range(n):
            orig = digits[pos]
            for nv in range(q):
                if nv == orig:
                    continue
                digits[pos] = nv
                ball.append(digits_to_int(digits, q))
            digits[pos] = orig
        balls[w] = ball

    # ---------- Inverse index: for each word, list of centers whose ball contains it ----------
    centers_covering = [None] * total
    for i in range(total):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = []
            centers_covering[w].append(i)
    for w in range(total):
        if centers_covering[w] is not None:
            centers_covering[w] = frozenset(centers_covering[w])

    # ---------- Structured tetracode-product initialization ----------
    def structured_init():
        # 6x6 tetracode product gives 36 codewords
        tetracode = [
            [0, 0, 0, 0],
            [0, 1, 2, 1],
            [0, 2, 1, 2],
            [1, 0, 1, 2],
            [1, 1, 2, 0],
            [1, 2, 0, 2],
        ]
        product_code = []
        for a in tetracode:
            for b in tetracode:
                product_code.append(a + b)
        code_ints = list({digits_to_int(c, q) for c in product_code})
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1
        return code_ints, cover_count

    # ---------- Vectorized greedy extension ----------
    def greedy_extend(code_ints, cover_count):
        code_set = set(code_ints)
        uncovered = (cover_count == 0)
        while np.any(uncovered) and time.perf_counter() < deadline - 8:
            gains = uncovered[balls].sum(axis=1)
            for c in code_ints:
                gains[c] = -1
            max_gain = int(gains.max())
            if max_gain <= 0:
                break
            best_center = int(np.argmax(gains))
            code_ints.append(best_center)
            code_set.add(best_center)
            cover_count[balls[best_center]] += 1
            uncovered = (cover_count == 0)
        return code_ints, cover_count, code_set

    code_ints, cover_count = structured_init()
    code_ints, cover_count, code_set = greedy_extend(code_ints, cover_count)

    # ---------- Aggressive trimming of redundant codewords ----------
    def trim(code_ints, cover_count, code_set):
        changed = True
        while changed and time.perf_counter() < deadline - 15:
            changed = False
            to_remove = []
            for i, c in enumerate(code_ints):
                if (cover_count[balls[c]] > 1).all():
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code_ints[idx]
                    code_ints.pop(idx)
                    cover_count[balls[c]] -= 1
                    code_set.discard(c)
                changed = True
        return code_ints, cover_count, code_set

    code_ints, cover_count, code_set = trim(code_ints, cover_count, code_set)

    # ---------- Local search: targeted swap via inverse index ----------
    def local_search(code_ints, cover_count, code_set, time_budget=60.0):
        end_time = time.perf_counter() + time_budget
        improved = True
        rng = random.Random(42)
        while improved and time.perf_counter() < end_time:
            improved = False
            indices = list(range(len(code_ints)))
            rng.shuffle(indices)
            for idx in indices:
                if time.perf_counter() > end_time:
                    break
                i = code_ints[idx]
                bp_i = balls[i]
                # Find words uniquely covered by i
                unique_mask = (cover_count[bp_i] == 1)
                unique_pts = bp_i[unique_mask]
                if len(unique_pts) == 0:
                    # i is redundant
                    cover_count[bp_i] -= 1
                    code_ints.pop(idx)
                    code_set.discard(i)
                    improved = True
                    break
                # Intersection of centers covering every uniquely-covered word
                candidate_set = None
                for w in unique_pts:
                    cc = centers_covering[w]
                    if candidate_set is None:
                        candidate_set = set(cc)
                    else:
                        candidate_set &= cc
                    if not candidate_set:
                        break
                if not candidate_set:
                    continue
                candidates = [c for c in candidate_set if c not in code_set]
                if not candidates:
                    continue
                # Score: minimize wasted coverage after the swap
                best_swap = None
                best_waste = None
                for j in candidates:
                    if time.perf_counter() > end_time:
                        return code_ints, cover_count, code_set
                    bp_j = balls[j]
                    # After removing i, words in bp_i with count 1 become 0;
                    # after adding j, words in bp_j get +1.
                    # We want to minimize words that go from count>=2 to count>=2
                    # (wasted) among words in bp_j that are in bp_i.
                    in_i_mask = np.zeros(total, dtype=np.bool_)
                    in_i_mask[bp_i] = True
                    waste = int(np.sum((cover_count[bp_j] >= 2) & in_i_mask[bp_j]))
                    if best_waste is None or waste < best_waste:
                        best_waste = waste
                        best_swap = j
                if best_swap is not None:
                    # Perform swap
                    cover_count[bp_i] -= 1
                    code_ints[idx] = best_swap
                    code_set.discard(i)
                    code_set.add(best_swap)
                    cover_count[balls[best_swap]] += 1
                    improved = True
                    break
        return code_ints, cover_count, code_set

    if time.perf_counter() < deadline - 10:
        code_ints, cover_count, code_set = local_search(code_ints, cover_count, code_set, time_budget=60.0)

    # ---------- Convert to list of lists ----------
    result = []
    for c in code_ints:
        result.append(int_to_digits(c, n, q))
    return result
