"""Combined program: vectorized greedy + MIS-style min-degree removal + inverse-neighborhood swap local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before the 150s wall-clock kill

    # The concrete task: ternary covering code K_3(8,1).
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback: direct sum of the ternary tetracode (perfect code, radius 1) with
    # every word on the remaining coordinates -> 9 * 3**(n-4) words.
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]

    # Generic fallback (small spaces only).
    if q ** n <= 200000:
        total = q ** n
        weights = [q ** (n - 1 - i) for i in range(n)]

        def neighbors(w):
            out = [w]
            for pos in range(n):
                wt = weights[pos]
                digit = (w // wt) % q
                for nd in range(q):
                    if nd != digit:
                        out.append(w + (nd - digit) * wt)
            return out

        nb = [neighbors(w) for w in range(total)]
        cov = np.zeros(total, dtype=np.int32)
        sel = []
        for w in range(total):
            if cov[w] == 0:
                sel.append(w)
                cov[np.array(nb[w])] += 1
        return [[(w // weights[i]) % q for i in range(n)] for w in sel]

    return []


def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Precompute the 17-word ball (center + 16 neighbors) for every word.
    balls = np.zeros((total, 17), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            wt = weights[pos]
            digit = (w // wt) % 3
            for nd in range(q):
                if nd == digit:
                    continue
                balls[w, idx] = w + (nd - digit) * wt
                idx += 1

    # Ball membership: ball_membership[c, w] = True iff w is in ball(c).
    # Because Hamming distance is symmetric this doubles as the inverse
    # neighborhood: ball_membership[:, w] is the set of codewords covering w.
    ball_membership = np.zeros((total, total), dtype=bool)
    for c in range(total):
        ball_membership[c, balls[c]] = True

    best_size = total
    best_set = None

    def greedy_init(best_size):
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while uncovered_mask.any() and time.time() < deadline - 6.0:
            # Early exit: this run cannot beat the incumbent.
            if len(selected) >= best_size:
                break
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = counts.max()
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            pick = candidates[np.random.randint(len(candidates))]
            selected.append(int(pick))
            cover_count[balls[pick]] += 1
            uncovered_mask = (cover_count == 0)
        return selected

    def local_search(codewords):
        current = list(dict.fromkeys(codewords))  # dedupe, keep order
        if not current:
            return []
        current_set = set(current)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in current:
            cover_count[balls[c]] += 1

        improved = True
        while improved and time.time() < deadline - 6.0:
            improved = False
            random.shuffle(current)
            for c in current:
                if time.time() > deadline - 6.0:
                    break
                b = balls[c]
                U_c = b[cover_count[b] == 1]  # critical words covered only by c
                if U_c.size == 0:
                    # Removable (MIS-style min-degree removal).
                    current_set.discard(c)
                    current.remove(c)
                    cover_count[b] -= 1
                    improved = True
                    break
                # Swap: find a c' that covers every critical word of c.
                # Intersect the inverse-neighborhood columns over U_c.
                candidate_mask = ball_membership[:, U_c[0]].copy()
                for w in U_c[1:]:
                    candidate_mask &= ball_membership[:, w]
                candidate_mask[c] = False
                valid = np.where(candidate_mask)[0]
                if valid.size > 0:
                    c_prime = int(valid[np.random.randint(len(valid))])
                    current_set.discard(c)
                    current.remove(c)
                    cover_count[b] -= 1
                    bp = balls[c_prime]
                    current_set.add(c_prime)
                    current.append(c_prime)
                    cover_count[bp] += 1
                    improved = True
                    break
        return list(current_set)

    # Fixed-seed restarts for reproducibility.
    seed = 42
    while time.time() < deadline - 8.0:
        random.seed(seed)
        seed += 1
        selected = greedy_init(best_size)
        if not selected:
            continue
        if len(selected) >= best_size:
            continue
        refined = local_search(selected)
        if refined and len(refined) < best_size:
            best_size = len(refined)
            best_set = refined

    if best_set is None:
        best_set = list(range(total))

    result = []
    for w in best_set:
        digits = []
        t = w
        for _ in range(n):
            digits.append(t % 3)
            t //= 3
        digits.reverse()
        result.append(digits)
    return result
# EVOLVE-BLOCK-END
