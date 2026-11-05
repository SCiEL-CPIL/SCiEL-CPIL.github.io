"""Ternary covering code K_3(7,1): bitset-accelerated greedy + local search."""
import time
import itertools
import random
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115.0  # safety margin before 150s wall-clock

    # Fallback for non-(3,7,1): simple product construction (valid, not optimal)
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q]
                     for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(q), repeat=n - 4)]

    N = 3 ** n  # 2187
    # all_words[i] = digits of base-3 representation of i
    all_words = np.zeros((N, n), dtype=np.int8)
    for idx in range(N):
        t = idx
        for pos in range(n - 1, -1, -1):
            all_words[idx, pos] = t % 3
            t //= 3

    # Build ball bitsets: ball_bits[i] has bit j set iff dist(i,j) <= radius
    ball_bits = [0] * N
    powers = [3 ** (n - 1 - pos) for pos in range(n)]
    for i in range(N):
        bits = 1 << i
        w = all_words[i]
        for pos in range(n):
            base = powers[pos]
            for val in range(3):
                if val == w[pos]:
                    continue
                bits |= 1 << (i + (val - int(w[pos])) * base)
        ball_bits[i] = bits

    # Precompute ball member lists (sorted) for candidate enumeration
    ball_members = []
    for i in range(N):
        b = ball_bits[i]
        members = []
        while b:
            low = b & (-b)
            members.append(low.bit_length() - 1)
            b ^= low
        ball_members.append(members)

    FULL_MASK = (1 << N) - 1

    def is_valid(code):
        if not code:
            return False
        union = 0
        for c in code:
            union |= ball_bits[c]
        return union == FULL_MASK

    def greedy_fill(code, covered_mask):
        """Add codewords until every word is covered. covered_mask is a Python int."""
        code = list(code)
        uncovered = FULL_MASK ^ covered_mask
        while uncovered:
            if time.time() > deadline - 5:
                break
            # Find word covering most uncovered positions
            best = -1
            best_gain = 0
            for i in range(N):
                gain = (ball_bits[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best = i
                    if gain == uncovered.bit_count():
                        break  # perfect: covers all uncovered
            if best < 0 or best_gain == 0:
                break
            code.append(best)
            uncovered &= ~ball_bits[best]
        return code, (FULL_MASK ^ uncovered)

    def local_search(code, time_budget=4.0):
        """Bitset-based deletion + swap local search."""
        code = list(code)
        cs = set(code)
        t0 = time.time()
        improved = True
        while improved and (time.time() - t0) < time_budget:
            improved = False

            # ---- Deletion pass: drop any codeword whose ball is fully covered by others ----
            to_remove = []
            for c in code:
                others_union = 0
                for d in code:
                    if d != c:
                        others_union |= ball_bits[d]
                if (ball_bits[c] & ~others_union) == 0:
                    to_remove.append(c)
            if to_remove:
                improved = True
                for c in to_remove:
                    code.remove(c)
                    cs.discard(c)

            # ---- Swap pass: relocate codewords to reduce critical points ----
            if (time.time() - t0) > time_budget - 0.5:
                break
            shuffled = list(code)
            random.shuffle(shuffled)
            for c in shuffled:
                if (time.time() - t0) > time_budget - 0.5:
                    break
                if c not in cs:
                    continue
                # Find words uniquely covered by c (critical points)
                others_union = 0
                for d in code:
                    if d != c:
                        others_union |= ball_bits[d]
                crit_mask = ball_bits[c] & ~others_union
                if crit_mask == 0:
                    continue  # would be removed by deletion
                if crit_mask.bit_count() > 4:
                    continue  # too many critical points to cover with one new position
                # Find candidate positions that cover ALL critical points
                cand_mask = FULL_MASK
                tmp = crit_mask
                while tmp:
                    low = tmp & (-tmp)
                    p = low.bit_length() - 1
                    cand_mask &= ball_bits[p]
                    tmp ^= low
                cand_mask &= ~0  # clear code members
                for d in cs:
                    cand_mask &= ~(1 << d)
                if cand_mask == 0:
                    continue
                # Evaluate candidates: score = new coverage - lost double-coverage
                best_w = -1
                best_score = 0
                tmp = cand_mask
                while tmp:
                    low = tmp & (-tmp)
                    w = low.bit_length() - 1
                    tmp ^= low
                    w_ball = ball_bits[w]
                    c_ball = ball_bits[c]
                    # Gain: words in w_ball that are critically covered (only by c)
                    gain = (w_ball & crit_mask).bit_count()
                    # Loss: words in c_ball that are double-covered and not in w_ball
                    loss = (c_ball & ~w_ball & others_union).bit_count()
                    score = gain - loss
                    if score > best_score:
                        best_score = score
                        best_w = w
                if best_w >= 0 and best_score > 0:
                    code.remove(c)
                    cs.discard(c)
                    code.append(best_w)
                    cs.add(best_w)
                    improved = True
        return code

    # ---- Initial greedy construction ----
    code, _ = greedy_fill([], 0)
    code = local_search(code, time_budget=3.0)
    if not is_valid(code):
        code, _ = greedy_fill([], 0)
        code = local_search(code, time_budget=3.0)
    best = code if is_valid(code) else []

    # ---- Kick / re-greedy / local search, keep best-so-far ----
    while time.time() < deadline - 10:
        cur = list(best)
        k = max(1, min(5, len(cur) // 5))
        to_remove = set(random.sample(cur, k))
        kicked = [c for c in cur if c not in to_remove]
        covered_mask = 0
        for c in kicked:
            covered_mask |= ball_bits[c]
        new_code, _ = greedy_fill(kicked, covered_mask)
        new_code = local_search(new_code, time_budget=3.0)
        if is_valid(new_code) and (not best or len(new_code) < len(best)):
            best = new_code

    if not best or not is_valid(best):
        code, _ = greedy_fill([], 0)
        code = local_search(code, time_budget=3.0)
        best = code

    # Final polish pass
    if time.time() < deadline - 3:
        final_code = local_search(best, time_budget=4.0)
        if is_valid(final_code) and len(final_code) < len(best):
            best = final_code

    if not is_valid(best):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(all_words[c]) for c in best]
