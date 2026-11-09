"""Reciprocal weight maximization for 3-AP-free subsets of [1,256]."""
import time
import random
from fractions import Fraction

N = 256
DEADLINE = 85.0  # safety margin before 90s kill

def _has_3ap_bitset(bits):
    for d in range(1, N // 2 + 1):
        if (bits & (bits >> d) & (bits >> 2 * d)) != 0:
            return True
    return False

def _can_add(bits, x):
    """Check if adding element x (1-indexed) keeps set 3-AP-free."""
    for d in range(1, (N - x + 1) // 2 + 1):
        if (bits >> (x + d - 1)) & 1 and (bits >> (x + 2*d - 1)) & 1:
            return False
    for d in range(1, x):
        if (bits >> (x - d - 1)) & 1 and (bits >> (x + d - 1)) & 1:
            return False
    for d in range(1, (x + 1) // 2):
        if (bits >> (x - 2*d - 1)) & 1 and (bits >> (x - d - 1)) & 1:
            return False
    return True

def _swap_valid(bits, x_remove, x_add):
    """Check if swapping x_remove with x_add keeps 3-AP-free."""
    for d in range(1, (N - x_add + 1) // 2 + 1):
        a = x_add + d
        c = x_add + 2 * d
        if a == x_remove or c == x_remove:
            continue
        if (bits >> (a - 1)) & 1 and (bits >> (c - 1)) & 1:
            return False
    for d in range(1, x_add):
        a = x_add - d
        c = x_add + d
        if a == x_remove or c == x_remove:
            continue
        if (bits >> (a - 1)) & 1 and (bits >> (c - 1)) & 1:
            return False
    for d in range(1, (x_add + 1) // 2):
        a = x_add - 2 * d
        b = x_add - d
        if a == x_remove or b == x_remove:
            continue
        if (bits >> (a - 1)) & 1 and (bits >> (b - 1)) & 1:
            return False
    return True

def _reciprocal_sum_float(bits):
    total = 0.0
    for i in range(N):
        if (bits >> i) & 1:
            total += 1.0 / (i + 1)
    return total

def _bits_to_list(bits):
    return [i + 1 for i in range(N) if (bits >> i) & 1]

def _list_to_bits(lst):
    b = 0
    for x in lst:
        b |= (1 << (x - 1))
    return b

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    def _now():
        return time.time()

    def _left():
        return deadline - _now()

    # ---------- Phase 1: Greedy initialization ----------
    best_bits = 0
    best_score = 0.0

    # Greedy: smallest x first (maximizes 1/x)
    bits = 0
    for x in range(1, N + 1):
        if _can_add(bits, x):
            bits |= (1 << (x - 1))
    score = _reciprocal_sum_float(bits)
    if score > best_score:
        best_score = score
        best_bits = bits

    # A few randomized greedy restarts (bias toward small numbers)
    rng = random.Random(12345)
    for _ in range(6):
        if _left() < 5.0:
            break
        order = list(range(1, N + 1))
        # Partial shuffle: keep small numbers roughly in place
        for i in range(len(order)):
            if rng.random() < 0.25:
                j = i + rng.randint(1, len(order) - i - 1)
                order[i], order[j] = order[j], order[i]
        bits = 0
        for x in order:
            if _can_add(bits, x):
                bits |= (1 << (x - 1))
        score = _reciprocal_sum_float(bits)
        if score > best_score:
            best_score = score
            best_bits = bits

    # ---------- Phase 2: Simulated annealing ----------
    if _left() > 10.0:
        current_bits = best_bits
        current_score = best_score
        best_bits_sa = best_bits
        best_score_sa = best_score

        T_start = 0.02
        T_end = 0.0005
        alpha = 0.997

        current_T = T_start
        while current_T > T_end and _left() > 8.0:
            cur_list = _bits_to_list(current_bits)
            if len(cur_list) < 2:
                break

            idx_remove = rng.randint(0, len(cur_list) - 1)
            x_remove = cur_list[idx_remove]

            if rng.random() < 0.7:
                candidates = [x for x in range(1, N + 1) if not ((current_bits >> (x - 1)) & 1)]
                if not candidates:
                    continue
                weights = [1.0 / x for x in candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                x_add = candidates[-1]
                for c, w in zip(candidates, weights):
                    cum += w
                    if r <= cum:
                        x_add = c
                        break
            else:
                candidates = [x for x in range(1, N + 1) if not ((current_bits >> (x - 1)) & 1)]
                if not candidates:
                    continue
                x_add = candidates[rng.randint(0, len(candidates) - 1)]

            if x_remove == x_add:
                continue

            if not _swap_valid(current_bits, x_remove, x_add):
                continue

            new_bits = (current_bits & ~(1 << (x_remove - 1))) | (1 << (x_add - 1))
            new_score = current_score - 1.0 / x_remove + 1.0 / x_add
            delta = new_score - current_score

            if delta > 0 or rng.random() < pow(2.718281828, delta / current_T):
                current_bits = new_bits
                current_score = new_score
                if current_score > best_score_sa:
                    best_score_sa = current_score
                    best_bits_sa = current_bits

            current_T *= alpha

        if best_score_sa > best_score:
            best_bits = best_bits_sa
            best_score = best_score_sa

    # ---------- Phase 3: Local search (additions, swaps, 2-for-1, 1-for-2) ----------
    improved = True
    while improved and _left() > 5.0:
        improved = False
        cur_list = _bits_to_list(best_bits)

        # Phase 3a: Try adding each unselected number
        for x in range(1, N + 1):
            if _left() < 5.0:
                break
            if (best_bits >> (x - 1)) & 1:
                continue
            if _can_add(best_bits, x):
                best_bits |= (1 << (x - 1))
                best_score += 1.0 / x
                improved = True

        # Phase 3b: Try single swaps (remove y, add x)
        cur_list = _bits_to_list(best_bits)
        for x_remove in cur_list:
            if _left() < 5.0:
                break
            for x_add in range(1, N + 1):
                if _left() < 5.0:
                    break
                if x_add == x_remove or ((best_bits >> (x_add - 1)) & 1):
                    continue
                if not _swap_valid(best_bits, x_remove, x_add):
                    continue
                new_score = best_score - 1.0 / x_remove + 1.0 / x_add
                if new_score > best_score:
                    best_bits = (best_bits & ~(1 << (x_remove - 1))) | (1 << (x_add - 1))
                    best_score = new_score
                    improved = True
                    break

        # Phase 3c: Try 2-for-1 swaps (remove y1, y2, add x)
        if _left() > 15.0:
            cur_list = _bits_to_list(best_bits)
            for i, y1 in enumerate(cur_list):
                if _left() < 15.0:
                    break
                for y2 in cur_list[i+1:]:
                    if _left() < 15.0:
                        break
                    for x in range(1, N + 1):
                        if _left() < 15.0:
                            break
                        if (best_bits >> (x - 1)) & 1:
                            continue
                        if x == y1 or x == y2:
                            continue
                        # Check if removing y1, y2 and adding x is valid
                        trial = (best_bits & ~(1 << (y1 - 1)) & ~(1 << (y2 - 1))) | (1 << (x - 1))
                        # Verify 3-AP-free (full check since two elements removed)
                        if _has_3ap_bitset(trial):
                            continue
                        new_score = best_score - 1.0 / y1 - 1.0 / y2 + 1.0 / x
                        if new_score > best_score:
                            best_bits = trial
                            best_score = new_score
                            improved = True
                            break

        # Phase 3d: Try 1-for-2 swaps (remove y, add x1, x2)
        if _left() > 20.0:
            cur_list = _bits_to_list(best_bits)
            for y in cur_list:
                if _left() < 20.0:
                    break
                for x1 in range(1, N + 1):
                    if _left() < 20.0:
                        break
                    if (best_bits >> (x1 - 1)) & 1 or x1 == y:
                        continue
                    for x2 in range(x1 + 1, N + 1):
                        if _left() < 20.0:
                            break
                        if (best_bits >> (x2 - 1)) & 1 or x2 == y:
                            continue
                        trial = (best_bits & ~(1 << (y - 1))) | (1 << (x1 - 1)) | (1 << (x2 - 1))
                        if _has_3ap_bitset(trial):
                            continue
                        new_score = best_score - 1.0 / y + 1.0 / x1 + 1.0 / x2
                        if new_score > best_score:
                            best_bits = trial
                            best_score = new_score
                            improved = True
                            break

    # ---------- Phase 4: Final polish ----------
    for x in range(1, N + 1):
        if _left() < 1.0:
            break
        if (best_bits >> (x - 1)) & 1:
            continue
        if _can_add(best_bits, x):
            best_bits |= (1 << (x - 1))
            best_score += 1.0 / x

    result = _bits_to_list(best_bits)
    if not result:
        return [1]
    return result
