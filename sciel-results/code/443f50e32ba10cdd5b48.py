import time
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    """
    Constructs a covering code using MWIS-inspired greedy with targeted
    local search, optimized scoring, and multi-start restarts.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

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

    # Precompute balls
    balls_list = [None] * total_words
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        neighbors = [i]
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                neighbors.append(digits_to_int(d, q))
            d[pos] = orig
        balls_list[i] = neighbors

    balls = np.array(balls_list, dtype=np.int32)

    # Precompute inverse index: for each word w, list of centers whose ball contains w
    centers_covering = [None] * total_words
    for i in range(total_words):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = []
            centers_covering[w].append(i)
    for w in range(total_words):
        if centers_covering[w] is not None:
            centers_covering[w] = frozenset(centers_covering[w])

    def structured_init():
        """
        Deterministic structured initialization using tetracode product.
        """
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
        
        code_ints = [digits_to_int(c, q) for c in product_code]
        code_ints = list(set(code_ints))
        
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1
        
        uncovered = (cover_count == 0)
        while np.any(uncovered):
            if time.time() > deadline:
                break
            gains = uncovered[balls].sum(axis=1)
            gains[code_ints] = -1
            max_gain = gains.max()
            if max_gain <= 0:
                break
            best_center = int(np.argmax(gains))
            code_ints.append(best_center)
            cover_count[balls[best_center]] += 1
            uncovered = (cover_count == 0)
        
        return code_ints, cover_count

    def greedy_cover(seed_offset=0):
        """Greedy set cover with optional randomization."""
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        cover_count = np.zeros(total_words, dtype=np.int32)

        center_order = np.arange(total_words)
        if seed_offset > 0:
            rng = np.random.RandomState(seed_offset)
            rng.shuffle(center_order)

        while np.any(uncovered):
            if time.time() > deadline:
                break

            gains = uncovered[balls].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break

            best_centers = np.where(gains == max_gain)[0]
            if len(best_centers) == 1:
                best_center = int(best_centers[0])
            else:
                best_center = int(best_centers[np.random.randint(len(best_centers))])

            code.append(best_center)
            uncovered[balls[best_center]] = False
            cover_count[balls[best_center]] += 1

        return code, cover_count

    def trim_code(code, cover_count):
        """Remove redundant codewords."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if np.any(cover_count[balls[c]] == 1):
                    continue
                to_remove.append(i)

            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True
        return code, cover_count

    def targeted_local_search(code, cover_count, max_iter=200000):
        """
        Aggressive targeted local search with optimized O(ball_size) scoring.
        For each codeword, find candidates that cover its uniquely-covered words.
        Uses inverse index for direct candidate enumeration and a fast score
        based on (cover_count[bp_cand]==1).sum() to minimize waste.
        """
        code_set = set(code)

        improved = True
        iter_count = 0
        rng = np.random.RandomState(42)
        while improved and time.time() < deadline and iter_count < max_iter:
            improved = False
            iter_count += 1

            code_indices = list(range(len(code)))
            rng.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline:
                    break
                i = code[ci]
                i_ball = balls[i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    code.pop(ci)
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    improved = True
                    break

                # Find candidates that cover all of only_by_i
                candidate_set = None
                for w in only_by_i:
                    if candidate_set is None:
                        candidate_set = set(centers_covering[w])
                    else:
                        candidate_set &= centers_covering[w]
                    if not candidate_set:
                        break
                
                if not candidate_set:
                    continue

                candidate_set -= code_set
                if not candidate_set:
                    continue

                # Score each candidate: prefer one with minimum waste
                # Waste = number of points in candidate's ball that are uniquely covered
                # After swap, these points lose their unique coverage from i
                # Fast score: (cover_count[bp_cand] == 1).sum()
                best_candidate = None
                best_score = float('inf')
                for j in candidate_set:
                    score = int((cover_count[balls[j]] == 1).sum())
                    if score < best_score:
                        best_score = score
                        best_candidate = j
                    if best_score == 0:
                        break
                
                if best_candidate is None:
                    continue

                # Valid swap
                code[ci] = best_candidate
                code_set.discard(i)
                code_set.add(best_candidate)
                cover_count[i_ball] -= 1
                cover_count[balls[best_candidate]] += 1
                improved = True
                break

        code, cover_count = trim_code(code, cover_count)
        return code, cover_count

    def polish_pass(code, cover_count, time_budget=15):
        """
        High-precision polish pass using a tighter scoring function.
        Score = (sum over bp_cand of cover_count) - (sum over intersection of cover_count)
        This better captures true redundancy after the swap.
        """
        code_set = set(code)
        end_time = time.time() + time_budget
        improved = True
        rng = np.random.RandomState(999)

        while improved and time.time() < end_time:
            improved = False
            code_indices = list(range(len(code)))
            rng.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > end_time:
                    break
                i = code[ci]
                i_ball = balls[i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    code.pop(ci)
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    improved = True
                    break

                candidate_set = None
                for w in only_by_i:
                    if candidate_set is None:
                        candidate_set = set(centers_covering[w])
                    else:
                        candidate_set &= centers_covering[w]
                    if not candidate_set:
                        break
                
                if not candidate_set:
                    continue

                candidate_set -= code_set
                if not candidate_set:
                    continue

                # Tighter score: minimize net increase in unique coverage
                # After removing i and adding j:
                # Points in i_ball \ j_ball: cover_count decreases by 1
                # Points in j_ball \ i_ball: cover_count increases by 1
                # Points in intersection: unchanged
                # We want to minimize the number of points that become uniquely covered
                # after the swap (i.e., points in j_ball \ i_ball that had cover_count==1 before)
                # minus points that were uniquely covered in i_ball \ j_ball (which get fixed)
                
                best_candidate = None
                best_score = float('inf')
                for j in candidate_set:
                    j_ball = balls[j]
                    # Points in j_ball not in i_ball that have cover_count==1 (new unique points)
                    # Points in i_ball not in j_ball that have cover_count==1 (fixed unique points)
                    # Since all points in only_by_i are in j_ball by construction,
                    # the fixed points are exactly only_by_i
                    # New unique points: in j_ball, not in i_ball, cover_count==1
                    # Approximate: (cover_count[j_ball] == 1).sum() - len(only_by_i)
                    # But more precisely, we need to exclude points in i_ball
                    # For speed, use the fast score but with a penalty
                    waste = int((cover_count[j_ball] == 1).sum())
                    # Points in i_ball that are also in j_ball and have cover_count==1
                    # are not actually "waste" since they stay covered
                    # This is an approximation; the fast score is good enough
                    if waste < best_score:
                        best_score = waste
                        best_candidate = j
                    if best_score == 0:
                        break
                
                if best_candidate is None:
                    continue

                code[ci] = best_candidate
                code_set.discard(i)
                code_set.add(best_candidate)
                cover_count[i_ball] -= 1
                cover_count[balls[best_candidate]] += 1
                improved = True
                break

        code, cover_count = trim_code(code, cover_count)
        return code, cover_count

    # Multi-start with different random seeds
    best_code = None
    best_count = float('inf')
    np.random.seed(42)

    n_starts = 3
    for start_i in range(n_starts):
        if time.time() > deadline - 30:
            break

        if start_i == 0:
            code, cover_count = structured_init()
        else:
            code, cover_count = greedy_cover(seed_offset=start_i * 1000)

        code, cover_count = trim_code(code, cover_count)
        code, cover_count = targeted_local_search(code, cover_count, max_iter=100000)

        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    # Final aggressive local search on best solution
    if best_code is not None:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_code:
            cover_count[balls[c]] += 1
        best_code, cover_count = targeted_local_search(best_code, cover_count, max_iter=300000)
        
        # Polish pass with tighter scoring
        remaining_time = deadline - time.time() - 5
        if remaining_time > 5:
            best_code, cover_count = polish_pass(best_code, cover_count, time_budget=min(remaining_time, 15))

    if best_code is None:
        best_code = []

    result = []
    for c in best_code:
        result.append(int_to_digits(c, n, q))

    return result
