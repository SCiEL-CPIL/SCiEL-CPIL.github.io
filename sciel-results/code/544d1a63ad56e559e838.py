import time
import numpy as np
import random
from collections import deque

def construct(q, n, radius):
    """
    Constructs a covering code using greedy set cover, trimming, and
    simulated annealing local search with swap moves.
    Uses inverse-index (reverse ball) for O(1) candidate enumeration.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    total_words = q ** n

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

    # Precompute balls using a tight Python loop
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
    ball_sizes = np.array([len(b) for b in balls_list], dtype=np.int32)

    # Precompute inverse index: for each word, list of centers whose ball contains it
    # This enables O(1) candidate enumeration for covering a set of unique words
    inv_ball = [None] * total_words
    for c in range(total_words):
        for w in balls_list[c]:
            if inv_ball[w] is None:
                inv_ball[w] = []
            inv_ball[w].append(c)

    # Greedy Set Cover with priority queue for efficiency
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    # Precompute ball sizes for initial gain computation
    ball_sizes_arr = np.array([len(b) for b in balls_list], dtype=np.int32)
    
    # Use a simpler but effective greedy: pick center with max uncovered coverage
    # Vectorized approach: for each center, gain = number of uncovered words in its ball
    while np.any(uncovered) and time.time() < deadline:
        # Vectorized gain computation using matrix-style accumulation
        # For each center c, gain[c] = sum of uncovered[b] for b in ball[c]
        # This is equivalent to: for each word w, add uncovered[w] to all centers whose ball contains w
        # Using inverse index: for each uncovered word w, increment gain for all centers in inv_ball[w]
        gains = np.zeros(total_words, dtype=np.int32)
        uncovered_indices = np.where(uncovered)[0]
        for w in uncovered_indices:
            if inv_ball[w] is not None:
                for c in inv_ball[w]:
                    gains[c] += 1
        
        best_center = int(np.argmax(gains))
        max_gain = gains[best_center]

        if max_gain == 0:
            break

        code.append(best_center)
        uncovered[balls[best_center]] = False
        cover_count[balls[best_center]] += 1

    # Trimming: remove redundant codewords
    def trim_code():
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

    trim_code()

    # Simulated annealing local search with inverse-index candidate enumeration
    def get_unique_words(c):
        return balls[c][cover_count[balls[c]] == 1]

    def local_search(seed, max_rounds=200):
        nonlocal code, cover_count
        random.seed(seed)
        code_set = set(code)
        
        improved = True
        swap_round = 0
        temp = 1.0  # temperature for simulated annealing

        while improved and time.time() < deadline and swap_round < max_rounds:
            improved = False
            swap_round += 1
            temp *= 0.99  # cool down

            # Try removing each codeword first
            for ci in range(len(code) - 1, -1, -1):
                if time.time() > deadline:
                    return
                i = code[ci]
                unique = get_unique_words(i)
                if len(unique) == 0:
                    code.pop(ci)
                    cover_count[balls[i]] -= 1
                    code_set.discard(i)
                    improved = True
                    break

            if improved:
                continue

            # Try swaps using inverse-index for fast candidate enumeration
            code_indices = list(range(len(code)))
            random.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline:
                    return
                i = code[ci]
                unique = get_unique_words(i)
                if len(unique) == 0:
                    continue

                # Use inverse index to find all centers that cover ALL unique words
                # Start with the first unique word's candidates, intersect with others
                unique_list = unique.tolist()
                if len(unique_list) == 0:
                    continue

                # Get candidates from the first unique word
                candidate_set = set(inv_ball[unique_list[0]])
                
                # Intersect with candidates from remaining unique words
                for w in unique_list[1:]:
                    candidate_set &= set(inv_ball[w])
                    if not candidate_set:
                        break

                # Remove current code words from candidates
                candidate_set -= code_set

                if not candidate_set:
                    continue

                # Try more candidates (shuffled for diversity)
                candidates = list(candidate_set)
                random.shuffle(candidates)
                n_candidates = min(50, len(candidates))
                candidates = candidates[:n_candidates]

                best_swap = None
                best_gain = -1

                # Precompute membership of balls[i] words for fast lookup
                bi_words = balls[i]
                bi_set = set(bi_words.tolist())

                for j in candidates:
                    if time.time() > deadline:
                        return
                    bj = balls[j]
                    # gain = number of w in ball[j] where:
                    #   if w in ball[i]: cover_count[w] == 1
                    #   if w not in ball[i]: cover_count[w] == 0
                    # Use set intersection for speed
                    bj_set = set(bj.tolist())
                    in_bi = bj_set & bi_set
                    not_in_bi = bj_set - bi_set
                    gain = 0
                    for w in in_bi:
                        if cover_count[w] == 1:
                            gain += 1
                    for w in not_in_bi:
                        if cover_count[w] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = j

                if best_swap is not None:
                    # Accept swap: always accept if gain >= 0 (it's neutral or better)
                    # The swap is valid because all unique words of i are covered by j
                    j = best_swap
                    code[ci] = j
                    code_set.discard(i)
                    code_set.add(j)
                    cover_count[balls[i]] -= 1
                    cover_count[balls[j]] += 1
                    improved = True
                    break

    # Run local search with different seeds and ILS perturbation
    def perturb(code_list, cover_cnt, code_set, strength=3):
        """Randomly swap 'strength' codewords to escape local minima."""
        if len(code_list) <= strength:
            return code_list, cover_cnt, code_set
        
        code_set_local = set(code_list)
        for _ in range(strength):
            if time.time() > deadline:
                break
            ci = random.randint(0, len(code_list) - 1)
            i = code_list[ci]
            unique = get_unique_words(i)
            if len(unique) == 0:
                continue
            
            unique_list = unique.tolist()
            if len(unique_list) == 0:
                continue
            
            # Find random valid replacement
            candidate_set = set(inv_ball[unique_list[0]])
            for w in unique_list[1:]:
                candidate_set &= set(inv_ball[w])
                if not candidate_set:
                    break
            
            candidate_set -= code_set_local
            if candidate_set:
                j = random.choice(list(candidate_set))
                code_list[ci] = j
                code_set_local.discard(i)
                code_set_local.add(j)
                cover_cnt[balls[i]] -= 1
                cover_cnt[balls[j]] += 1
        
        return code_list, cover_cnt, code_set_local

    # Iterated Local Search with multiple restarts
    best_code = list(code)
    best_count = len(code)
    
    for seed in [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]:
        if time.time() < deadline:
            # Perturb current solution
            code, cover_count, code_set = perturb(code, cover_count, set(code), strength=2)
            local_search(seed, max_rounds=100)
            trim_code()
            
            # Keep best solution found
            if len(code) < best_count:
                best_count = len(code)
                best_code = list(code)
    
    # Restore best solution
    code = best_code
    # Recompute cover_count for best code
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code:
        cover_count[balls[c]] += 1
    trim_code()

    # High-precision polish pass with multi-round improvement
    def polish_pass():
        nonlocal code, cover_count
        if time.time() > deadline:
            return
        code_set = set(code)
        improved = True
        polish_round = 0
        max_polish_rounds = 80

        while improved and time.time() < deadline and polish_round < max_polish_rounds:
            improved = False
            polish_round += 1

            # Try removing each codeword first (multiple passes)
            for ci in range(len(code) - 1, -1, -1):
                if time.time() > deadline:
                    return
                i = code[ci]
                unique = get_unique_words(i)
                if len(unique) == 0:
                    code.pop(ci)
                    cover_count[balls[i]] -= 1
                    code_set.discard(i)
                    improved = True
                    break

            if improved:
                continue

            # Try swaps with exact scoring, prioritizing high-gain swaps
            best_ci = -1
            best_j = -1
            best_gain_val = 0

            for ci in range(len(code)):
                if time.time() > deadline:
                    return
                i = code[ci]
                unique = get_unique_words(i)
                if len(unique) == 0:
                    continue

                unique_list = unique.tolist()
                if len(unique_list) == 0:
                    continue

                # Get candidates from the first unique word
                candidate_set = set(inv_ball[unique_list[0]])
                
                # Intersect with candidates from remaining unique words
                for w in unique_list[1:]:
                    candidate_set &= set(inv_ball[w])
                    if not candidate_set:
                        break

                # Remove current code words from candidates
                candidate_set -= code_set

                if not candidate_set:
                    continue

                candidates = list(candidate_set)
                random.shuffle(candidates)
                n_candidates = min(150, len(candidates))
                candidates = candidates[:n_candidates]

                bi_words = balls[i]
                bi_set = set(bi_words.tolist())

                for j in candidates:
                    if time.time() > deadline:
                        return
                    bj = balls[j]
                    bj_set = set(bj.tolist())
                    
                    # Exact scoring
                    sum_cand = 0
                    sum_inter = 0
                    for w in bj_set:
                        if w in bi_set:
                            if cover_count[w] == 1:
                                sum_cand += 1
                                sum_inter += 1
                        else:
                            if cover_count[w] == 0:
                                sum_cand += 1
                    
                    gain = sum_cand - sum_inter
                    
                    if gain > best_gain_val:
                        best_gain_val = gain
                        best_ci = ci
                        best_j = j

            if best_ci >= 0 and best_j >= 0 and best_gain_val > 0:
                i = code[best_ci]
                j = best_j
                code[best_ci] = j
                code_set.discard(i)
                code_set.add(j)
                cover_count[balls[i]] -= 1
                cover_count[balls[j]] += 1
                improved = True

    polish_pass()
    trim_code()

    # Convert back to list of lists of integers
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))

    return result
