import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    total = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

    def to_int(digits):
        val = 0
        for d in digits:
            val = val * q + d
        return val

    def to_digits(val):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = val % q
            val //= q
        return digits

    # Precompute balls
    balls = np.empty((total, ball_size), dtype=np.int32)
    for w in range(total):
        digits = to_digits(w)
        ball = [w]
        for pos in range(n):
            orig = digits[pos]
            for nv in range(q):
                if nv != orig:
                    digits[pos] = nv
                    ball.append(to_int(digits))
            digits[pos] = orig
        balls[w] = ball

    # Precompute inverse index: for each word, which centers cover it
    # Using frozensets for fast intersection
    centers_covering = [None] * total
    for i in range(total):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = set()
            centers_covering[w].add(i)
    
    for w in range(total):
        if centers_covering[w] is not None:
            centers_covering[w] = frozenset(centers_covering[w])

    def structured_init():
        """Tetracode product initialization for q=3, n=8."""
        if q != 3 or n != 8:
            return greedy_init()

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
        code_ints = list(set(to_int(c) for c in product_code))

        cover_count = np.zeros(total, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1

        uncovered = (cover_count == 0)
        while np.any(uncovered) and time.time() < deadline - 10:
            gains = uncovered[balls].sum(axis=1)
            code_set = set(code_ints)
            for c in code_set:
                gains[c] = -1
            max_gain = gains.max()
            if max_gain <= 0:
                break
            best = int(np.argmax(gains))
            code_ints.append(best)
            cover_count[balls[best]] += 1
            uncovered = (cover_count == 0)

        return code_ints, cover_count

    def greedy_init():
        """Standard greedy set cover."""
        uncovered = np.ones(total, dtype=bool)
        code = []
        cover_count = np.zeros(total, dtype=np.int32)
        while np.any(uncovered) and time.time() < deadline - 10:
            gains = uncovered[balls].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            best = int(np.argmax(gains))
            code.append(best)
            uncovered[balls[best]] = False
            cover_count[balls[best]] += 1
        return code, cover_count

    def trim_code(code, cover_count):
        """Remove all redundant codewords."""
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if np.all(cover_count[balls[c]] > 1):
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True
        return code, cover_count

    def targeted_local_search(code, cover_count):
        """
        Local search using inverse index for fast candidate finding.
        Strategy:
        1. Try to remove any codeword that is fully redundant.
        2. If none, try to swap a 'stuck' codeword (has unique points) with a candidate
           that covers those unique points and has the best 'reinforcement' score.
        """
        code_set = set(code)
        improved = True
        iter_count = 0

        while improved and time.time() < deadline - 5:
            improved = False
            iter_count += 1
            rng = random.Random(iter_count)
            code_indices = list(range(len(code)))
            rng.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline - 5:
                    break
                i = code[ci]
                i_ball = balls[i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    # Redundant, remove it
                    code.pop(ci)
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    improved = True
                    break

                # Find candidates covering all only_by_i
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

                # Exclude current code
                candidate_set -= code_set

                if not candidate_set:
                    continue

                # Score candidates: 
                # We want to swap to a position that "reinforces" points currently 
                # with count 1 (making them count 2), which makes other codewords 
                # more likely to become redundant.
                # Score = number of points in balls[cand] where cover_count == 1
                best_cand = None
                best_score = -1
                
                # Optimization: Precompute cover_count==1 mask? 
                # cover_count changes, so we compute on the fly.
                # balls[cand] is a vector. cover_count[balls[cand]] is a vector.
                
                for cand in candidate_set:
                    # How many points in cand's ball are currently uniquely covered by someone?
                    # If we add cand, those points become count 2.
                    # This is good for stability.
                    score = int(np.sum(cover_count[balls[cand]] == 1))
                    
                    # Tie-breaker: maybe prefer candidates that cover more points with count >= 2? 
                    # No, reinforcing count 1 is the key to unlocking removals.
                    
                    if score > best_score:
                        best_score = score
                        best_cand = cand

                if best_cand is not None and best_score > 0:
                    # Perform swap
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    code[ci] = best_cand
                    code_set.add(best_cand)
                    cover_count[balls[best_cand]] += 1
                    improved = True
                    break

        return code, cover_count

    # Main execution
    code, cover_count = structured_init()
    code, cover_count = trim_code(code, cover_count)
    
    # Run local search with a generous budget
    # The local search is the main driver for improvement here
    code, cover_count = targeted_local_search(code, cover_count)
    
    # Final trim to catch any redundancies created by swaps
    code, cover_count = trim_code(code, cover_count)
    
    # If we have time, run local search again
    if time.time() < deadline - 10:
        code, cover_count = targeted_local_search(code, cover_count)
        code, cover_count = trim_code(code, cover_count)

    # Convert to word lists
    result = [to_digits(w) for w in code]

    # Ensure distinct
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)

    return unique_result
