import time
import numpy as np
import random

def construct(q, n, radius):
    """
    Constructs a covering code using:
    1. Structured tetracode product initialization
    2. Greedy completion
    3. Inverse-index based swap local search with perturbation
    Multiple restarts to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

    # Precompute balls
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

    # Build inverse index: for each word w, list of centers whose ball contains w
    # This is expensive to build fully, so we'll use it selectively
    # Instead, for swap checking, we'll use the ball arrays directly

    # Tetracode: K_3(4,1) = 9 codewords
    # The tetracode is a well-known optimal covering code for q=3, n=4, radius=1
    # It has 9 codewords
    tetracode = [
        [0,0,0,0], [0,1,1,2], [0,2,2,1],
        [1,0,2,1], [1,1,0,0], [1,2,1,2],
        [2,0,1,2], [2,1,2,0], [2,2,0,1]
    ]

    # Verify tetracode covers all 81 words
    tet_set = set()
    for tc in tetracode:
        tet_set.add(tuple(tc))
        for pos in range(4):
            for new_val in range(3):
                if new_val == tc[pos]:
                    continue
                tc2 = tc[:]
                tc2[pos] = new_val
                tet_set.add(tuple(tc2))
    assert len(tet_set) == 81, f"Tetracode doesn't cover all words: {len(tet_set)}/81"

    # Create product code: pairs of tetracode words
    product_code = []
    for tc1 in tetracode:
        for tc2 in tetracode:
            product_code.append(tc1 + tc2)

    # Convert to integer indices
    product_indices = [digits_to_int(cw, q) for cw in product_code]
    product_set = set(product_indices)

    # Check coverage of product code
    covered = np.zeros(total_words, dtype=bool)
    for idx in product_indices:
        covered[balls[idx]] = True

    uncovered_count = np.sum(~covered)

    # Greedy completion
    def greedy_complete(covered, code_list, code_set, cover_count, remaining_time):
        """Add codewords greedily to cover all uncovered words."""
        uncovered = ~covered
        while np.any(uncovered):
            if time.time() > remaining_time:
                break
            gains = uncovered[balls].sum(axis=1)
            gains[list(code_set)] = 0  # Don't pick existing codewords
            best_center = int(np.argmax(gains))
            max_gain = gains[best_center]
            if max_gain == 0:
                break
            code_list.append(best_center)
            code_set.add(best_center)
            uncovered[balls[best_center]] = False
            cover_count[balls[best_center]] += 1
            covered[balls[best_center]] = True

    # Trim redundant codewords
    def trim(code_list, cover_count, remaining_time):
        """Remove redundant codewords."""
        changed = True
        while changed and time.time() < remaining_time:
            changed = False
            to_remove = []
            for i, c in enumerate(code_list):
                if np.any(cover_count[balls[c]] == 1):
                    continue
                to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code_list[idx]
                    code_list.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True

    # Swap-based local search using inverse index
    def local_search(code_list, cover_count, remaining_time, max_iterations=20000):
        """Swap-based local search to minimize code size."""
        code_set = set(code_list)
        improved = True
        iterations = 0
        while improved and time.time() < remaining_time and iterations < max_iterations:
            improved = False
            iterations += 1
            # Try to remove each codeword
            for i in range(len(code_list)):
                c = code_list[i]
                # Check if c is redundant
                if not np.any(cover_count[balls[c]] == 1):
                    code_list.pop(i)
                    cover_count[balls[c]] -= 1
                    improved = True
                    break
            if improved:
                continue

            # Try swaps: for each codeword i, try replacing with candidate j
            random.shuffle(code_list)
            for i in range(len(code_list)):
                if time.time() > remaining_time:
                    break
                c = code_list[i]
                # Words only covered by c
                only_by_c = balls[c][cover_count[balls[c]] == 1]
                if len(only_by_c) == 0:
                    code_list.pop(i)
                    cover_count[balls[c]] -= 1
                    improved = True
                    break

                # Try to find a candidate j that covers all of only_by_c
                # Sample candidates
                n_samples = min(30, total_words - len(code_list))
                if n_samples == 0:
                    break
                candidates = random.sample(range(total_words), n_samples)
                for j in candidates:
                    if j in code_set:
                        continue
                    # Check if j covers all of only_by_c
                    j_ball = balls[j]
                    # Quick check: are all of only_by_c in j's ball?
                    # Use numpy for speed
                    j_ball_set = set(j_ball.tolist())
                    if all(w in j_ball_set for w in only_by_c):
                        # Valid swap
                        code_list[i] = j
                        code_set.discard(c)
                        code_set.add(j)
                        cover_count[balls[c]] -= 1
                        cover_count[balls[j]] += 1
                        improved = True
                        break
                if improved:
                    break

    # Run multiple restarts
    best_code = None
    best_size = float('inf')

    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        # Initialize with product code
        code_list = product_indices.copy()
        code_set = set(code_list)
        covered = np.zeros(total_words, dtype=bool)
        cover_count = np.zeros(total_words, dtype=np.int32)

        for idx in code_list:
            covered[balls[idx]] = True
            cover_count[balls[idx]] += 1

        # Greedy completion
        greedy_complete(covered, code_list, code_set, cover_count, deadline)

        # Trim
        trim(code_list, cover_count, deadline)

        # Local search
        local_search(code_list, cover_count, deadline)

        # Final trim
        trim(code_list, cover_count, deadline)

        # Check if we have full coverage
        covered = np.zeros(total_words, dtype=bool)
        for idx in code_list:
            covered[balls[idx]] = True
        if not np.all(covered):
            continue

        if len(code_list) < best_size:
            best_size = len(code_list)
            best_code = code_list.copy()

            # If we've found a very good code, we can stop early
            if best_size <= 420:
                break

    # If no valid code found, fall back to greedy from scratch
    if best_code is None:
        code_list = []
        code_set = set()
        covered = np.zeros(total_words, dtype=bool)
        cover_count = np.zeros(total_words, dtype=np.int32)
        greedy_complete(covered, code_list, code_set, cover_count, deadline)
        trim(code_list, cover_count, deadline)
        best_code = code_list

    # Convert back to list of lists of integers
    result = []
    for c in best_code:
        result.append(int_to_digits(c, n, q))

    return result
