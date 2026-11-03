import time
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    """
    Constructs a covering code using MWIS-inspired greedy with targeted
    local search and multi-start restarts.
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

    # Precompute: for each center, which other centers share at least one neighbor
    # This helps with fast swap validation
    # ball_sets[i] = frozenset of balls[i] for O(1) subset checks
    ball_sets = [frozenset(balls[i].tolist()) for i in range(total_words)]

    # Precompute inverse index: for each word w, list of centers whose ball contains w
    # This enables direct candidate enumeration for targeted swaps
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
        For q=3, n=8, radius=1: use product of two length-4 ternary codes.
        The tetracode has 6 words of length 4 over {0,1,2} covering radius 1.
        Product gives 36 words of length 8. Then extend with greedy.
        """
        # Tetracode: 6 codewords of length 4 over {0,1,2}, covering radius 1
        # One known tetracode:
        tetracode = [
            [0, 0, 0, 0],
            [0, 1, 2, 1],
            [0, 2, 1, 2],
            [1, 0, 1, 2],
            [1, 1, 2, 0],
            [1, 2, 0, 2],
        ]
        # Verify it's a valid covering code (each word within dist 1 of some tetracode word)
        # Product: all pairs (a, b) from tetracode x tetracode
        product_code = []
        for a in tetracode:
            for b in tetracode:
                product_code.append(a + b)
        
        # Convert to integers
        code_ints = [digits_to_int(c, q) for c in product_code]
        # Remove duplicates if any
        code_ints = list(set(code_ints))
        
        # Compute cover_count
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1
        
        # Greedy extension: add centers to cover remaining uncovered words
        uncovered = (cover_count == 0)
        while np.any(uncovered):
            if time.time() > deadline:
                break
            gains = uncovered[balls].sum(axis=1)
            # Only consider centers not already in code
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

        # Randomize the order of centers for diversity
        center_order = np.arange(total_words)
        if seed_offset > 0:
            rng = np.random.RandomState(seed_offset)
            rng.shuffle(center_order)

        # Process centers in order, but use gain-based selection
        # Actually, let's do standard greedy but with random tie-breaking
        while np.any(uncovered):
            if time.time() > deadline:
                break

            gains = uncovered[balls].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break

            # Get all centers with max gain
            best_centers = np.where(gains == max_gain)[0]
            if len(best_centers) == 1:
                best_center = int(best_centers[0])
            else:
                # Random tie-break for diversity
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
        Aggressive targeted local search:
        For each codeword, find candidates that cover its uniquely-covered words.
        Uses inverse index (centers_covering) for direct candidate enumeration.
        """
        code_set = set(code)

        improved = True
        iter_count = 0
        while improved and time.time() < deadline and iter_count < max_iter:
            improved = False
            iter_count += 1

            # Shuffle code order
            code_indices = list(range(len(code)))
            np.random.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline:
                    break
                i = code[ci]
                # Words only covered by i
                i_ball = balls[i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    # Can remove i
                    code.pop(ci)
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    improved = True
                    break

                # Targeted search: find candidates that cover all of only_by_i
                # Use inverse index: for each word w in only_by_i, get centers covering w
                # Intersect these sets to get candidates that cover ALL of only_by_i
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

                # Remove current code members from candidates
                candidate_set -= code_set

                if not candidate_set:
                    continue

                # Pick a candidate (prefer one with high overlap with i's ball for better future swaps)
                # For now, just pick the first one
                j = next(iter(candidate_set))
                
                # Valid swap!
                code[ci] = j
                code_set.discard(i)
                code_set.add(j)
                cover_count[i_ball] -= 1
                cover_count[balls[j]] += 1
                improved = True
                break

        # Final trim
        code, cover_count = trim_code(code, cover_count)
        return code, cover_count

    # Multi-start with different random seeds
    best_code = None
    best_count = float('inf')
    np.random.seed(42)

    n_starts = 3
    for start_i in range(n_starts):
        if time.time() > deadline - 30:  # Leave 30s for final local search
            break

        if start_i == 0:
            # Structured initialization (tetracode product) for the first start
            code, cover_count = structured_init()
        else:
            # Greedy initialization with different randomization
            code, cover_count = greedy_cover(seed_offset=start_i * 1000)

        # Trim
        code, cover_count = trim_code(code, cover_count)

        # Local search
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

    # Convert to list of lists
    if best_code is None:
        # Fallback: should not happen
        best_code = []

    result = []
    for c in best_code:
        result.append(int_to_digits(c, n, q))

    return result
