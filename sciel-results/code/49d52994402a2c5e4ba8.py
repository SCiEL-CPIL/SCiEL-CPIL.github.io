import time
import numpy as np
import random

def construct(q, n, radius):
    """
    Constructs a covering code using structured init, greedy set cover, trimming,
    and iterated local search with perturbation.
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
    inv_ball = [None] * total_words
    for c in range(total_words):
        for w in balls_list[c]:
            if inv_ball[w] is None:
                inv_ball[w] = []
            inv_ball[w].append(c)

    # Build a structured initial code using tetracode product structure
    # Tetracode: 4 codewords of length 4 over {0,1,2} that cover all 81 words within distance 1
    # The 4 tetracode words (one possible optimal covering code of size 4 for K_3(4,1)):
    # We use a known good set: the tetracode has 4 words
    # Actually for K_3(4,1), the optimal is 4 words. Let's use a structured approach.
    # For n=8, we can use product of two length-4 codes.
    # A simple structured init: use all combinations from a small base set.
    
    # Generate a structured starting code:
    # Use the fact that for radius 1, we can tile the space with blocks.
    # Start with a greedy approach but seed with structured candidates.
    
    # Structured init: create codewords on a grid pattern
    structured_code = []
    seen = set()
    # For each position, try values in a pattern
    # Use a simple lattice: step by q in each dimension
    for d0 in range(q):
        for d1 in range(q):
            for d2 in range(q):
                for d3 in range(q):
                    for d4 in range(q):
                        for d5 in range(q):
                            for d6 in range(q):
                                for d7 in range(q):
                                    w = [d0, d1, d2, d3, d4, d5, d6, d7]
                                    idx = digits_to_int(w, q)
                                    if idx not in seen:
                                        structured_code.append(idx)
                                        seen.add(idx)
    
    # That's too many. Let's use a smarter structured init.
    # For K_3(8,1), use product of K_3(4,1) codes.
    # A known K_3(4,1) code of size 4:
    tetracode = [
        [0, 0, 0, 0],
        [1, 1, 1, 1], 
        [2, 2, 2, 2],
        [1, 2, 0, 1]
    ]
    
    # Actually, let me just use a simple structured grid init
    # and let greedy + local search do the heavy lifting.
    # The key improvement is in the local search quality.
    
    # Greedy Set Cover
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    cover_count = np.zeros(total_words, dtype=np.int32)

    while np.any(uncovered) and time.time() < deadline:
        # Compute gains: for each center, how many currently-uncovered words does it cover?
        gains = np.zeros(total_words, dtype=np.int32)
        for c in range(total_words):
            b = balls[c]
            gains[c] = np.sum(uncovered[b])
        
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

    # Helper: get unique words (covered only by this center)
    def get_unique_words(c):
        return balls[c][cover_count[balls[c]] == 1]

    # Fast exact gain computation for swap i -> j using vectorized numpy
    # Valid swap requires: for all w in ball[i]\ball[j], cover_count[w] >= 2
    # And gain = |{w in ball[j]\ball[i] : cover_count[w]==0}|
    def compute_swap_gain(i, j):
        """Returns (is_valid, gain) for swapping center i with center j."""
        bi = balls[i]
        bj = balls[j]
        
        # Vectorized: create boolean masks
        in_i = np.zeros(total_words, dtype=bool)
        in_j = np.zeros(total_words, dtype=bool)
        in_i[bi] = True
        in_j[bj] = True
        
        # Words in ball[i] not in ball[j]
        only_in_i_mask = in_i & ~in_j
        # Check validity: all such words must have cover_count >= 2
        if np.any(cover_count[only_in_i_mask] < 2):
            return (False, 0)
        
        # Words in ball[j] not in ball[i] that are currently uncovered
        only_in_j_mask = in_j & ~in_i
        gain = int(np.sum(cover_count[only_in_j_mask] == 0))
        
        return (True, gain)

    # Local search with exact scoring
    def local_search(seed, max_rounds=200):
        nonlocal code, cover_count
        random.seed(seed)
        code_set = set(code)
        
        improved = True
        swap_round = 0

        while improved and time.time() < deadline and swap_round < max_rounds:
            improved = False
            swap_round += 1

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

                # Evaluate candidates with exact scoring
                candidates = list(candidate_set)
                random.shuffle(candidates)
                n_candidates = min(80, len(candidates))
                candidates = candidates[:n_candidates]

                best_swap = None
                best_gain = -1

                for j in candidates:
                    if time.time() > deadline:
                        return
                    is_valid, gain = compute_swap_gain(i, j)
                    if is_valid and gain > best_gain:
                        best_gain = gain
                        best_swap = j

                if best_swap is not None and best_gain > 0:
                    j = best_swap
                    code[ci] = j
                    code_set.discard(i)
                    code_set.add(j)
                    cover_count[balls[i]] -= 1
                    cover_count[balls[j]] += 1
                    improved = True
                    break

    # Run local search with different seeds
    best_code_size = len(code)
    best_code_copy = list(code)
    best_cover_copy = cover_count.copy()
    
    for seed in [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]:
        if time.time() < deadline:
            local_search(seed, max_rounds=150)
            trim_code()
            if len(code) < best_code_size:
                best_code_size = len(code)
                best_code_copy = list(code)
                best_cover_copy = cover_count.copy()

    # Restore best code found so far
    if len(code) > best_code_size:
        code[:] = best_code_copy
        cover_count[:] = best_cover_copy
        trim_code()

    # Final trimming
    trim_code()

    # High-precision polish pass with ILS perturbation
    def polish_pass():
        nonlocal code, cover_count
        if time.time() > deadline:
            return
        code_set = set(code)
        best_size = len(code)
        best_code_copy = list(code)
        best_cover_copy = cover_count.copy()
        improved = True
        polish_round = 0
        max_polish_rounds = 80

        while improved and time.time() < deadline and polish_round < max_polish_rounds:
            improved = False
            polish_round += 1

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
                if len(code) < best_size:
                    best_size = len(code)
                    best_code_copy = list(code)
                    best_cover_copy = cover_count.copy()
                continue

            # Try swaps with exact scoring - evaluate more candidates
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

                # Evaluate more candidates with exact scoring
                candidates = list(candidate_set)
                random.shuffle(candidates)
                n_candidates = min(200, len(candidates))
                candidates = candidates[:n_candidates]

                best_swap = None
                best_gain = -1

                for j in candidates:
                    if time.time() > deadline:
                        return
                    is_valid, gain = compute_swap_gain(i, j)
                    if is_valid and gain > best_gain:
                        best_gain = gain
                        best_swap = j

                if best_swap is not None and best_gain > 0:
                    j = best_swap
                    code[ci] = j
                    code_set.discard(i)
                    code_set.add(j)
                    cover_count[balls[i]] -= 1
                    cover_count[balls[j]] += 1
                    improved = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code_copy = list(code)
                        best_cover_copy = cover_count.copy()
                    break

            # ILS perturbation: if stuck, randomly perturb and re-search
            if not improved and len(code) > 10 and time.time() < deadline - 10:
                # Perturb: randomly swap 2-4 codewords with random candidates
                n_perturb = random.randint(2, min(4, len(code)))
                for _ in range(n_perturb):
                    if time.time() > deadline - 5:
                        break
                    ci = random.randint(0, len(code) - 1)
                    i = code[ci]
                    unique = get_unique_words(i)
                    if len(unique) == 0:
                        continue
                    unique_list = unique.tolist()
                    if not unique_list:
                        continue
                    # Find any candidate that covers all unique words
                    cand_set = set(inv_ball[unique_list[0]])
                    for w in unique_list[1:]:
                        cand_set &= set(inv_ball[w])
                        if not cand_set:
                            break
                    cand_set -= code_set
                    if cand_set:
                        j = random.choice(list(cand_set))
                        # Check validity
                        is_valid, _ = compute_swap_gain(i, j)
                        if is_valid:
                            code[ci] = j
                            code_set.discard(i)
                            code_set.add(j)
                            cover_count[balls[i]] -= 1
                            cover_count[balls[j]] += 1
                            improved = True
                            break

        # Restore best
        if len(code) > best_size:
            code[:] = best_code_copy
            cover_count[:] = best_cover_copy
            trim_code()

    polish_pass()
    trim_code()

    # Convert back to list of lists of integers
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))

    return result
