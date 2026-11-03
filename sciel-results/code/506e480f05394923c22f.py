"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    """
    Vectorized greedy set-cover for covering code K_q(n, radius) with
    randomized restarts and local search polish (removal + swap).
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

    def index_to_word(idx, q, n):
        """Convert a linear index back to a word (tuple of ints)."""
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return tuple(word)

    # Precompute all ball indices as a 2D numpy array: shape (total_words, ball_size)
    # For radius=1: ball(idx) = {idx} ∪ {idx ± weight for each position, each alternative digit}
    weights = [q ** (n - 1 - pos) for pos in range(n)]
    ball_flat = np.empty((total_words, ball_size), dtype=np.int64)
    for idx in range(total_words):
        # Get digits
        temp = idx
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = temp % q
            temp //= q
        # Ball entries
        ball_flat[idx, 0] = idx
        offset = 1
        for pos in range(n):
            w = weights[pos]
            d = digits[pos]
            for val in range(q):
                if val != d:
                    ball_flat[idx, offset] = idx + (val - d) * w
                    offset += 1

    # Helper: greedy set cover with optional randomization
    def greedy_cover(randomize=False, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        uncovered = np.ones(total_words, dtype=np.uint8)
        selected = []

        while uncovered.any():
            if time.time() > deadline - 5:
                break

            if randomize:
                # Randomized greedy: among top-k candidates, pick one with noise
                # Compute coverage counts for all candidates
                # For each candidate c, count = number of uncovered words in ball(c)
                # Use matrix approach: for each uncovered word, which candidates cover it?
                # Alternative: iterate but with random tie-breaking
                # Faster: compute all counts vectorized is hard; use sampling
                # Pick a random subset of candidates to evaluate, then pick best
                sample_size = min(200, total_words)
                candidates = rng.choice(total_words, size=sample_size, replace=False)
                best_count = -1
                best_idx = -1
                for cand in candidates:
                    count = np.count_nonzero(uncovered[ball_flat[cand]])
                    if count > best_count:
                        best_count = count
                        best_idx = cand
                if best_idx == -1:
                    # Fallback: full scan
                    for idx in range(total_words):
                        count = np.count_nonzero(uncovered[ball_flat[idx]])
                        if count > best_count:
                            best_count = count
                            best_idx = idx
            else:
                # Deterministic greedy: find max coverage
                # Vectorized: for each candidate, count uncovered in its ball
                # This is expensive; do it in chunks
                best_count = -1
                best_idx = -1
                chunk = 256
                for start in range(0, total_words, chunk):
                    end = min(start + chunk, total_words)
                    # For candidates in [start, end), compute coverage
                    # uncovered[ball_flat[start:end]] has shape (chunk, ball_size)
                    counts = np.count_nonzero(uncovered[ball_flat[start:end]], axis=1)
                    local_max = np.argmax(counts)
                    if counts[local_max] > best_count:
                        best_count = counts[local_max]
                        best_idx = start + local_max

            if best_idx == -1:
                break

            selected.append(best_idx)
            uncovered[ball_flat[best_idx]] = 0

        return selected

    # Helper: local search polish (removal + swap)
    def local_search(selected, time_budget=5.0):
        if not selected:
            return selected

        selected = list(selected)
        end_time = time.time() + time_budget

        # Build cover count
        cover_count = np.zeros(total_words, dtype=np.int32)
        for ci in selected:
            cover_count[ball_flat[ci]] += 1

        # Pass 1: removal
        changed = True
        while changed and time.time() < end_time:
            changed = False
            for i in range(len(selected)):
                ci = selected[i]
                ball = ball_flat[ci]
                if not (cover_count[ball] == 1).any():
                    cover_count[ball] -= 1
                    selected.pop(i)
                    changed = True
                    break

        # Pass 2: swap-based local search
        if time.time() < end_time - 2:
            # Rebuild cover count
            cover_count = np.zeros(total_words, dtype=np.int32)
            for ci in selected:
                cover_count[ball_flat[ci]] += 1

            selected_set = set(selected)
            improved = True
            while improved and time.time() < end_time - 2:
                improved = False
                for i in range(len(selected)):
                    if time.time() > end_time - 3:
                        break
                    ci = selected[i]
                    ball_i = ball_flat[ci]
                    unique_mask = (cover_count[ball_i] == 1)
                    n_unique = np.count_nonzero(unique_mask)

                    # Try random swaps
                    max_tries = 30
                    for _ in range(max_tries):
                        if time.time() > end_time - 3:
                            break
                        cand = np.random.randint(0, total_words)
                        if cand in selected_set:
                            continue

                        ball_cand = ball_flat[cand]

                        # Check: after removing ci and adding cand, is coverage still valid?
                        # Simulate: cover_count[ball_i] -= 1; cover_count[ball_cand] += 1
                        # Check min >= 0
                        # Efficient: check if any word in ball_i has count 1 AND is not in ball_cand
                        # i.e., unique words of ci must all be in ball_cand
                        if n_unique > 0:
                            # Get unique word indices
                            unique_indices = ball_i[unique_mask]
                            # Check if all are in ball_cand
                            # Use: for each unique index, check membership in ball_cand
                            # Faster: count how many unique indices are NOT in ball_cand
                            # Create a boolean array for ball_cand membership (expensive)
                            # Alternative: use np.isin (slow for large arrays)
                            # Alternative: simulate the swap and check
                            pass

                        # Just simulate the swap
                        cover_count[ball_i] -= 1
                        cover_count[ball_cand] += 1
                        if np.any(cover_count == 0):
                            # Invalid
                            cover_count[ball_cand] -= 1
                            cover_count[ball_i] += 1
                        else:
                            # Valid swap
                            selected[i] = cand
                            selected_set.discard(ci)
                            selected_set.add(cand)
                            improved = True
                            break

                    if improved:
                        break

            # After swaps, try more removals
            if time.time() < end_time - 1:
                cover_count = np.zeros(total_words, dtype=np.int32)
                for ci in selected:
                    cover_count[ball_flat[ci]] += 1

                changed = True
                while changed and time.time() < end_time:
                    changed = False
                    for i in range(len(selected)):
                        ci = selected[i]
                        ball = ball_flat[ci]
                        if not (cover_count[ball] == 1).any():
                            cover_count[ball] -= 1
                            selected.pop(i)
                            changed = True
                            break

        return selected

    # Helper: validate solution
    def is_valid(selected):
        cover = np.zeros(total_words, dtype=np.uint8)
        for ci in selected:
            cover[ball_flat[ci]] = 1
        return np.all(cover == 1)

    # Fallback construction
    def fallback():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Main: try multiple strategies and keep the best
    best_solution = None
    best_size = float('inf')

    # Strategy 1: Deterministic greedy + local search
    if time.time() < deadline - 20:
        t0 = time.time()
        selected = greedy_cover(randomize=False)
        if len(selected) > 0 and is_valid(selected):
            polished = local_search(selected, time_budget=min(10, deadline - time.time() - 5))
            if is_valid(polished):
                if len(polished) < best_size:
                    best_size = len(polished)
                    best_solution = polished

    # Strategy 2: Multiple randomized greedy + local search
    remaining_time = deadline - time.time() - 5
    if remaining_time > 5:
        num_restarts = int(min(20, remaining_time / 3))  # ~3s per restart
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            seed = r * 12345 + 67890
            t0 = time.time()
            selected = greedy_cover(randomize=True, seed=seed)
            if len(selected) == 0:
                continue
            if not is_valid(selected):
                continue
            # Polish with limited time
            polish_time = min(3, deadline - time.time() - 3)
            if polish_time > 0.5:
                polished = local_search(selected, time_budget=polish_time)
                if is_valid(polished):
                    if len(polished) < best_size:
                        best_size = len(polished)
                        best_solution = polished

    # If no valid solution found, use fallback
    if best_solution is None:
        fb = fallback()
        return fb

    # Convert to list of lists
    codewords = [list(index_to_word(ci, q, n)) for ci in best_solution]

    # Final validity check
    if not is_valid(best_solution):
        return fallback()

    return codewords

# EVOLVE-BLOCK-END
