import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline)

    # Fallbacks
    if q == 3 and n == 4 and radius == 1:
        return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _construct_ternary_8_1_optimized(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    # Precompute balls
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # Precompute ball membership: for each word, which balls contain it
    # ball_membership[w] = list of ball indices that contain w
    # More useful: for each word w, the set of codewords c such that w in balls[c]
    # That's just: c is within distance 1 of w, i.e., c in balls[w]
    # So ball_membership[w] = balls[w] (symmetric)
    # The key insight: to check if removing codeword c is safe,
    # we need: for all w in balls[c], there exists another selected c' with w in balls[c']
    # i.e., cover_count[w] > 1 for all w in balls[c]

    # Precompute word priority for greedy tie-breaking
    # Words with more "diverse" digits tend to cover more unique territory
    word_priority = np.zeros(total, dtype=np.float64)
    for w in range(total):
        temp = w
        digits = []
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        distinct = len(set(digits))
        # Prefer words that use all three symbols
        word_priority[w] = distinct * 10000 + (sum(1 for d in digits if d == 0) * 100 +
                                                 sum(1 for d in digits if d == 1) * 10 +
                                                 sum(1 for d in digits if d == 2))

    # Step 1: Greedy set cover
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    current_uncovered = np.ones(total, dtype=bool)

    while np.any(current_uncovered) and time.time() < deadline - 40.0:
        # For each candidate codeword, count how many currently uncovered words it covers
        counts = np.zeros(total, dtype=np.int32)
        for w in range(total):
            if current_uncovered[w]:
                counts[balls[w]] += 1

        valid_mask = counts > 0
        if not np.any(valid_mask):
            break

        max_count = np.max(counts[valid_mask])
        candidates = np.where((counts == max_count) & valid_mask)[0]

        if len(candidates) == 1:
            best_idx = candidates[0]
        else:
            # Tie-break: prefer word with highest priority, then lowest index for determinism
            prios = word_priority[candidates]
            best_idx = candidates[np.argmax(prios)]

        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)

    selected_list = list(selected)
    selected_set = set(selected_list)

    # Step 2: Aggressive pruning - remove codewords whose entire ball is over-covered
    changed = True
    while changed and time.time() < deadline - 35.0:
        changed = False
        i = 0
        while i < len(selected_list):
            c = selected_list[i]
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                selected_list.pop(i)
                selected_set.discard(c)
                changed = True
            else:
                i += 1

    # Step 3: Local search - iterative improvement
    # Moves: remove a codeword if safe, or swap (remove one, add another)
    # Key optimization: precompute for each codeword c, the set of words that are
    # "critical" for c (i.e., only covered by c). If we remove c, those words become uncovered.
    # A valid swap: find c_add such that balls[c_add] covers all critical words of c_remove.

    best_size = len(selected_list)
    best_solution = list(selected_list)
    best_cover = cover_count.copy()

    # Multiple passes of local search
    pass_count = 0
    max_passes = 50
    while time.time() < deadline - 5.0 and pass_count < max_passes:
        pass_count += 1
        improved = False

        # Pass A: Try removing each codeword (in random order)
        order = np.random.default_rng(pass_count * 1000 + 42).permutation(len(selected_list))
        for idx in order:
            if time.time() > deadline - 5.0:
                break
            c = selected_list[idx]
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                selected_list.pop(idx)
                selected_set.discard(c)
                improved = True
                # Shift indices after removal
                order = order[order > idx] - 1
                order = order[order >= 0]

        # Pass B: Try swaps
        if time.time() < deadline - 5.0:
            rng = np.random.default_rng(pass_count * 2000 + 7)
            # For each codeword, try to find a swap that reduces size or maintains size but improves coverage distribution
            for i in range(len(selected_list)):
                if time.time() > deadline - 5.0:
                    break
                c_remove = selected_list[i]
                # Find critical words (only covered by c_remove)
                critical = balls[c_remove]
                critical_mask = (cover_count[critical] == 1)
                critical_words = critical[critical_mask]

                if len(critical_words) == 0:
                    # Can remove directly (should have been caught in Pass A, but double-check)
                    if np.min(cover_count[balls[c_remove]]) > 1:
                        cover_count[balls[c_remove]] -= 1
                        selected_list.pop(i)
                        selected_set.discard(c_remove)
                        improved = True
                    continue

                # Find candidate additions that cover all critical words
                # A word c_add covers critical word w iff w in balls[c_add], iff c_add in balls[w]
                # So c_add must be in intersection of balls[w] for all w in critical_words
                # For efficiency, intersect incrementally
                if len(critical_words) > 3:
                    # Too many critical words, skip (unlikely to find good swap)
                    continue

                # Start with candidates from first critical word
                candidates = set(balls[critical_words[0]].tolist())
                for w in critical_words[1:]:
                    candidates &= set(balls[w].tolist())
                    if not candidates:
                        break

                if not candidates:
                    continue

                # Filter: c_add not already selected, and c_add != c_remove
                candidates = candidates - selected_set
                if not candidates:
                    continue

                # Try each candidate (or a random subset if too many)
                cand_list = list(candidates)
                if len(cand_list) > 20:
                    cand_list = [cand_list[rng.integers(0, len(cand_list))] for _ in range(20)]

                for c_add in cand_list:
                    # Check if swap is valid: after removing c_remove and adding c_add,
                    # all words are still covered
                    # Words that become uncovered: those in balls[c_remove] not in balls[c_add]
                    # that had coverage exactly 1 (only covered by c_remove)
                    # We already know critical_words are covered only by c_remove
                    # After adding c_add, they're covered if they're in balls[c_add]
                    # We ensured c_add covers all critical_words, so that's fine
                    # But we also need: no word becomes uncovered that wasn't already
                    # A word w becomes uncovered if:
                    #   w in balls[c_remove], w not in balls[c_add], cover_count[w] == 1
                    # i.e., w is critical for c_remove AND not in balls[c_add]
                    # But we already checked all critical words are in balls[c_add]
                    # So the swap should be valid!
                    # Let's verify:
                    new_cover = cover_count.copy()
                    new_cover[balls[c_remove]] -= 1
                    new_cover[balls[c_add]] += 1
                    if np.min(new_cover) >= 0:
                        # Valid swap!
                        cover_count = new_cover
                        selected_list[i] = c_add
                        selected_set.discard(c_remove)
                        selected_set.add(c_add)
                        improved = True
                        break  # Move to next i

        if not improved:
            break

    # Step 4: Final aggressive pruning
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        i = 0
        while i < len(selected_list):
            c = selected_list[i]
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                selected_list.pop(i)
                selected_set.discard(c)
                changed = True
            else:
                i += 1

    # Verify coverage
    if np.min(cover_count) < 0:
        # Shouldn't happen, but fix
        uncovered = np.where(cover_count == 0)[0]
        for w in uncovered:
            for nb in balls[w]:
                if nb not in selected_set:
                    selected_list.append(nb)
                    selected_set.add(nb)
                    cover_count[balls[nb]] += 1
                    break

    # Convert to list of digit lists
    result = []
    for w in selected_list:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)

    return result
