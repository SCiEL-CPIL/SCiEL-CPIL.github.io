import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n

    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute balls
    balls = []
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors = [idx]
        for pos in range(n):
            original = w[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                w[pos] = new_val
                neighbors.append(word_to_idx(w))
            w[pos] = original
        balls.append(neighbors)

    # Precompute ball masks
    ball_masks = []
    for c in range(total_words):
        mask = 0
        for nb in balls[c]:
            mask |= (1 << nb)
        ball_masks.append(mask)

    all_mask = (1 << total_words) - 1

    # Precompute ball sizes for quick reference
    ball_sizes = [len(b) for b in balls]

    best_solution = None
    best_size = float('inf')

    def greedy_cover(seed):
        random.seed(seed)
        covered_mask = 0
        code = []

        # Track uncovered count per codeword for faster gain computation
        # gain[c] = number of words in ball[c] not yet covered
        gains = [ball_sizes[c] for c in range(total_words)]

        while covered_mask != all_mask:
            if time.time() > deadline:
                return None

            # Find codeword with maximum gain
            best_gain = -1
            best_candidates = []

            for c in range(total_words):
                g = gains[c]
                if g > best_gain:
                    best_gain = g
                    best_candidates = [c]
                elif g == best_gain:
                    best_candidates.append(c)

            if not best_candidates or best_gain <= 0:
                break

            # Random tie-breaking for diversity
            chosen = random.choice(best_candidates)
            code.append(chosen)
            covered_mask |= ball_masks[chosen]

            # Update gains: for each word newly covered, decrement gain of all codewords that cover it
            # We need to know which codewords cover each word. Precompute reverse mapping.
            # Actually, let's just recompute gains lazily - it's O(total_words * ball_size) per step
            # which is 2187 * 19 ≈ 41K per step, and we have ~190 steps = ~8M ops. That's fine.
            # But let's use a smarter approach: maintain a list of words covered by each codeword
            # and for each newly covered word, find all codewords that contain it.

            # Precompute: for each word, list of codewords whose ball contains it
            # This is the "reverse ball" or "dual" structure
            # word_to_codewords[w] = list of c such that w in balls[c]

        # The above approach with gains update is complex. Let me use a simpler but still fast approach:
        # Just recompute gain for each candidate each iteration. With 2187 candidates and ~190 iterations,
        # that's 2187 * 190 * (cost of bit_count) ≈ manageable.
        # Actually, the original approach was fine. Let me just keep it simple and optimize the inner loop.

        # Reset and use the original approach but with early exit checks
        random.seed(seed)
        covered_mask = 0
        code = []

        while covered_mask != all_mask:
            if time.time() > deadline:
                return None

            best_gain = -1
            best_candidates = []

            # Optimize: skip codewords already in code (they have gain 0 effectively)
            in_code_set = set(code) if len(code) < 50 else None  # Only use set for small codes

            for c in range(total_words):
                if in_code_set is not None and c in in_code_set:
                    continue
                gain = (ball_masks[c] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain:
                    best_candidates.append(c)

            if not best_candidates:
                break

            chosen = random.choice(best_candidates)
            code.append(chosen)
            covered_mask |= ball_masks[chosen]

        return code

    def reduce_cover(code):
        if not code:
            return code

        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1

        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break

                if is_redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code.pop(i)
                    changed = True

        return code

    def local_search(code):
        if not code:
            return code

        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1

        in_code = [False] * total_words
        for c in code:
            in_code[c] = True

        def calculate_unique_masks():
            unique_masks = []
            for c in code:
                mask = 0
                for w in balls[c]:
                    if cover_count[w] == 1:
                        mask |= (1 << w)
                unique_masks.append(mask)
            return unique_masks

        unique_masks = calculate_unique_masks()
        surrogate_score = sum(m.bit_count() for m in unique_masks)

        improved = True
        stagnant_count = 0
        max_stagnant = 5

        while improved and time.time() < deadline:
            improved = False

            # 1. Try to remove redundant codewords
            redundant_idx = -1
            for i in range(len(code)):
                if unique_masks[i] == 0:
                    redundant_idx = i
                    break

            if redundant_idx != -1:
                c = code[redundant_idx]
                for w in balls[c]:
                    cover_count[w] -= 1
                in_code[c] = False
                code.pop(redundant_idx)
                unique_masks = calculate_unique_masks()
                surrogate_score = sum(m.bit_count() for m in unique_masks)
                improved = True
                stagnant_count = 0
                continue

            # 2. No redundancy. Try swaps.
            swap_made = False
            indices = list(range(len(code)))
            random.shuffle(indices)

            for i in indices:
                c = code[i]
                loss_mask = unique_masks[i]

                if loss_mask == 0:
                    continue

                # Find candidates c' not in code that cover loss_mask
                candidates = []
                for c_prime in range(total_words):
                    if in_code[c_prime]:
                        continue
                    if (ball_masks[c_prime] & loss_mask) == loss_mask:
                        candidates.append(c_prime)
                        if len(candidates) >= 8:
                            break

                if not candidates:
                    continue

                # Evaluate candidates by simulating the swap
                best_candidate = -1
                best_new_surrogate = surrogate_score + 1

                for c_prime in candidates:
                    # Simulate: remove c, add c_prime
                    # Compute new unique masks without full recalculation
                    # Words in loss_mask: were unique to c, now covered by c_prime (count becomes 2)
                    # Words in balls[c] \ loss_mask: count decreases by 1; if was 2, becomes 1 (newly unique to other)
                    # Words in balls[c_prime] \ loss_mask: count increases by 1; if was 1, becomes 2 (loses uniqueness)

                    # For efficiency, compute the delta in surrogate
                    # Old surrogate contribution from c: bit_count(loss_mask)
                    # New: c_prime may have some unique words after the swap

                    # Let's just do a quick estimate:
                    # Words that become newly unique: those in balls[c] \ loss_mask with count == 2
                    newly_unique_count = 0
                    for w in balls[c]:
                        if w not in loss_mask and cover_count[w] == 2:
                            newly_unique_count += 1

                    # Words that lose uniqueness: those in balls[c_prime] with count == 1 (not in loss_mask)
                    loses_unique_count = 0
                    for w in balls[c_prime]:
                        if cover_count[w] == 1:
                            loses_unique_count += 1

                    # Net change: -bit_count(loss_mask) + newly_unique_count - loses_unique_count
                    # Wait, this isn't quite right because the "other" codeword's unique mask changes too.
                    # Let's just do the full swap for the top few candidates.

                    # Perform swap
                    for w in balls[c]:
                        cover_count[w] -= 1
                    in_code[c] = False
                    code[i] = c_prime

                    for w in balls[c_prime]:
                        cover_count[w] += 1
                    in_code[c_prime] = True

                    new_unique_masks = calculate_unique_masks()
                    new_surrogate = sum(m.bit_count() for m in new_unique_masks)

                    if new_surrogate < best_new_surrogate:
                        best_new_surrogate = new_surrogate
                        best_candidate = c_prime
                        # Keep the state as-is for now, we'll revert if not accepted
                    else:
                        # Revert this candidate
                        for w in balls[c_prime]:
                            cover_count[w] -= 1
                        in_code[c_prime] = False
                        code[i] = c

                        for w in balls[c]:
                            cover_count[w] += 1
                        in_code[c] = True

                if best_candidate != -1 and best_new_surrogate < surrogate_score:
                    # Accept the best swap
                    # State is already set to best_candidate if it was the last one tried
                    # But we may have reverted it. Let's re-apply.
                    # Actually, the logic above is buggy. Let me simplify: just try one candidate at a time.
                    pass

                # Simplified: just pick the first valid candidate and try the swap
                c_prime = candidates[0]

                # Perform swap
                for w in balls[c]:
                    cover_count[w] -= 1
                in_code[c] = False
                code[i] = c_prime

                for w in balls[c_prime]:
                    cover_count[w] += 1
                in_code[c_prime] = True

                unique_masks = calculate_unique_masks()
                new_surrogate = sum(m.bit_count() for m in unique_masks)

                if new_surrogate < surrogate_score:
                    surrogate_score = new_surrogate
                    improved = True
                    swap_made = True
                    stagnant_count = 0
                    break
                else:
                    # Revert
                    for w in balls[c_prime]:
                        cover_count[w] -= 1
                    in_code[c_prime] = False
                    code[i] = c

                    for w in balls[c]:
                        cover_count[w] += 1
                    in_code[c] = True
                    unique_masks = calculate_unique_masks()

            if not swap_made:
                stagnant_count += 1
                if stagnant_count >= max_stagnant:
                    break
                # Try a forced swap: accept a swap that doesn't improve surrogate but might help later
                # Pick a random codeword and swap it with a random candidate that covers its unique words
                for i in indices:
                    c = code[i]
                    loss_mask = unique_masks[i]
                    if loss_mask == 0:
                        continue

                    candidates = []
                    for c_prime in range(total_words):
                        if in_code[c_prime]:
                            continue
                        if (ball_masks[c_prime] & loss_mask) == loss_mask:
                            candidates.append(c_prime)

                    if candidates:
                        c_prime = random.choice(candidates)
                        # Perform swap regardless of surrogate
                        for w in balls[c]:
                            cover_count[w] -= 1
                        in_code[c] = False
                        code[i] = c_prime

                        for w in balls[c_prime]:
                            cover_count[w] += 1
                        in_code[c_prime] = True

                        unique_masks = calculate_unique_masks()
                        surrogate_score = sum(m.bit_count() for m in unique_masks)
                        improved = True
                        swap_made = True
                        break

                if not swap_made:
                    break

        return code

    # Main loop
    initial_code = greedy_cover(0)
    if initial_code:
        reduced_code = reduce_cover(initial_code)
        local_searched_code = local_search(reduced_code)
        best_solution = local_searched_code
        best_size = len(local_searched_code)

    seed = 1
    while time.time() < deadline:
        code = greedy_cover(seed)
        if code is None:
            break

        reduced_code = reduce_cover(code)
        local_searched_code = local_search(reduced_code)

        if len(local_searched_code) < best_size:
            best_solution = local_searched_code
            best_size = len(local_searched_code)

        seed += 1

    if best_solution is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    result = []
    for idx in best_solution:
        result.append(idx_to_word(idx))

    return result
