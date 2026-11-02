"""Ternary covering code K_3(6,1) via greedy set cover + local search."""
import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    # Total space size
    space_size = q ** n  # 729 for q=3, n=6

    # Precompute all indices and their digit representations
    # idx_to_digits: list of tuples, digits[i] = digits of word i
    idx_to_digits = []
    for idx in range(space_size):
        digits = []
        x = idx
        for i in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()  # Most significant first
        idx_to_digits.append(tuple(digits))

    def index_of_digits(digits):
        idx = 0
        for d in digits:
            idx = idx * q + d
        return idx

    def ball_indices(center_digits):
        """Return list of indices within Hamming distance <= radius from center_digits."""
        center_idx = index_of_digits(center_digits)
        if radius == 0:
            return [center_idx]
        if radius == 1:
            result = [center_idx]
            for pos in range(n):
                for d in range(q):
                    if d == center_digits[pos]:
                        continue
                    new_digits = list(center_digits)
                    new_digits[pos] = d
                    result.append(index_of_digits(tuple(new_digits)))
            return result
        # General radius (not expected for this task)
        result = [center_idx]
        for delta_tuple in itertools.product(range(q), repeat=n):
            dist = sum(1 for i in range(n) if delta_tuple[i] != 0)
            if dist == 0 or dist > radius:
                continue
            new_digits = [(center_digits[i] + delta_tuple[i]) % q for i in range(n)]
            result.append(index_of_digits(tuple(new_digits)))
        return result

    # Build a pool of candidate codewords: all 729 words
    # We'll use a greedy set cover approach
    # Precompute balls for all candidates
    all_candidates = [tuple(range(0, 0)) for _ in range(space_size)]  # placeholder
    all_candidate_digits = idx_to_digits  # list of tuples
    all_balls = [ball_indices(digits) for digits in all_candidate_digits]

    # Greedy set cover: repeatedly pick the candidate that covers the most uncovered words
    uncovered = [True] * space_size
    uncovered_count = space_size
    selected = []
    selected_balls = []

    # For efficiency, maintain for each candidate how many uncovered words it covers
    # and for each uncovered word, which candidates cover it (or just recompute)

    # Since space is small (729 words, 729 candidates, each ball has 19 words),
    # we can afford a reasonable greedy approach.

    # Precompute: for each candidate index, the set of words it covers
    # Actually all_balls[i] is the list of indices covered by candidate i

    # Greedy: at each step, find candidate maximizing number of newly covered words
    # To speed up, we track coverage count per word
    word_cover_count = [0] * space_size  # How many selected codewords cover this word

    while uncovered_count > 0:
        if time.time() > deadline:
            break

        best_gain = 0
        best_candidate = -1

        # For each candidate not yet selected, compute gain
        # Optimization: only consider candidates that cover at least one uncovered word
        for ci in range(space_size):
            if time.time() > deadline:
                break
            gain = 0
            for w in all_balls[ci]:
                if uncovered[w]:
                    gain += 1
                    if gain > best_gain:
                        # Early exit if we can't beat current best
                        pass
            if gain > best_gain:
                best_gain = gain
                best_candidate = ci
            # Early exit: if best_gain is already high, we can stop checking others
            # but we need to check all to be sure (or use a heuristic)

        if best_candidate == -1 or best_gain == 0:
            break

        # Select this candidate
        selected.append(all_candidate_digits[best_candidate])
        selected_balls.append(all_balls[best_candidate])
        for w in all_balls[best_candidate]:
            if uncovered[w]:
                uncovered[w] = False
                uncovered_count -= 1

        # Early exit if fully covered
        if uncovered_count == 0:
            break

    # Phase 2: Local search to reduce the number of codewords
    # Try removing codewords that are redundant
    def is_redundant(i, selected_balls_list):
        """Check if codeword i is redundant (its ball is covered by others)."""
        ball_i = selected_balls_list[i]
        for w in ball_i:
            found = False
            for j in range(len(selected_balls_list)):
                if j == i:
                    continue
                if w in selected_balls_list[j]:
                    found = True
                    break
            if not found:
                return False
        return True

    # Redundancy removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            if is_redundant(i, selected_balls):
                del selected[i]
                del selected_balls[i]
                changed = True
                break

    # Phase 3: Try swapping codewords to reduce count
    # For each codeword, try replacing it with another candidate that still covers everything
    # but allows removal of another codeword
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to find a swap: remove codeword i, add codeword j, then remove some k
        for i in range(len(selected)):
            if time.time() > deadline:
                break

            # Temporarily remove codeword i
            remaining_balls = [b for j, b in enumerate(selected_balls) if j != i]

            # Find words no longer covered
            # Compute union of remaining balls
            covered_set = set()
            for b in remaining_balls:
                covered_set.update(b)

            missing = [w for w in range(space_size) if w not in covered_set]

            if not missing:
                # Codeword i was redundant (shouldn't happen after cleanup, but just in case)
                del selected[i]
                del selected_balls[i]
                improved = True
                break

            # Try to find a candidate j (not already in selected) that covers all missing words
            # and whose addition allows removing some other codeword
            for j in range(space_size):
                if time.time() > deadline:
                    break

                # Skip if j is already in selected
                if j in [index_of_digits(cw) for cw in selected]:
                    continue

                ball_j = all_balls[j]
                # Check if ball_j covers all missing words
                if not all(w in ball_j for w in missing):
                    continue

                # If we add j, can we remove some other codeword?
                # New set: remaining_balls + [ball_j]
                new_balls = remaining_balls + [ball_j]

                # Check if any codeword in new_balls is redundant
                for k in range(len(new_balls)):
                    if is_redundant(k, new_balls):
                        # We can do this swap: remove i, add j, remove k
                        # But we need to be careful with indices
                        # selected without i is: [selected[m] for m != i]
                        # After adding j and removing k from that list...
                        # This is getting complex. Let's just do it directly.
                        new_selected = [selected[m] for m in range(len(selected)) if m != i]
                        new_selected.append(all_candidate_digits[j])
                        new_selected_balls = [selected_balls[m] for m in range(len(selected)) if m != i]
                        new_selected_balls.append(ball_j)
                        # Remove k
                        del new_selected[k]
                        del new_selected_balls[k]

                        selected = new_selected
                        selected_balls = new_selected_balls
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Final cleanup
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            if is_redundant(i, selected_balls):
                del selected[i]
                del selected_balls[i]
                changed = True
                break

    # Convert tuples to lists for return
    return [list(cw) for cw in selected]
