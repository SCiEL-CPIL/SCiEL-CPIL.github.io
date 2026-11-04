"""Improved program: greedy set-covering with simulated annealing for ternary covering code."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time


def construct(q, n, radius):
    """
    Greedy set-covering heuristic initialized with random seed,
    optimized via simulated annealing to minimize codeword count.
    """
    start_time = time.time()
    time_limit = 115  # buffer before 120s timeout

    # Total space size
    total_words = q ** n

    # Generate all words as tuples
    all_words = [tuple(word) for word in np.ndindex(*([q] * n))]
    all_words_set = set(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}

    # Precompute balls for all possible centers
    # Ball of radius r around a center: all words at Hamming distance <= r
    all_balls = {}
    for w in all_words:
        ball = set()
        # Generate all words at distance 0, 1, ..., radius
        for dist in range(radius + 1):
            if dist == 0:
                ball.add(w)
            else:
                # Choose 'dist' positions to change
                from itertools import combinations, product
                for positions in combinations(range(n), dist):
                    # For each position, choose a different symbol
                    choices = []
                    for pos in positions:
                        orig = w[pos]
                        choices.append([v for v in range(q) if v != orig])
                    for combo in product(*choices):
                        new_word = list(w)
                        for pos, val in zip(positions, combo):
                            new_word[pos] = val
                        ball.add(tuple(new_word))
        all_balls[w] = ball

    # Convert balls to sets of indices for faster operations
    all_balls_idx = {}
    for w in all_words:
        all_balls_idx[w] = set(word_to_idx[x] for x in all_balls[w])

    # === Phase 1: Greedy Set Covering ===
    uncovered = set(range(total_words))
    centers = []

    while uncovered:
        if time.time() - start_time > time_limit * 0.4:
            break

        best_center = None
        best_coverage = -1
        best_newly_covered = None

        for center in all_words:
            ball_idx = all_balls_idx[center]
            newly_covered = ball_idx & uncovered
            coverage = len(newly_covered)
            if coverage > best_coverage:
                best_coverage = coverage
                best_center = center
                best_newly_covered = newly_covered

        if best_center is None or best_coverage <= 0:
            break

        centers.append(best_center)
        uncovered -= best_newly_covered

    # Ensure complete coverage
    if uncovered:
        for idx in uncovered:
            w = idx_to_word[idx]
            if w not in centers:
                centers.append(w)
            uncovered -= all_balls_idx[w]

    # === Phase 2: Simulated Annealing Optimization ===
    current_centers = list(centers)
    current_count = len(current_centers)

    best_solution = list(current_centers)
    best_count = current_count

    # Coverage count: for each word index, how many centers cover it
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for c in current_centers:
        for idx in all_balls_idx[c]:
            coverage_count[idx] += 1

    # Verify initial solution is valid
    if np.any(coverage_count == 0):
        # Fallback: ensure validity
        for idx in range(total_words):
            if coverage_count[idx] == 0:
                # Find any center that covers this word
                for c in all_words:
                    if idx in all_balls_idx[c]:
                        current_centers.append(c)
                        for i2 in all_balls_idx[c]:
                            coverage_count[i2] += 1
                        break

    # Simulated Annealing parameters
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    random.seed(42)

    no_improve_steps = 0
    max_no_improve = 2000

    while T > T_min and time.time() - start_time < time_limit and no_improve_steps < max_no_improve:
        # Random move selection
        move_type = random.random()
        improved_this_iter = False

        if move_type < 0.4 and len(current_centers) > 1:
            # Try to remove a random center
            idx_remove = random.randint(0, len(current_centers) - 1)
            c_remove = current_centers[idx_remove]

            # Check if removable: all words in its ball must have coverage > 1
            removable = True
            for idx in all_balls_idx[c_remove]:
                if coverage_count[idx] <= 1:
                    removable = False
                    break

            if removable:
                # Remove the center
                for idx in all_balls_idx[c_remove]:
                    coverage_count[idx] -= 1
                current_centers.pop(idx_remove)
                new_count = len(current_centers)

                if new_count < best_count:
                    best_count = new_count
                    best_solution = list(current_centers)
                    no_improve_steps = 0
                    improved_this_iter = True
                else:
                    # Revert
                    current_centers.insert(idx_remove, c_remove)
                    for idx in all_balls_idx[c_remove]:
                        coverage_count[idx] += 1
                    no_improve_steps += 1

        elif move_type < 0.7 and len(current_centers) >= 1:
            # Swap: remove one center, add another
            idx_remove = random.randint(0, len(current_centers) - 1)
            c_remove = current_centers[idx_remove]

            # Find words only covered by c_remove
            only_covered = set()
            for idx in all_balls_idx[c_remove]:
                if coverage_count[idx] == 1:
                    only_covered.add(idx)

            # Try random new centers until we find one that covers all only_covered
            c_new = None
            for _ in range(20):
                candidate = random.choice(all_words)
                if candidate == c_remove:
                    continue
                ball_candidate = all_balls_idx[candidate]
                if only_covered.issubset(ball_candidate):
                    c_new = candidate
                    break

            if c_new is not None:
                # Perform swap
                for idx in all_balls_idx[c_remove]:
                    coverage_count[idx] -= 1
                for idx in all_balls_idx[c_new]:
                    coverage_count[idx] += 1
                current_centers[idx_remove] = c_new

                # Check if we can now remove another center
                removed_extra = False
                for i in range(len(current_centers)):
                    if i == idx_remove:
                        continue
                    c_check = current_centers[i]
                    can_remove = True
                    for idx in all_balls_idx[c_check]:
                        if coverage_count[idx] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        for idx in all_balls_idx[c_check]:
                            coverage_count[idx] -= 1
                        current_centers.pop(i)
                        removed_extra = True
                        if len(current_centers) < best_count:
                            best_count = len(current_centers)
                            best_solution = list(current_centers)
                            no_improve_steps = 0
                            improved_this_iter = True
                        break

                if not improved_this_iter:
                    no_improve_steps += 1

        elif move_type < 0.9 and len(current_centers) >= 2:
            # Remove two centers, add one that covers their union of uniquely-covered words
            i, j = random.sample(range(len(current_centers)), 2)
            c1 = current_centers[i]
            c2 = current_centers[j]

            # Words only covered by c1 or c2
            only_c1 = set()
            only_c2 = set()
            for idx in all_balls_idx[c1]:
                if coverage_count[idx] == 1:
                    only_c1.add(idx)
            for idx in all_balls_idx[c2]:
                if coverage_count[idx] == 1:
                    only_c2.add(idx)

            # Need a new center that covers only_c1 ∪ only_c2
            needed = only_c1 | only_c2
            c_new = None
            for _ in range(20):
                candidate = random.choice(all_words)
                if candidate in (c1, c2):
                    continue
                if needed.issubset(all_balls_idx[candidate]):
                    c_new = candidate
                    break

            if c_new is not None:
                # Remove both, add new
                for idx in all_balls_idx[c1]:
                    coverage_count[idx] -= 1
                for idx in all_balls_idx[c2]:
                    coverage_count[idx] -= 1
                for idx in all_balls_idx[c_new]:
                    coverage_count[idx] += 1

                # Adjust list
                if i < j:
                    current_centers.pop(j)
                    current_centers.pop(i)
                else:
                    current_centers.pop(i)
                    current_centers.pop(j)
                current_centers.append(c_new)

                if len(current_centers) < best_count:
                    best_count = len(current_centers)
                    best_solution = list(current_centers)
                    no_improve_steps = 0
                    improved_this_iter = True
                else:
                    # Revert
                    current_centers.pop()
                    current_centers.insert(i, c1)
                    current_centers.insert(j if i < j else j - 1, c2)
                    for idx in all_balls_idx[c1]:
                        coverage_count[idx] += 1
                    for idx in all_balls_idx[c2]:
                        coverage_count[idx] += 1
                    for idx in all_balls_idx[c_new]:
                        coverage_count[idx] -= 1
                    no_improve_steps += 1

        else:
            # Random perturbation: temporarily accept a worse solution
            if len(current_centers) > 2:
                idx_remove = random.randint(0, len(current_centers) - 1)
                c_remove = current_centers[idx_remove]
                c_new = random.choice(all_words)

                # Compute delta
                delta = 0
                for idx in all_balls_idx[c_remove]:
                    if coverage_count[idx] == 1:
                        delta += 1
                for idx in all_balls_idx[c_new]:
                    if idx not in all_balls_idx[c_remove]:
                        delta -= 1

                # Accept with probability exp(-delta/T) if delta > 0
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    for idx in all_balls_idx[c_remove]:
                        coverage_count[idx] -= 1
                    for idx in all_balls_idx[c_new]:
                        coverage_count[idx] += 1
                    current_centers[idx_remove] = c_new

                    if len(current_centers) < best_count:
                        best_count = len(current_centers)
                        best_solution = list(current_centers)
                        no_improve_steps = 0
                        improved_this_iter = True

        T *= cooling_rate

    # Use best solution found
    final_centers = best_solution

    # Final verification
    covered = set()
    for c in final_centers:
        covered |= all_balls_idx[c]

    if covered != set(range(total_words)):
        # Fallback to greedy solution
        final_centers = centers

    return [list(c) for c in final_centers]
# EVOLVE-BLOCK-END
