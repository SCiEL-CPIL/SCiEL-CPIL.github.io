import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 105  # safety margin before 150s kill

    N = 1000
    max_rep = 2

    # We'll use an array to track representation counts for sums 2..2N
    # counts[s] = number of pairs (a,b) with a <= b, a,b in S, a+b = s
    counts = np.zeros(2 * N + 1, dtype=np.int32)

    def try_add(x, chosen_arr, chosen_set):
        """Try to add x to the set. Returns True if valid."""
        # Check all new sums: x + y for y in chosen, and 2*x
        # For each such sum s, we need counts[s] < max_rep
        valid = True
        if x in chosen_set:
            return False
        for y in chosen_arr:
            s = x + y
            if counts[s] >= max_rep:
                valid = False
                break
        if valid:
            if counts[2 * x] >= max_rep:
                valid = False
        return valid

    def commit_add(x, chosen_arr, chosen_set):
        """Actually add x to the set and update counts."""
        for y in chosen_arr:
            counts[x + y] += 1
        counts[2 * x] += 1
        chosen_arr.append(x)
        chosen_set.add(x)

    def revert_add(x, chosen_arr, chosen_set):
        """Remove x from the set and update counts."""
        for y in chosen_arr:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        chosen_arr.remove(x)
        chosen_set.discard(x)

    # Greedy construction with multiple randomized strategies
    best_set = []
    best_size = 0

    # Strategy 1: Deterministic greedy (smallest first)
    chosen = []
    chosen_set = set()
    for x in range(1, N + 1):
        if try_add(x, chosen, chosen_set):
            commit_add(x, chosen, chosen_set)
    if len(chosen) > best_size:
        best_size = len(chosen)
        best_set = chosen.copy()
    counts[:] = 0

    # Strategy 2: Randomized greedy with multiple restarts
    # Use different random seeds / orderings
    rng = random.Random(42)

    while time.time() < deadline:
        # Create a random ordering
        order = list(range(1, N + 1))
        rng.shuffle(order)

        chosen = []
        chosen_set = set()
        for x in order:
            if try_add(x, chosen, chosen_set):
                commit_add(x, chosen, chosen_set)

        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen.copy()
            # Try to improve with local search
            # Remove some elements and try to re-add them in different order
            # But first, let's just continue with more restarts

        # Also try a "priority" greedy: at each step, pick the x that
        # minimizes the maximum count among new sums (least constraining)
        # This is more expensive but might find better sets

    # Try a smarter heuristic: minimize peak counts
    # Use a scoring function based on the counts vector
    while time.time() < deadline:
        chosen = []
        chosen_set = set()
        counts[:] = 0

        remaining = set(range(1, N + 1))
        while remaining and time.time() < deadline:
            # Find the best x to add: minimize the max count it would create
            best_x = None
            best_score = None
            # Sample a subset of candidates for speed
            candidates = list(remaining)
            if len(candidates) > 100:
                candidates = rng.sample(candidates, 100)

            for x in candidates:
                if x in chosen_set:
                    continue
                # Compute max count among new sums
                max_new_count = 0
                for y in chosen:
                    s = x + y
                    if counts[s] > max_new_count:
                        max_new_count = counts[s]
                if counts[2 * x] > max_new_count:
                    max_new_count = counts[2 * x]
                # If any new sum would exceed max_rep, skip
                skip = False
                for y in chosen:
                    if counts[x + y] >= max_rep:
                        skip = True
                        break
                if not skip and counts[2 * x] >= max_rep:
                    skip = True
                if skip:
                    continue
                # Score: prefer lower max count, break ties by smaller x
                if best_score is None or (max_new_count, x) < best_score:
                    best_score = (max_new_count, x)
                    best_x = x

            if best_x is None:
                break

            commit_add(best_x, chosen, chosen_set)
            remaining.discard(best_x)

        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen.copy()

    # Local search polish: try to swap elements to improve
    while time.time() < deadline - 5:
        improved = False
        for i in range(len(best_set)):
            x = best_set[i]
            # Try removing x and adding a different y
            # This is expensive, so only do a few swaps
            if time.time() > deadline - 2:
                break

            # Try a few random y values not in the set
            for _ in range(5):
                y = rng.randint(1, N)
                if y in best_set:
                    continue
                # Temporarily remove x
                chosen = best_set.copy()
                chosen_set = set(chosen)
                # Rebuild counts
                counts[:] = 0
                for a in chosen:
                    for b in chosen:
                        if a <= b:
                            counts[a + b] += 1

                # Now try to add y
                if try_add(y, chosen, chosen_set):
                    commit_add(y, chosen, chosen_set)
                    # Check if we can re-add x
                    if try_add(x, chosen, chosen_set):
                        # Both can be in the set - no improvement
                        counts[:] = 0
                        for a in best_set:
                            for b in best_set:
                                if a <= b:
                                    counts[a + b] += 1
                    else:
                        # x can't be re-added, but we have same size
                        # No improvement
                        counts[:] = 0
                        for a in best_set:
                            for b in best_set:
                                if a <= b:
                                    counts[a + b] += 1
                    break
            if time.time() > deadline - 2:
                break
        if not improved:
            break

    return best_set
