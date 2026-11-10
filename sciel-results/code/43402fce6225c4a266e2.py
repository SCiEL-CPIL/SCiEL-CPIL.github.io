import time
import sys
from collections import deque

def construct():
    """
    Return a strictly increasing sequence of integers in [1,256].
    Sums of all nonempty consecutive blocks must be distinct.
    Maximize sequence length.
    """
    deadline = time.time() + 85.0  # 5s safety margin before 90s kill

    # First, run the greedy algorithm to get a baseline
    greedy_seq = greedy_construct()
    best_seq = list(greedy_seq)
    best_len = len(best_seq)

    # Try DFS with backtracking to find longer sequences
    # Use the greedy result as a starting point and try to extend it
    # or find alternative paths

    # Strategy: DFS with smart pruning
    # State: (current_sequence, set_of_sums, last_element)
    # We want to maximize length

    # Use iterative deepening or bounded DFS
    # Start from the greedy length and try to beat it

    def greedy_construct():
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    # Try to improve via local search / backtracking
    # Use a priority queue: states with more "room" to grow are preferred
    # A state is (seq, sums_set, next_min_value)
    # We prioritize states that can potentially reach longer sequences

    # For efficiency, we'll use a beam search approach
    beam_width = 50
    max_depth = 25  # We expect sequences up to ~20+ elements

    # State: (sequence, frozenset_of_sums, last_value)
    # We use a priority queue ordered by: (potential_length, -len(seq), seq)
    # Potential length heuristic: how many more elements could fit

    import heapq

    def compute_sums(seq):
        """Compute all consecutive block sums for a sequence."""
        sums = set()
        n = len(seq)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                sums.add(total)
        return sums

    def can_extend(seq, sums_set, candidate):
        """Check if candidate can be appended to seq without sum collision."""
        if candidate <= seq[-1]:
            return False
        # Compute new sums: candidate, candidate+seq[-1], candidate+seq[-1]+seq[-2], ...
        new_sums = set()
        total = candidate
        for a in reversed(seq):
            total += a
            new_sums.add(total)
        new_sums.add(candidate)
        if sums_set.intersection(new_sums):
            return False
        return True

    # Beam search with pruning
    # Each state: (priority, seq, sums_frozen)
    # Priority: we want to explore states that lead to longer sequences

    # Initial state: empty sequence
    # But start from greedy to get a good baseline quickly

    # Let's do a proper DFS with backtracking, using the greedy sequence as a guide
    # Try to build a sequence longer than greedy by backtracking at key decision points

    # More efficient: use the greedy sequence as a "skeleton" and try to insert
    # or replace elements to extend the sequence

    # Actually, let's do a randomized local search + greedy restart
    best_found = list(greedy_seq)

    def try_greedy_with_start(start_vals, sums_init, deadline):
        """Continue greedy from a given state."""
        values = list(start_vals)
        sums = set(sums_init)
        for x in range(values[-1] + 1, 257):
            if time.time() > deadline:
                break
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    # Try multiple randomized starts
    import random
    random.seed(42)

    for attempt in range(20):
        if time.time() > deadline:
            break

        # Create a perturbed starting sequence
        # Take first k elements of greedy, then try different choices
        k = min(attempt + 2, len(greedy_seq) - 2, 15)

        if k < 2:
            continue

        # Start with first k elements from greedy
        start = list(greedy_seq[:k])
        sums = compute_sums(start)

        # Now try different next values
        found_better = False
        for _ in range(3):  # Try a few random branches
            if time.time() > deadline:
                break

            # Try a random valid next value
            candidates = []
            last_val = start[-1]
            for x in range(last_val + 1, min(last_val + 20, 257)):
                if can_extend(start, sums, x):
                    candidates.append(x)

            if not candidates:
                # Try continuing greedy
                result = try_greedy_with_start(start, sums, deadline)
                if len(result) > len(best_found):
                    best_found = list(result)
                continue

            # Pick a random candidate (not always greedy)
            if random.random() < 0.5:
                # Greedy choice (smallest valid)
                choice = candidates[0]
            else:
                choice = random.choice(candidates)

            new_start = start + [choice]
            new_sums = sums.copy()
            # Add new sums
            total = choice
            for a in reversed(start):
                total += a
                new_sums.add(total)
            new_sums.add(choice)

            # Continue greedily from here
            result = try_greedy_with_start(new_start, new_sums, deadline)
            if len(result) > len(best_found):
                best_found = list(result)
                found_better = True

        if found_better:
            break  # Found something better, try more refinements

    # Final refinement: try to extend best_found further
    if time.time() < deadline - 2:
        sums_best = compute_sums(best_found)
        extended = try_greedy_with_start(best_found, sums_best, deadline)
        if len(extended) > len(best_found):
            best_found = extended

    # Validate final result
    def validate(seq):
        if not seq:
            return True
        # Strictly increasing
        for i in range(len(seq) - 1):
            if seq[i + 1] <= seq[i]:
                return False
        # All in [1, 256]
        if seq[0] < 1 or seq[-1] > 256:
            return False
        # All consecutive sums distinct
        sums = set()
        n = len(seq)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                if total in sums:
                    return False
                sums.add(total)
        return True

    if not validate(best_found):
        # Fallback to greedy
        return greedy_seq

    return best_found
