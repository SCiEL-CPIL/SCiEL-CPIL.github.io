"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    # Deadline guard: evaluator kills after 150s wall-clock; keep a safety margin.
    deadline = time.time() + 120.0

    # For the specific case q=3, n=6, radius=1, we aim for the known optimum of 73.
    # Strategy: Greedy set cover + Simulated Annealing to escape local minima.

    if q == 3 and n == 6 and radius == 1:
        return _solve_ternary_6_1(deadline)

    # Generic fallback: direct sum construction or greedy
    if n <= 4:
        # Small case: brute force or direct
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4:
            return tetracode
        else:
            # For n < 4, use a simple greedy
            pass

    # Generic greedy set cover
    return _greedy_set_cover(q, n, radius, deadline)


def _solve_ternary_6_1(deadline):
    q, n, radius = 3, 6, 1
    total = q ** n  # 729

    # Precompute: for each codeword, which words it covers (itself + 6*2 neighbors = 13 words)
    # Represent words as integers 0..728
    # word_to_idx: map from integer to itself (identity)
    # We'll track coverage as a boolean array of size 729

    # Helper: generate all words within Hamming distance 1 of a given word
    def ball1(word):
        # word is a list of 6 ints
        result = [tuple(word)]
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = list(word)
                    new_word[pos] = val
                    result.append(tuple(new_word))
        return result

    # Convert word tuple to index
    def tuple_to_idx(tup):
        idx = 0
        for v in tup:
            idx = idx * q + v
        return idx

    # Precompute balls for all 729 words
    balls = []
    for i in range(total):
        # Convert index to tuple
        digits = []
        tmp = i
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        w = tuple(digits)
        ball = ball1(w)
        idxs = [tuple_to_idx(b) for b in ball]
        balls.append(idxs)

    # Greedy initialization
    covered = np.zeros(total, dtype=bool)
    codewords = []

    # Start with a random or structured initial set
    # Use the tetracode-based construction as a starting point, then optimize
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    # Extend to length 6 by adding all possibilities for last 2 positions? That gives 9*9=81 words.
    # Actually, the direct sum gives 9 * 3^(6-4) = 81 words. Let's start with that.
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_codewords.append(h + list(tail))

    # Convert to indices
    initial_indices = []
    for cw in initial_codewords:
        idx = 0
        for v in cw:
            idx = idx * 3 + v
        initial_indices.append(idx)

    # Check coverage of initial set
    covered = np.zeros(total, dtype=bool)
    for idx in initial_indices:
        for b_idx in balls[idx]:
            covered[b_idx] = True

    uncovered_count = total - covered.sum()

    # If not fully covered, add codewords greedily until covered
    codeword_set = set(initial_indices)
    while uncovered_count > 0 and time.time() < deadline - 5.0:
        # Find the word that covers the most uncovered words
        best_idx = -1
        best_gain = -1
        for idx in range(total):
            if idx in codeword_set:
                continue
            gain = 0
            for b_idx in balls[idx]:
                if not covered[b_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx < 0 or best_gain <= 0:
            break
        codeword_set.add(best_idx)
        for b_idx in balls[best_idx]:
            covered[b_idx] = True
        uncovered_count -= best_gain

    # Now we have a valid covering. Let's try to reduce it with simulated annealing.
    codewords_list = list(codeword_set)

    # Simulated Annealing
    if time.time() < deadline - 10.0:
        codewords_list = _simulated_annealing(codewords_list, balls, covered, total, deadline)

    # Convert back to lists
    result = []
    for idx in codewords_list:
        digits = []
        tmp = idx
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        result.append(digits)

    return result


def _simulated_annealing(codewords_list, balls, covered, total, deadline):
    # We'll work with a set of codeword indices
    cw_set = set(codewords_list)
    best_size = len(cw_set)
    best_set = set(cw_set)

    # Temperature schedule
    T = 1.0
    cooling = 0.995
    min_T = 0.01

    # Track coverage counts for each word (how many codewords cover it)
    # This allows us to check if removing a codeword leaves any word uncovered
    coverage_count = np.zeros(total, dtype=np.int32)
    for idx in cw_set:
        for b_idx in balls[idx]:
            coverage_count[b_idx] += 1

    # A word is "critically covered" if coverage_count[word] == 1
    # We can only remove a codeword if all words it uniquely covers are also covered by others
    # Actually, we can remove a codeword if no word has coverage_count == 1 that is in its ball

    def can_remove(idx, coverage_count):
        for b_idx in balls[idx]:
            if coverage_count[b_idx] == 1:
                return False
        return True

    def remove_codeword(idx, coverage_count):
        for b_idx in balls[idx]:
            coverage_count[b_idx] -= 1

    def add_codeword(idx, coverage_count):
        for b_idx in balls[idx]:
            coverage_count[b_idx] += 1

    step = 0
    while T > min_T and time.time() < deadline - 5.0:
        step += 1
        # Choose a random move
        move_type = random.random()
        if len(cw_set) > 0 and move_type < 0.6:
            # Try to remove a random codeword
            if len(cw_set) == 0:
                T *= cooling
                continue
            idx = random.choice(list(cw_set))
            if can_remove(idx, coverage_count):
                # Accept
                cw_set.remove(idx)
                remove_codeword(idx, coverage_count)
                if len(cw_set) < best_size:
                    best_size = len(cw_set)
                    best_set = set(cw_set)
            else:
                # Reject (or accept with probability exp(-1/T) for exploration)
                if random.random() < np.exp(-1.0 / T):
                    # This would make an invalid cover, so we can't accept
                    # Instead, try a swap: remove this and add another
                    pass
        elif move_type < 0.8:
            # Try to add a random codeword (to escape local minima)
            idx = random.randint(0, total - 1)
            if idx not in cw_set:
                cw_set.add(idx)
                add_codeword(idx, coverage_count)
                # Then try to remove some codeword
                if len(cw_set) > 0:
                    # Try to remove a random codeword that is removable
                    for _ in range(10):
                        rem_idx = random.choice(list(cw_set))
                        if can_remove(rem_idx, coverage_count):
                            cw_set.remove(rem_idx)
                            remove_codeword(rem_idx, coverage_count)
                            break
        else:
            # Swap: remove one, add one
            if len(cw_set) == 0:
                T *= cooling
                continue
            rem_idx = random.choice(list(cw_set))
            add_idx = random.randint(0, total - 1)
            if add_idx not in cw_set:
                # Check if we can remove rem_idx after adding add_idx
                # Temporarily add
                add_codeword(add_idx, coverage_count)
                cw_set.add(add_idx)
                if can_remove(rem_idx, coverage_count):
                    cw_set.remove(rem_idx)
                    remove_codeword(rem_idx, coverage_count)
                else:
                    # Revert
                    cw_set.remove(add_idx)
                    remove_codeword(add_idx, coverage_count)

        T *= cooling

    # Return the best set found
    return list(best_set)


def _greedy_set_cover(q, n, radius, deadline):
    total = q ** n

    def tuple_to_idx(tup):
        idx = 0
        for v in tup:
            idx = idx * q + v
        return idx

    def idx_to_tuple(idx):
        digits = []
        tmp = idx
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        return tuple(digits)

    # Precompute balls
    balls = []
    for i in range(total):
        w = idx_to_tuple(i)
        ball = [w]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    ball.append(tuple(new_w))
        idxs = [tuple_to_idx(b) for b in ball]
        balls.append(idxs)

    covered = np.zeros(total, dtype=bool)
    codeword_set = set()

    while not covered.all() and time.time() < deadline - 5.0:
        best_idx = -1
        best_gain = -1
        for idx in range(total):
            if idx in codeword_set:
                continue
            gain = 0
            for b_idx in balls[idx]:
                if not covered[b_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx < 0 or best_gain <= 0:
            break
        codeword_set.add(best_idx)
        for b_idx in balls[best_idx]:
            covered[b_idx] = True

    result = []
    for idx in codeword_set:
        result.append(list(idx_to_tuple(idx)))
    return result
# EVOLVE-BLOCK-END
