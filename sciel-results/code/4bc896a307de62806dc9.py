# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

_DEADLINE = None
_BUDGET = 150.0
_MARGIN = 10.0


def _time_left():
    if _DEADLINE is None:
        return _BUDGET
    return max(0.0, _DEADLINE - time.time() - _MARGIN)


def construct(q, n, radius):
    global _DEADLINE
    _DEADLINE = time.time()

    total_words = q ** n

    # Represent all words as base-q digits
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)

    # Precompute balls: for each word i, the list of word indices within Hamming distance 1
    ball_size = 1 + n * (q - 1)
    balls = np.empty((total_words, ball_size), dtype=np.int32)
    balls[:, 0] = np.arange(total_words)

    # For each position p and value v, compute which words have that value at position p
    # and what the index change would be if we set position p to v.
    q_powers = np.array([q ** (n - 1 - p) for p in range(n)], dtype=np.int64)

    offset = 1
    for p in range(n):
        for v in range(q):
            # Words where all_words[i][p] != v
            mask = all_words[:, p] != v
            # For those words, the ball includes the word with position p set to v
            # new_index = i + (v - all_words[i][p]) * q_powers[p]
            new_indices = (np.arange(total_words)[mask] +
                           (v - all_words[mask, p]) * q_powers[p]).astype(np.int32)
            # We need to place these in balls at column offset, but only for the masked words.
            # This is tricky with a 2D array. Let's use a different approach.
            # Store balls as a list of arrays instead.
            break
        break

    # Simpler approach: build balls as a list of numpy arrays
    balls_list = []
    for i in range(total_words):
        w = all_words[i]
        ball = [i]
        for p in range(n):
            for v in range(q):
                if v != w[p]:
                    # Compute index of word with position p changed to v
                    # index = sum(w[j] * q^(n-1-j))
                    # new_index = index + (v - w[p]) * q^(n-1-p)
                    new_idx = i + (v - w[p]) * (q ** (n - 1 - p))
                    ball.append(new_idx)
        balls_list.append(ball)

    # Convert to a single array for faster numpy operations
    # All balls have the same size (ball_size), so we can stack them
    balls_arr = np.array(balls_list, dtype=np.int32)

    # Greedy covering code
    uncovered = np.ones(total_words, dtype=bool)
    code = []

    while np.any(uncovered) and _time_left() > 5.0:
        # Find the word that covers the most uncovered words
        # count[i] = number of uncovered words in ball[i]
        # This is: for each i, sum(uncovered[balls_arr[i]])
        # We can do this with a matrix operation:
        # uncovered[balls_arr] gives a (total_words, ball_size) boolean array
        # Then sum along axis 1.
        # But this creates a (total_words, ball_size) array which is 6561 * 17 ≈ 111k elements. Fine.
        counts = uncovered[balls_arr].sum(axis=1)
        best_word = int(np.argmax(counts))
        best_count = int(counts[best_word])
        if best_count == 0:
            break
        code.append(all_words[best_word].tolist())
        uncovered[balls_arr[best_word]] = False

    result = code

    # Verify coverage (safety check)
    if _time_left() > 15.0 and result:
        # Build a set of codeword indices for fast lookup
        # Actually, let's verify by checking each word against the code
        # Using the balls: a word i is covered if it's in the ball of some codeword.
        # Equivalently, for each word i, check if any codeword j has i in ball[j].
        # This is: for each i, check if any codeword index c has balls_arr[c] containing i.
        # Simpler: mark all words covered by the code, then check if all are covered.
        covered = np.zeros(total_words, dtype=bool)
        for cw in result:
            # Find the index of cw
            # cw is a list of n ints, its index is sum(cw[j] * q^(n-1-j))
            idx = 0
            for j in range(n):
                idx = idx * q + cw[j]
            covered[balls_arr[idx]] = True
        if not np.all(covered):
            # Add uncovered words to the code
            for i in range(total_words):
                if not covered[i] and _time_left() > 5.0:
                    result.append(all_words[i].tolist())
                    covered[balls_arr[i]] = True

    return result
# EVOLVE-BLOCK-END
