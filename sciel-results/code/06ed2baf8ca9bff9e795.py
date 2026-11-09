"""Covering code K_3(8,1) via direct-sum construction + greedy + vectorized local search."""
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Total number of words
    total_words = q ** n

    # Precompute all words as a list of tuples
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute the ball (neighbors within radius) for each word
    # For radius=1, q=3, n=8: each word covers 1 + n*(q-1) = 1 + 8*2 = 17 words
    # We'll store ball indices as numpy arrays for vectorized operations
    ball_indices = []
    for w in all_words:
        ball = [w]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    nw = list(w)
                    nw[pos] = val
                    ball.append(tuple(nw))
        ball_indices.append(np.array([word_to_idx[b] for b in ball], dtype=np.int32))

    # Build coverage matrix: ball_masks[i] is a boolean array of length total_words
    # indicating which words are covered by word i
    ball_masks = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        ball_masks[i, ball_indices[i]] = True

    # Initial seed: use tetracode direct sum as starting point
    # Tetracode: [a, b, (a+b)%3, (a+2b)%3] for a,b in {0,1,2}
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(tuple(h + list(tail)))

    # Greedy set cover: start from initial, then add words that cover most uncovered
    covered = np.zeros(total_words, dtype=bool)
    for cw in initial_codewords:
        idx = word_to_idx[cw]
        covered |= ball_masks[idx]

    uncovered_count = total_words - np.sum(covered)
    current_code = list(initial_codewords)
    code_indices = [word_to_idx[cw] for cw in current_code]

    # If not fully covered, add words greedily
    # Use vectorized approach: for each candidate, count how many uncovered words it covers
    while uncovered_count > 0 and (time.time() - start_time) < deadline - 30:
        uncovered_mask = ~covered
        # For each word, count how many uncovered words it covers
        # This is expensive, so we use a heuristic: try all words but in a smart order
        best_idx = -1
        best_count = 0
        for i in range(total_words):
            cnt = np.sum(ball_masks[i] & uncovered_mask)
            if cnt > best_count:
                best_count = cnt
                best_idx = i
            if (time.time() - start_time) > deadline - 30:
                break
        if best_idx == -1 or best_count == 0:
            break
        covered |= ball_masks[best_idx]
        current_code.append(all_words[best_idx])
        code_indices.append(best_idx)
        uncovered_count = total_words - np.sum(covered)

    # Local search: try to remove redundant codewords using vectorized count
    # count_arr[x] = number of codewords whose ball covers word x
    count_arr = np.zeros(total_words, dtype=np.int32)
    for idx in code_indices:
        count_arr[ball_indices[idx]] += 1

    improved = True
    while improved and (time.time() - start_time) < deadline - 15:
        improved = False
        # Check each codeword: is it redundant?
        # A codeword i is redundant if for all x in ball[i], count_arr[x] > 1
        for ci in range(len(code_indices)):
            if (time.time() - start_time) > deadline - 15:
                break
            idx = code_indices[ci]
            ball_words = ball_indices[idx]
            if np.all(count_arr[ball_words] > 1):
                # Redundant, remove it
                count_arr[ball_words] -= 1
                code_indices.pop(ci)
                current_code.pop(ci)
                improved = True
                break

    # Try a few more rounds of removal
    for _ in range(3):
        if (time.time() - start_time) > deadline - 10:
            break
        improved = False
        for ci in range(len(code_indices)):
            if (time.time() - start_time) > deadline - 10:
                break
            idx = code_indices[ci]
            ball_words = ball_indices[idx]
            if np.all(count_arr[ball_words] > 1):
                count_arr[ball_words] -= 1
                code_indices.pop(ci)
                current_code.pop(ci)
                improved = True
                break
        if not improved:
            break

    # Convert back to list of lists
    result = [list(cw) for cw in current_code]

    # Ensure no duplicates
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)

    return unique_result
