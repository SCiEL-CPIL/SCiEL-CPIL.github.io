import random
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 130.0  # Safety margin before 150s kill

    total_words = q ** n

    # Convert index to tuple of digits
    def index_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(t)

    def tuple_to_index(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Generate all words
    words = [index_to_tuple(i) for i in range(total_words)]

    # Precompute neighbor bitsets
    # neighbor_bits[i] is an integer where bit j is 1 if word j is within distance radius of word i
    neighbor_bits = [0] * total_words

    # For each word, find its neighbors
    for i in range(total_words):
        w_i = words[i]
        mask = 0
        for j in range(total_words):
            w_j = words[j]
            dist = 0
            for k in range(n):
                if w_i[k] != w_j[k]:
                    dist += 1
            if dist <= radius:
                mask |= (1 << j)
        neighbor_bits[i] = mask

    # Greedy algorithm with restarts
    best_code = None
    best_size = total_words + 1

    num_restarts = 200
    # Adjust restarts based on time
    # Each restart takes approx (total_words * best_size) bit_count ops.
    # For 729 words, ~73 steps. 729*73 = 53k ops.
    # 200 restarts = 10.6M ops. Very fast.

    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break

        uncovered = (1 << total_words) - 1
        used = [False] * total_words
        code = []

        # Shuffle candidates to break ties randomly
        candidates = list(range(total_words))
        random.shuffle(candidates)

        while uncovered:
            best_gain = -1
            best_c = -1

            for c in candidates:
                if used[c]:
                    continue
                # Calculate gain
                g = (uncovered & neighbor_bits[c]).bit_count()
                if g > best_gain:
                    best_gain = g
                    best_c = c
                    # If we found a perfect cover for the remaining (unlikely in intermediate steps), 
                    # we can't get better than this specific gain, but we still need to finish.
                    # However, if best_gain is 0, we are stuck (should not happen if radius is valid).
                    if best_gain == 0:
                        break

            if best_c == -1:
                # No candidate found? Should not happen.
                break

            code.append(best_c)
            used[best_c] = True
            uncovered &= ~neighbor_bits[best_c]

        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]

            # Early termination if we hit the theoretical lower bound or a known good value?
            # Lower bound for K_3(6,1) is 71. We know 73 is possible.
            if best_size <= 73:
                # We found a very good solution. We can stop or keep searching for 71/72.
                # 71 is lower bound, 73 is upper bound.
                # Let's keep searching a bit more if time permits.
                pass

    # Local search: try to remove codewords if possible
    if best_code is not None:
        # Sort by some heuristic to improve removal chances
        # Try removing codewords in random order
        for attempt in range(100):
            if time.time() - start_time > time_limit:
                break

            # Try to remove a random codeword
            code_copy = best_code[:]
            random.shuffle(code_copy)

            # Check if we can remove each codeword
            for i in range(len(code_copy)):
                if time.time() - start_time > time_limit:
                    break

                # Check if removing code_copy[i] still covers all words
                # This is expensive, so only do it for a few attempts
                if i < 10:
                    remaining = code_copy[:i] + code_copy[i+1:]
                    # Check coverage
                    covered = 0
                    for c in remaining:
                        covered |= neighbor_bits[c]
                    if covered == (1 << total_words) - 1:
                        # We can remove this codeword
                        best_code = remaining
                        best_size = len(best_code)
                        code_copy = remaining[:]
                        break

    if best_code is None:
        # Fallback if something went wrong
        return [words[i] for i in range(total_words)]

    # Convert indices back to lists of digits
    result = [list(words[i]) for i in best_code]

    return result
