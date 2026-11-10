"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    if n <= 4:
        return code

    # Precompute powers for fast encoding
    q_powers = np.array([q ** i for i in range(n)], dtype=np.int64)
    total_words = q ** n

    def word_to_int(w):
        return int(np.dot(w, q_powers))

    def int_to_word(v):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = v % q
            v //= q
        return w

    # Convert code to integer representations
    code_ints = np.array([word_to_int(c) for c in code], dtype=np.int64)
    m = len(code_ints)

    # Precompute neighbor offsets for radius=1: for each position, the q-1 alternative values
    # neighbor_offset[pos][val] = offset to add to word_int to change position pos from val to some other value
    # We'll precompute all possible neighbor offsets
    neighbor_offsets = []  # list of (pos, orig_val, new_val, offset)
    for pos in range(n):
        for orig_val in range(q):
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # offset = (new_val - orig_val) * q^pos
                offset = (new_val - orig_val) * (q ** pos)
                neighbor_offsets.append((pos, orig_val, new_val, offset))

    # For each codeword, precompute the list of neighbor ints
    # This is expensive but we do it once
    all_neighbors = []  # all_neighbors[i] = list of ints covered by code_ints[i]
    for ci in code_ints:
        word = int_to_word(int(ci))
        neighbors = [int(ci)]  # distance 0
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    word[pos] = new_val
                    neighbors.append(word_to_int(word))
                word[pos] = orig
        all_neighbors.append(np.array(neighbors, dtype=np.int64))

    # Build coverage counts using NumPy
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for neighbors in all_neighbors:
        coverage_count[neighbors] += 1

    # Greedy removal: try to remove codewords that don't leave any word uncovered
    removed = np.zeros(m, dtype=bool)
    improvement_found = True
    while improvement_found and time.time() < deadline:
        improvement_found = False
        # Try removing codewords in random order
        indices = np.random.permutation(m)
        for idx in indices:
            if removed[idx]:
                continue
            neighbors = all_neighbors[idx]
            # Check if removing this codeword leaves any word with coverage 0
            if np.any(coverage_count[neighbors] == 1):
                continue
            # Remove this codeword
            removed[idx] = True
            coverage_count[neighbors] -= 1
            improvement_found = True

    # Build final code
    final_code = [code[i] for i in range(m) if not removed[i]]

    return final_code
# EVOLVE-BLOCK-END
