"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n > 4:
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    elif n == 4:
        code = [h[:] for h in tetracode]
    else:
        # For n < 4, use all words (brute force)
        code = [list(w) for w in itertools.product(range(q), repeat=n)]
        return code

    # Precompute powers for fast encoding
    q_powers = np.array([q ** i for i in range(n)], dtype=np.int64)
    total_words = q ** n

    def word_to_int(w):
        return int(np.dot(np.array(w, dtype=np.int64), q_powers))

    def int_to_word(v):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = v % q
            v //= q
        return w

    # Convert code to integer representations
    code_ints = np.array([word_to_int(c) for c in code], dtype=np.int64)
    m = len(code_ints)

    # Precompute neighbor lists for each codeword using vectorized approach
    # For radius=1: each codeword covers 1 + n*(q-1) words
    neighbor_size = 1 + n * (q - 1)
    all_neighbors = np.zeros((m, neighbor_size), dtype=np.int64)

    for idx, ci in enumerate(code_ints):
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
        all_neighbors[idx] = neighbors

    # Build coverage counts
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for neighbors in all_neighbors:
        coverage_count[neighbors] += 1

    # Fast greedy removal: use vectorized scoring
    removed = np.zeros(m, dtype=bool)
    
    # Multiple passes with different strategies
    for pass_num in range(5):
        if time.time() > deadline:
            break
        
        # Compute redundancy score: number of neighbors with coverage > 1
        active = ~removed
        if not np.any(active):
            break
            
        # Vectorized score computation
        # For each active codeword, count how many of its neighbors have coverage > 1
        scores = np.zeros(m, dtype=np.int32)
        active_indices = np.where(active)[0]
        
        # Process in batches for memory efficiency
        batch_size = min(1000, len(active_indices))
        for batch_start in range(0, len(active_indices), batch_size):
            if time.time() > deadline:
                break
            batch = active_indices[batch_start:batch_start + batch_size]
            # Get neighbors for this batch
            batch_neighbors = all_neighbors[batch]  # shape: (batch_len, neighbor_size)
            # Get coverage counts for these neighbors
            batch_cov = coverage_count[batch_neighbors]  # shape: (batch_len, neighbor_size)
            # Count neighbors with coverage > 1
            scores[batch] = np.sum(batch_cov > 1, axis=1)

        # Try removing in order of highest score first
        order = np.argsort(-scores[active])
        for idx in order:
            actual_idx = active_indices[idx]
            if removed[actual_idx] or time.time() > deadline:
                continue
            neighbors = all_neighbors[actual_idx]
            # Check if removing this codeword leaves any word with coverage 0
            if np.any(coverage_count[neighbors] == 1):
                continue
            # Remove this codeword
            removed[actual_idx] = True
            coverage_count[neighbors] -= 1

    # Random order removal passes
    for _ in range(5):
        if time.time() > deadline:
            break
        improvement_found = False
        active_indices = np.where(~removed)[0]
        if len(active_indices) == 0:
            break
        np.random.shuffle(active_indices)
        for idx in active_indices:
            if removed[idx] or time.time() > deadline:
                continue
            neighbors = all_neighbors[idx]
            if np.any(coverage_count[neighbors] == 1):
                continue
            removed[idx] = True
            coverage_count[neighbors] -= 1
            improvement_found = True
        if not improvement_found:
            break

    # Build final code
    final_code = [code[i] for i in range(m) if not removed[i]]

    return final_code
# EVOLVE-BLOCK-END
