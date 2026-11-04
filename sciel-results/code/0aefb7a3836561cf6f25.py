"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Generate the direct sum construction (81 words for q=3, n=6)
    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    # For the specific case q=3, n=6, radius=1, use optimized greedy set cover
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6
        powers = [3 ** p for p in range(n)]
        
        # Precompute neighbors as a numpy array of shape (total, 13)
        # Each row contains the index of the word and its 12 neighbors
        neighbors = np.zeros((total, 13), dtype=np.int32)
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(n)]
            neighbors[i, 0] = i
            for p in range(n):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        # Find the slot to insert
                        # It's easier to just fill it
                        # We know there are exactly 12 neighbors + self = 13
                        # We can just append or use a counter
                        pass
        
        # Refill neighbors properly
        neighbors = np.empty((total, 13), dtype=np.int32)
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(n)]
            neighbors[i, 0] = i
            k = 1
            for p in range(n):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        k += 1

        uncovered = np.ones(total, dtype=bool)
        codewords = []

        while uncovered.any():
            if time.time() > deadline:
                break

            # Vectorized coverage count
            # counts[j] = number of uncovered words in neighbors[j]
            counts = neighbors @ uncovered
            
            # Find the best candidate
            best_candidate = np.argmax(counts)
            
            if counts[best_candidate] == 0:
                # This should not happen if uncovered.any() is True
                # because every uncovered word covers itself
                break

            # Add this codeword
            codeword = [(best_candidate // powers[p]) % 3 for p in range(n)]
            codewords.append(codeword)

            # Mark covered
            uncovered[neighbors[best_candidate]] = False

        # Verify we have a valid covering
        if len(codewords) > 0 and not uncovered.any():
            # Compare with direct sum
            ds = get_direct_sum()
            if len(codewords) <= len(ds):
                return codewords
            else:
                return ds

    # General case: fallback to direct sum
    return get_direct_sum()
# EVOLVE-BLOCK-END
