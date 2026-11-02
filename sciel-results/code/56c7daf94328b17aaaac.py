"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    # Generate all words in the space
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    space_size = q ** n

    # Helper: compute Hamming distance between a word and all codewords
    def hamming_distances(word, codewords):
        # word: shape (n,), codewords: shape (m, n)
        return np.sum(codewords != word[None, :], axis=1)

    # Helper: check if every word is within radius of some codeword
    def is_covering(codewords):
        if len(codewords) == 0:
            return space_size == 0
        # Use numpy for speed
        cd = np.array(codewords, dtype=np.uint8)
        # Check in chunks to avoid memory issues
        chunk_size = 500
        for i in range(0, space_size, chunk_size):
            chunk = all_words[i:i+chunk_size]
            # distances: (chunk_size, len(codewords))
            dists = np.sum(chunk[:, None, :] != cd[None, :, :], axis=2)
            if np.any(np.min(dists, axis=1) > radius):
                return False
        return True

    # Helper: greedy add codeword to cover most uncovered words
    def greedy_cover(codewords, uncovered_mask):
        """Add one codeword that covers the most uncovered words."""
        if not np.any(uncovered_mask):
            return None
        # For each candidate codeword (all space words), count how many uncovered words it covers
        # This is expensive, so we sample
        candidates = all_words
        best_cd = None
        best_count = -1
        # Sample candidates for speed
        if space_size > 5000:
            sample_idx = np.random.choice(space_size, min(5000, space_size), replace=False)
            candidates = all_words[sample_idx]
        cd_arr = candidates.astype(np.int16)
        unv_idx = np.where(uncovered_mask)[0]
        if len(unv_idx) > 2000:
            unv_idx = np.random.choice(unv_idx, 2000, replace=False)
        unv_words = all_words[unv_idx].astype(np.int16)
        # distances: (num_candidates, num_unv_sample)
        dists = np.sum(cd_arr[:, None, :] != unv_words[None, :, :], axis=2)
        covered = (dists <= radius)
        counts = covered.sum(axis=1)
        best_idx = np.argmax(counts)
        if counts[best_idx] > 0:
            best_cd = all_words[sample_idx[best_idx]].tolist() if space_size > 5000 else all_words[best_idx].tolist()
        return best_cd

    # Phase 1: Start with the tetracode-based construction as initial solution
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n >= 4:
        initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # For n < 4, just use all words
        initial = [list(w) for w in all_words]

    # Ensure distinctness
    seen = set()
    codewords = []
    for cw in initial:
        key = tuple(cw)
        if key not in seen:
            seen.add(key)
            codewords.append(cw)

    best_codewords = list(codewords)
    best_count = len(codewords)

    # Phase 2: Local search - try to remove codewords and re-cover
    random.seed(42)

    def try_remove_and_recover(cw_list, max_iterations=50):
        """Try to remove codewords one by one and recover coverage."""
        cw_set = set(tuple(cw) for cw in cw_list)
        current = list(cw_list)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing each codeword
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                trial = current[:i] + current[i+1:]
                if is_covering(trial):
                    current = trial
                    improved = True
                    break
        return current

    # Try to improve by removing redundant codewords
    codewords = try_remove_and_recover(codewords)
    if len(codewords) < best_count:
        best_count = len(codewords)
        best_codewords = list(codewords)

    # Phase 3: Randomized local search - swap codewords
    for iteration in range(20):
        if time.time() > deadline - 10:
            break

        # Try swapping a codeword with a random one
        current = list(best_codewords)
        random.shuffle(current)
        swapped = False
        for i in range(len(current)):
            if time.time() > deadline - 10:
                break
            # Try replacing current[i] with a random word
            new_word = all_words[random.randint(0, space_size - 1)].tolist()
            if tuple(new_word) in set(tuple(cw) for cw in current):
                continue
            trial = current[:i] + [new_word] + current[i+1:]
            if is_covering(trial):
                # Check if we can now remove something
                current = trial
                swapped = True
                # Try to remove redundant
                current = try_remove_and_recover(current, max_iterations=20)
                if len(current) < best_count:
                    best_count = len(current)
                    best_codewords = list(current)
                break

    # Phase 4: Hungarian-based optimization on a subset
    # This is a heuristic: pick a subset of codewords and try to optimize
    if time.time() < deadline - 20 and len(best_codewords) > 10:
        # Try a smaller set: greedily select codewords
        current = list(best_codewords)
        # Try to reduce by greedy selection
        uncovered = np.ones(space_size, dtype=bool)
        new_codewords = []
        random.shuffle(current)
        for cw in current:
            if time.time() > deadline - 10:
                break
            cw_arr = np.array(cw, dtype=np.uint8)
            dists = np.sum(all_words != cw_arr[None, :], axis=1)
            covered_mask = dists <= radius
            if np.any(uncovered & covered_mask):
                new_codewords.append(cw)
                uncovered &= ~covered_mask
            if not np.any(uncovered):
                break
        if len(new_codewords) < best_count and is_covering(new_codewords):
            best_count = len(new_codewords)
            best_codewords = list(new_codewords)

    # Final check
    if not is_covering(best_codewords):
        # Fallback to initial construction
        return initial

    return best_codewords
# EVOLVE-BLOCK-END
