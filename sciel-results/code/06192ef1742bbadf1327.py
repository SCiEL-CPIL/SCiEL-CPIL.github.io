"""
Ternary covering code K_3(6,1) construction.
Two-phase: greedy set cover initialization + simulated annealing reduction.
Uses numpy for efficient coverage tracking.
"""
import time
import numpy as np
import random
import math
import itertools


def _direct_sum_construction(q, n, radius):
    """Fallback: direct sum of ternary tetracode with all words on remaining coords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    Returns a list of codewords (each a list of n integers).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s

    total_words = q ** n
    cover_size = 1 + n * (q - 1)  # Each codeword covers this many words

    # Generate all words and map to indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbor indices for each word
    neighbor_lists = np.zeros((total_words, cover_size), dtype=np.int32)
    for i, w in enumerate(all_words):
        neighbor_lists[i, 0] = i
        pos = 1
        for coord in range(n):
            for v in range(q):
                if v != w[coord]:
                    nw = list(w)
                    nw[coord] = v
                    neighbor_lists[i, pos] = word_to_idx[tuple(nw)]
                    pos += 1

    # Phase 1: Greedy set cover to get a valid initial solution
    uncovered = np.ones(total_words, dtype=bool)
    covered_count = np.zeros(total_words, dtype=np.int32)
    code_set = set()
    codewords = []

    # Greedy: repeatedly pick the word that covers the most uncovered
    # Use a scoring array updated incrementally
    scores = np.full(total_words, cover_size, dtype=np.int32)

    while np.any(uncovered) and time.time() < deadline - 5:
        # Find best candidate
        best_idx = -1
        best_score = 0
        for i in range(total_words):
            if i in code_set:
                continue
            # Count how many uncovered words this covers
            s = 0
            for nbr in neighbor_lists[i]:
                if uncovered[nbr]:
                    s += 1
            if s > best_score:
                best_score = s
                best_idx = i
            if time.time() > deadline - 10:
                break

        if best_idx < 0 or best_score == 0:
            break

        # Add best_idx to code
        code_set.add(best_idx)
        codewords.append(list(all_words[best_idx]))
        for nbr in neighbor_lists[best_idx]:
            if uncovered[nbr]:
                uncovered[nbr] = False
                covered_count[nbr] += 1
            else:
                covered_count[nbr] += 1

    # Check if we have a valid solution
    if not np.any(uncovered):
        # Phase 2: Simulated Annealing to reduce code size
        in_code = np.zeros(total_words, dtype=bool)
        for idx in code_set:
            in_code[idx] = True

        current_size = len(code_set)
        current_uncovered = 0

        best_size = current_size
        best_indices = list(code_set)

        T = 5.0
        T_min = 0.05
        alpha = 0.999
        penalty = 10.0

        random.seed(42)

        while time.time() < deadline:
            if current_uncovered == 0:
                # Valid state: try to remove a codeword
                if current_size > 0:
                    # Try removing each codeword, pick the one with least impact
                    # For speed, try a random subset
                    candidates = list(code_set)
                    random.shuffle(candidates)
                    tried = 0
                    removed = False
                    for idx in candidates:
                        if tried > 20:
                            break
                        tried += 1
                        # Count how many words would become uncovered
                        delta_uncov = 0
                        for nbr in neighbor_lists[idx]:
                            if covered_count[nbr] == 1:
                                delta_uncov += 1

                        if delta_uncov == 0:
                            # Safe removal
                            for nbr in neighbor_lists[idx]:
                                covered_count[nbr] -= 1
                            in_code[idx] = False
                            code_set.discard(idx)
                            current_size -= 1
                            removed = True
                            break
                        else:
                            # Maybe accept with probability
                            delta_cost = -1 + penalty * delta_uncov
                            if random.random() < math.exp(-max(0, delta_cost) / T):
                                for nbr in neighbor_lists[idx]:
                                    covered_count[nbr] -= 1
                                    if covered_count[nbr] == 0:
                                        current_uncovered += 1
                                in_code[idx] = False
                                code_set.discard(idx)
                                current_size -= 1
                                removed = True
                                break

                    if current_uncovered == 0 and current_size < best_size:
                        best_size = current_size
                        best_indices = list(code_set)

            else:
                # Invalid state: add a codeword that covers uncovered words
                # Pick a random uncovered word, add a random neighbor
                uncov_idx = np.where(uncovered)[0]
                if len(uncov_idx) > 0:
                    target = random.choice(uncov_idx)
                    # Pick a random neighbor of target
                    nbrs_of_target = neighbor_lists[target]
                    candidate = random.choice(nbrs_of_target)
                    if not in_code[candidate]:
                        # Count how many uncovered it covers
                        delta_cover = 0
                        for nbr in neighbor_lists[candidate]:
                            if covered_count[nbr] == 0:
                                delta_cover += 1

                        delta_cost = 1 - penalty * delta_cover
                        if delta_cost < 0 or random.random() < math.exp(-max(0, delta_cost) / T):
                            for nbr in neighbor_lists[candidate]:
                                if covered_count[nbr] == 0:
                                    current_uncovered -= 1
                                covered_count[nbr] += 1
                            in_code[candidate] = True
                            code_set.add(candidate)
                            current_size += 1
                            # Update uncovered array
                            for nbr in neighbor_lists[candidate]:
                                if covered_count[nbr] == 1:
                                    uncovered[nbr] = False

            # Cooling
            T *= alpha
            if T < T_min:
                T = T_min

        # Return best found
        if current_uncovered == 0 and current_size < best_size:
            best_size = current_size
            best_indices = list(code_set)

        if best_size < len(codewords):
            return [list(all_words[idx]) for idx in best_indices]
        else:
            return codewords

    # Fallback to direct sum
    return _direct_sum_construction(q, n, radius)
