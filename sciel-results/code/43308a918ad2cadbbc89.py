"""Combined program: group-structured ILS + vectorized numpy coverage for K_3(6,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if n == 0:
        return [[]]

    total_words = q ** n

    # Specialized solver for the hard case
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Generic fallback
    if total_words > 50000:
        return _generic_construct(q, n, radius, deadline)
    return _generic_bitset_construct(q, n, radius, deadline)


def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    total_words = 729

    # Precompute all tuples and index mapping
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}

    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbor lists (radius 1: self + 18 single-coordinate changes)
    neighbor_indices = [[] for _ in range(total_words)]
    for i, t in enumerate(all_tuples):
        neighbor_indices[i].append(i)
        for pos in range(n):
            orig = t[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                t_list = list(t)
                t_list[pos] = new_val
                j = idx_map[tuple(t_list)]
                neighbor_indices[i].append(j)

    # Vectorized neighbor masks for fast coverage
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbor_indices]
    neighbor_matrix = np.zeros((total_words, total_words), dtype=np.int8)
    for i in range(total_words):
        neighbor_matrix[i, neighbors_np[i]] = 1

    # Precompute perturbation vectors in Z_3^6
    # Single-coordinate perturbations
    perturbations = []
    for pos in range(n):
        for delta in (1, 2):
            p = [0] * n
            p[pos] = delta
            perturbations.append(p)
    # Double-coordinate perturbations
    for pos1 in range(n):
        for pos2 in range(pos1 + 1, n):
            for d1 in (1, 2):
                for d2 in (1, 2):
                    p = [0] * n
                    p[pos1] = d1
                    p[pos2] = d2
                    perturbations.append(p)

    def apply_perturbation(word_idx, p):
        """Apply perturbation vector p to word at word_idx, return new index."""
        w = idx_to_word(word_idx)
        for pos in range(n):
            if p[pos] != 0:
                w[pos] = (w[pos] + p[pos]) % q
        return word_to_idx(w)

    def coverage_count(codeword_set):
        """Compute coverage count array for a set of codeword indices."""
        counts = np.zeros(total_words, dtype=np.int32)
        for cw in codeword_set:
            counts += neighbor_matrix[cw]
        return counts

    def is_covered(codeword_set):
        """Check if all words are covered."""
        counts = np.zeros(total_words, dtype=np.int32)
        for cw in codeword_set:
            counts += neighbor_matrix[cw]
        return (counts > 0).all()

    def greedy_cover_from(codewords, seed=0):
        """Greedy set cover starting from initial codewords."""
        if time.time() > deadline:
            return None

        rng = random.Random(seed)
        codewords = list(codewords)
        uncovered_arr = np.ones(total_words, dtype=bool)
        for cw in codewords:
            uncovered_arr[neighbors_np[cw]] = False

        while time.time() < deadline:
            if not uncovered_arr.any():
                break

            counts = neighbor_matrix.dot(uncovered_arr.astype(np.int32))
            best_count = counts.max()
            if best_count == 0:
                break

            max_mask = (counts == best_count)
            candidates = np.where(max_mask)[0]
            if len(candidates) == 1:
                best_cw = int(candidates[0])
            else:
                best_cw = int(rng.choice(candidates.tolist()))

            codewords.append(best_cw)
            uncovered_arr[neighbors_np[best_cw]] = False

        return codewords

    def remove_redundant(codewords):
        """Remove redundant codewords iteratively."""
        if time.time() > deadline:
            return codewords

        if not codewords:
            return codewords

        coverage = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            coverage[neighbors_np[cw]] += 1

        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                cw = codewords[i]
                if coverage[neighbors_np[cw]].min() >= 2:
                    coverage[neighbors_np[cw]] -= 1
                    codewords.pop(i)
                    changed = True
                    break

        return codewords

    def swap_improve(codewords):
        """Try 1-for-1 swaps to reduce count while maintaining coverage."""
        if time.time() > deadline:
            return codewords

        codewords = list(codewords)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            coverage = np.zeros(total_words, dtype=np.int32)
            for cw in codewords:
                coverage[neighbors_np[cw]] += 1

            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                cw = codewords[i]
                # Temporarily remove cw
                coverage[neighbors_np[cw]] -= 1
                uncovered = (coverage == 0)

                if not uncovered.any():
                    codewords.pop(i)
                    improved = True
                    break

                # Find best replacement
                uncovered_arr = uncovered.astype(np.int32)
                counts = neighbor_matrix.dot(uncovered_arr)
                for c in codewords:
                    counts[c] = -1

                best_count = counts.max()
                if best_count >= uncovered.sum():
                    best_cw = int(counts.argmax())
                    codewords[i] = best_cw
                    coverage[neighbors_np[cw]] += 1
                    coverage[neighbors_np[best_cw]] += 1
                    improved = True
                    break
                else:
                    # Restore
                    coverage[neighbors_np[cw]] += 1

        return codewords

    # Initial construction: tetracode direct sum (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for tail_a in range(3):
        for tail_b in range(3):
            for h in tetracode:
                word = h + [tail_a, tail_b]
                idx = word_to_idx(word)
                initial_codewords.append(idx)

    # Reduce initial construction
    reduced = remove_redundant(initial_codewords.copy())
    reduced = swap_improve(reduced)

    best_result = reduced
    best_count = len(reduced)

    # ILS: iterated local search with structured perturbations
    rng = random.Random(42)
    current = list(best_result)
    current_count = len(current)

    for iteration in range(5000):
        if time.time() > deadline:
            break

        # Perturb current solution
        num_perturbations = rng.randint(1, 3)
        perturbed = list(current)
        for _ in range(num_perturbations):
            if perturbed:
                idx_to_perturb = rng.randrange(len(perturbed))
                p = rng.choice(perturbations)
                new_idx = apply_perturbation(perturbed[idx_to_perturb], p)
                # Ensure distinctness
                if new_idx not in perturbed:
                    perturbed[idx_to_perturb] = new_idx

        # Local search: greedy fill + redundancy removal + swap improvement
        filled = greedy_cover_from(perturbed, seed=iteration)
        if filled is None:
            continue

        filled = remove_redundant(filled)
        filled = swap_improve(filled)

        if len(filled) < best_count:
            best_count = len(filled)
            best_result = list(filled)
            current = list(filled)
            current_count = len(filled)

        # Accept with probability based on improvement (simulated annealing-like)
        elif len(filled) <= current_count:
            # Accept equal or slightly worse to explore
            if rng.random() < 0.5:
                current = list(filled)
                current_count = len(filled)

    # Final polish on best result
    if best_result is not None:
        best_result = remove_redundant(best_result)
        best_result = swap_improve(best_result)

    # Convert back to lists
    if best_result is None:
        return [idx_to_word(idx) for idx in initial_codewords]

    return [idx_to_word(idx) for idx in best_result]


def _generic_bitset_construct(q, n, radius, deadline):
    """Generic bitset-based solver for small spaces."""
    total_words = q ** n

    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}

    neighbor_indices = [[] for _ in range(total_words)]
    for i, t in enumerate(all_tuples):
        neighbor_indices[i].append(i)
        for pos in range(n):
            orig = t[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                t_list = list(t)
                t_list[pos] = new_val
                j = idx_map[tuple(t_list)]
                if j not in neighbor_indices[i]:
                    neighbor_indices[i].append(j)

    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbor_indices]
    neighbor_matrix = np.zeros((total_words, total_words), dtype=np.int8)
    for i in range(total_words):
        neighbor_matrix[i, neighbors_np[i]] = 1

    # Greedy set cover
    uncovered_arr = np.ones(total_words, dtype=bool)
    codewords = []

    while time.time() < deadline:
        if not uncovered_arr.any():
            break

        counts = neighbor_matrix.dot(uncovered_arr.astype(np.int32))
        best_count = counts.max()
        if best_count == 0:
            break

        best_cw = int(counts.argmax())
        codewords.append(best_cw)
        uncovered_arr[neighbors_np[best_cw]] = False

    # Remove redundant
    coverage = np.zeros(total_words, dtype=np.int32)
    for cw in codewords:
        coverage[neighbors_np[cw]] += 1

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            cw = codewords[i]
            if coverage[neighbors_np[cw]].min() >= 2:
                coverage[neighbors_np[cw]] -= 1
                codewords.pop(i)
                changed = True
                break

    return [idx_to_word(idx) for idx in codewords]


def _generic_construct(q, n, radius, deadline):
    """Fallback for large spaces: greedy with time limit."""
    total_words = q ** n

    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Simple greedy: pick random words, check coverage
    rng = random.Random(42)
    codewords = []
    covered = set()

    while time.time() < deadline:
        # Check if all covered
        if len(covered) >= total_words:
            break

        # Pick a random uncovered word
        uncovered = set(range(total_words)) - covered
        if not uncovered:
            break

        target = rng.choice(list(uncovered))
        # Add a random word near target
        w = idx_to_word(target)
        for pos in range(n):
            if rng.random() < 0.3:
                w[pos] = rng.randint(0, q - 1)
        idx = word_to_idx(w)

        if idx not in codewords:
            codewords.append(idx)
            # Update covered
            w_arr = idx_to_word(idx)
            for pos in range(n):
                orig = w_arr[pos]
                for new_val in range(q):
                    if new_val != orig:
                        w_arr[pos] = new_val
                        nb_idx = word_to_idx(w_arr)
                        covered.add(nb_idx)
                    else:
                        w_arr[pos] = new_val
                w_arr[pos] = orig
            covered.add(idx)

    return [idx_to_word(idx) for idx in codewords]
# EVOLVE-BLOCK-END
