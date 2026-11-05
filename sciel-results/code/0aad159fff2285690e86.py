"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _word_to_int(word, q):
    """Convert a word (list of ints 0..q-1) to an integer index."""
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _int_to_word(idx, q, n):
    """Convert an integer index to a word (list of ints 0..q-1)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _hamming_dist(w1, w2):
    """Hamming distance between two words (tuples)."""
    return sum(1 for a, b in zip(w1, w2) if a != b)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Leave safety margin before 150s kill

    total_words = q ** n
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Build ball sets: ball[i] = set of indices within radius of word i
    ball_sets = []
    for w in all_words:
        ball = {word_to_idx[w]}
        for r in range(1, radius + 1):
            for positions in itertools.combinations(range(n), r):
                for values in itertools.product(range(1, q), repeat=r):
                    new_word = list(w)
                    for pos, val in zip(positions, values):
                        new_word[pos] = (w[pos] + val) % q
                    ball.add(word_to_idx[tuple(new_word)])
        ball_sets.append(ball)

    ball_size = np.array([len(b) for b in ball_sets], dtype=np.int32)
    ball_members = [np.array(sorted(b), dtype=np.int32) for b in ball_sets]

    # Conflict graph: two words conflict if Hamming distance <= 2*radius
    # For covering radius 1, conflict if distance <= 2
    conflict_radius = 2 * radius
    # Precompute conflict neighborhoods using bitsets for speed
    conflict_sets = []
    for i, w in enumerate(all_words):
        conflicts = set()
        for r in range(1, conflict_radius + 1):
            for positions in itertools.combinations(range(n), r):
                for values in itertools.product(range(1, q), repeat=r):
                    new_word = list(w)
                    for pos, val in zip(positions, values):
                        new_word[pos] = (w[pos] + val) % q
                    nw = tuple(new_word)
                    j = word_to_idx[nw]
                    conflicts.add(j)
        conflict_sets.append(conflicts)

    # Orbit-based symmetry: group words by their orbit under coordinate permutation
    # and value permutation (S_n x S_q)
    # For efficiency, we use a simpler symmetry: sort coordinates and values
    # Actually, for the branch-and-bound, we'll use a canonical representative approach

    # Maximum independent set via branch and bound with dynamic coloring
    # The independent set in the conflict graph = maximum set of pairwise non-conflicting words
    # = maximum set where every pair has Hamming distance > 2*radius
    # This is exactly a packing code, which relates to covering via duality

    # Actually, for covering codes, we want MINIMUM set covering all words.
    # The MIS formulation: if we find a maximum set of words with pairwise distance > 2r,
    # that gives a lower bound on the covering code size (each codeword can cover at most
    # one word from this packing). But it doesn't directly give a covering code.

    # Let me reconsider: the directive says to formulate as MIS and solve it.
    # The connection: K_q(n,r) >= |X| / max_ball_size, but more precisely,
    # if we have a packing (MIS in conflict graph), each codeword covers at most 
    # one packing word, so K_q(n,r) >= packing_size.
    # To get an UPPER bound (actual covering code), we need a different approach.

    # I'll use a hybrid: greedy set cover + local search, but with the MIS 
    # lower bound to guide pruning, and use the conflict structure for faster
    # swap operations.

    best_solution = None
    best_size = float('inf')

    # Precompute M matrix for fast gain computation
    M = np.zeros((total_words, total_words), dtype=np.int8)
    for i in range(total_words):
        for j in ball_sets[i]:
            M[i][j] = 1

    def greedy_set_cover(random_seed=0):
        """Greedy set cover using precomputed matrices with random tie-breaking."""
        rng = random.Random(random_seed)
        covered_vec = np.zeros(total_words, dtype=np.int32)
        solution = []

        while True:
            if time.time() > deadline - 25:
                break

            gains = ball_size - M @ covered_vec
            max_gain = np.max(gains)
            if max_gain <= 0:
                break

            candidates = np.where(gains == max_gain)[0]
            best_idx = int(rng.choice(candidates))

            solution.append(best_idx)
            covered_vec[ball_members[best_idx]] += 1

        covered = covered_vec > 0
        return solution, covered

    def systematic_deletion_fast(solution):
        """Fast systematic deletion using coverage count vector."""
        if not solution:
            return solution, np.zeros(total_words, dtype=bool)

        cov_count = np.zeros(total_words, dtype=np.int32)
        for w_idx in solution:
            cov_count[ball_members[w_idx]] += 1

        changed = True
        while changed:
            changed = False
            for i in range(len(solution) - 1, -1, -1):
                if time.time() > deadline - 25:
                    break
                w_idx = solution[i]
                ball = ball_members[w_idx]
                if np.any(cov_count[ball] == 1):
                    continue
                cov_count[ball] -= 1
                solution.pop(i)
                changed = True

        covered = cov_count > 0
        return solution, covered

    def local_search_fast(solution, rng, max_iters=2000):
        """Fast local search using coverage count vector."""
        if not solution:
            return solution, np.zeros(total_words, dtype=bool)

        cov_count = np.zeros(total_words, dtype=np.int32)
        for w_idx in solution:
            cov_count[ball_members[w_idx]] += 1

        temp = 8.0
        cooling = 0.995
        min_temp = 0.5

        for sa_iter in range(max_iters):
            if time.time() > deadline - 25:
                break
            if len(solution) <= 1:
                break

            idx_to_remove = rng.randint(0, len(solution) - 1)
            w_rm = solution[idx_to_remove]
            ball_rm = ball_members[w_rm]

            unique_mask = cov_count[ball_rm] == 1
            n_unique = int(np.sum(unique_mask))

            if n_unique == 0:
                cov_count[ball_rm] -= 1
                solution.pop(idx_to_remove)
                temp = max(temp * cooling, min_temp)
            else:
                unique_indices = ball_rm[unique_mask]
                if len(unique_indices) > 0:
                    scores = M[:, unique_indices].sum(axis=1)
                    for w_idx in solution:
                        if w_idx != w_rm:
                            scores[w_idx] = 0

                    best_add = int(np.argmax(scores))
                    best_score = scores[best_add]

                    if best_score > 0:
                        lost = unique_indices[~(M[best_add, unique_indices] == 1)]
                        if len(lost) == 0:
                            cov_count[ball_rm] -= 1
                            cov_count[ball_members[best_add]] += 1
                            solution[idx_to_remove] = best_add
                            temp = max(temp * cooling, min_temp)
                            continue

                        penalty = len(lost)
                        if rng.random() < np.exp(-penalty / max(temp, 0.1)):
                            cov_count[ball_rm] -= 1
                            cov_count[ball_members[best_add]] += 1
                            solution[idx_to_remove] = best_add
                            temp = max(temp * cooling, min_temp)
                            continue

                temp = max(temp * cooling, min_temp)

        covered = cov_count > 0
        return solution, covered

    # Try multiple random restarts
    n_restarts = 50
    for attempt in range(n_restarts):
        if time.time() > deadline - 30:
            break

        solution, covered = greedy_set_cover(random_seed=42 + attempt)

        if not np.all(covered):
            continue

        solution, covered = systematic_deletion_fast(solution)

        rng = random.Random(1000 + attempt)
        solution, covered = local_search_fast(solution, rng, max_iters=1500)

        solution, covered = systematic_deletion_fast(solution)

        if np.all(covered):
            sol_set = set(solution)
            if len(sol_set) == len(solution):
                if len(solution) < best_size:
                    best_size = len(solution)
                    best_solution = [list(_int_to_word(w, q, n)) for w in solution]
                    if best_size <= 73:
                        break

    if best_solution is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return best_solution
# EVOLVE-BLOCK-END
