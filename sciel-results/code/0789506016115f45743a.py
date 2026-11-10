"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin: 55s out of 60s/90s limit

    def count_nonzero(poly):
        return int(np.count_nonzero(poly))

    def poly_square_from_dense(dense):
        """Square a dense polynomial array using convolution."""
        if len(dense) == 0:
            return np.array([0], dtype=np.int64)
        return np.convolve(dense, dense, mode='full')

    def terms_to_dense(terms):
        """Convert list of (pos, val) to dense numpy array."""
        if not terms:
            return np.array([0], dtype=np.int64)
        max_deg = max(pos for pos, _ in terms)
        dense = np.zeros(max_deg + 1, dtype=np.int64)
        for pos, val in terms:
            dense[pos] = val
        return dense

    def evaluate(terms):
        """terms: list of (position, value) tuples, exactly 12 nonzero"""
        if len(terms) != 12:
            return 1000
        positions = [pos for pos, _ in terms]
        if len(set(positions)) != 12:
            return 1000
        for pos, val in terms:
            if val == 0 or abs(val) > 64 or pos > 40:
                return 1000

        dense = terms_to_dense(terms)
        sq = poly_square_from_dense(dense)
        return count_nonzero(sq)

    def random_initial():
        """Generate a random valid 12-term polynomial"""
        positions = random.sample(range(41), 12)
        values = [random.choice([1, -1]) * random.randint(1, 64) for _ in range(12)]
        return list(zip(positions, values))

    def neighbor(terms):
        """Generate a neighboring solution by modifying one term"""
        terms = list(terms)
        move_type = random.random()

        if move_type < 0.4:
            # Change position of one term
            idx = random.randint(0, 11)
            old_pos = terms[idx][0]
            new_pos = random.randint(0, 40)
            if new_pos == old_pos:
                return terms
            # Check no collision
            for i, (pos, val) in enumerate(terms):
                if i != idx and pos == new_pos:
                    return terms
            terms[idx] = (new_pos, terms[idx][1])
        elif move_type < 0.7:
            # Change value of one term
            idx = random.randint(0, 11)
            new_val = random.choice([1, -1]) * random.randint(1, 64)
            terms[idx] = (terms[idx][0], new_val)
        elif move_type < 0.9:
            # Change sign of one term
            idx = random.randint(0, 11)
            terms[idx] = (terms[idx][0], -terms[idx][1])
        else:
            # Swap positions of two terms
            i, j = random.sample(range(12), 2)
            terms[i], terms[j] = (terms[j][0], terms[i][1]), (terms[i][0], terms[j][1])

        return terms

    def targeted_search():
        """Try structured patterns that may yield sparse squares."""
        best_terms = None
        best_score = 1000

        # Pattern 1: Geometric-like spacing with alternating signs
        for base in range(0, 30, 2):
            for step in range(1, 4):
                positions = [base + i * step for i in range(12)]
                if positions[-1] > 40:
                    continue
                for sign_pattern in [[1] * 12, [1, -1] * 6, [1, 1, -1] * 4]:
                    values = [sign_pattern[i] * random.randint(1, 64) for i in range(12)]
                    terms = list(zip(positions, values))
                    score = evaluate(terms)
                    if score < best_score:
                        best_score = score
                        best_terms = terms

        # Pattern 2: Clusters of terms
        for cluster_size in [3, 4, 6]:
            num_clusters = 12 // cluster_size
            if num_clusters * cluster_size != 12:
                continue
            base_positions = random.sample(range(0, 41 - cluster_size), num_clusters)
            for _ in range(50):
                positions = []
                for bp in sorted(base_positions):
                    offsets = random.sample(range(0, cluster_size), cluster_size)
                    positions.extend(bp + o for o in offsets)
                if len(set(positions)) != 12 or any(p > 40 for p in positions):
                    continue
                values = [random.choice([1, -1]) * random.randint(1, 64) for _ in range(12)]
                terms = list(zip(positions, values))
                score = evaluate(terms)
                if score < best_score:
                    best_score = score
                    best_terms = terms

        return best_terms, best_score

    best_terms = random_initial()
    best_score = evaluate(best_terms)

    # Try targeted search first
    tt_terms, tt_score = targeted_search()
    if tt_terms is not None and tt_score < best_score:
        best_terms = tt_terms
        best_score = tt_score

    # Random Restart with Local Search
    max_iterations = 2000000  # High cap, but time is the real limit

    for _ in range(max_iterations):
        if time.time() > deadline:
            break

        # Random restart or continue from current best?
        if random.random() < 0.3:
            current_terms = random_initial()
            current_score = evaluate(current_terms)
        else:
            current_terms = best_terms[:]
            current_score = best_score

        # Local search phase: try to improve current_terms
        improved = True
        local_iter = 0
        while improved and local_iter < 15 and time.time() < deadline:
            improved = False
            local_iter += 1
            for _ in range(20):
                if time.time() > deadline:
                    break
                new_terms = neighbor(current_terms)
                new_score = evaluate(new_terms)
                if new_score < current_score:
                    current_terms = new_terms
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_terms = current_terms[:]
                        if best_score <= 12:
                            break
            if best_score <= 12:
                break

        if best_score <= 12:
            break

    # Convert best terms back to coefficient list
    if not best_terms:
        return [1]  # Fallback, though should not happen

    max_pos = max(pos for pos, _ in best_terms)
    coeff_list = [0] * (max_pos + 1)
    for pos, val in best_terms:
        coeff_list[pos] = val

    # Ensure length <= 41 and nonzero final coefficient
    if len(coeff_list) > 41:
        coeff_list = coeff_list[:41]
    if coeff_list[-1] == 0:
        # Find last nonzero
        while coeff_list and coeff_list[-1] == 0:
            coeff_list.pop()
        if not coeff_list:
            coeff_list = [1]

    return coeff_list
# EVOLVE-BLOCK-END
