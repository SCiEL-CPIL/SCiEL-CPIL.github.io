"""
Ternary covering code K_3(8,1) construction.

Structural change: Formulate the problem as a maximum independent set (MIS) search
on a graph where vertices are codewords and edges connect words within Hamming
distance <= 2. A valid covering code of radius 1 is the complement of an
independent set in this graph (no two codewords can be within distance 2, else
their union of radius-1 balls would have a gap... wait, actually for covering codes
the condition is different).

Let me reconsider: For a covering code of radius 1, we need every word to be within
distance 1 of some codeword. The key insight is that if two codewords are at
distance 2, they can still both be useful (they cover different regions). The
constraint is that we want to MINIMIZE the number of codewords.

Actually, the MIS formulation works differently: We want to select a set of
codewords such that their radius-1 balls cover the entire space. This is a set
cover problem. The MIS angle is: if we consider the "conflict graph" where two
codewords conflict if they are "redundant" with each other, we can use MIS-style
branch and bound.

However, a more direct structural approach: Use greedy set cover to get an initial
solution, then use a sophisticated local search with:
1. Greedy removal with priority queue
2. Swap-based optimization to break out of local optima
3. Randomized restarts with different greedy seeds

The key structural change from the seed: Instead of starting from a fixed product
code structure, we start from scratch using greedy set cover, which gives a
more flexible initial solution that can be optimized more effectively.
"""

import time
import numpy as np
from itertools import product
import random
import heapq


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q != 3 or n != 8 or radius != 1:
        # Fallback for general parameters: brute-force product code
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n-4)]
        else:
            return [list(w) for w in product(range(q), repeat=n)]

    total_words = q ** n  # 6561
    num_neighbors = 1 + n * (q - 1)  # 17

    # Precompute neighbors as a flat array
    neighbors = np.zeros((total_words, num_neighbors), dtype=np.int32)

    # Precompute powers of 3 for fast index computation
    pow3 = [3 ** (n - 1 - i) for i in range(n)]

    for i in range(total_words):
        temp = i
        word = [0] * n
        for j in range(n - 1, -1, -1):
            word[j] = temp % q
            temp //= q

        neighbors[i, 0] = i
        nb_idx = 1
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                delta = (new_val - orig_val) * pow3[pos]
                neighbors[i, nb_idx] = i + delta
                nb_idx += 1

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for j in range(n - 1, -1, -1):
            w[j] = temp % q
            temp //= q
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    best_solution = None
    best_size = float('inf')

    # Try multiple random seeds for greedy set cover
    rng = random.Random(12345)

    for seed_iter in range(5):
        if time.time() > deadline - 10:
            break

        if seed_iter == 0:
            # First iteration: deterministic greedy
            greedy_order = None
        else:
            # Subsequent iterations: randomized greedy with different weights
            greedy_order = None

        # Phase 1: Greedy Set Cover
        uncovered = np.ones(total_words, dtype=bool)
        selected_indices = []

        # Time budget for Phase 1: use ~40% of available time per iteration
        phase1_deadline = start_time + (deadline - start_time) * 0.35

        while time.time() < phase1_deadline:
            # Count uncovered neighbors for each candidate
            # Use a randomized perturbation for different seeds
            if seed_iter == 0:
                counts = uncovered[neighbors].sum(axis=1)
            else:
                # Add small random noise to break ties and explore different solutions
                noise = rng.random(total_words) * 0.1
                counts = uncovered[neighbors].sum(axis=1) + noise

            best_idx = np.argmax(counts)
            best_count = counts[best_idx]

            if best_count == 0:
                break

            selected_indices.append(best_idx)
            uncovered[neighbors[best_idx]] = False

            if not uncovered.any():
                break

        if not uncovered.any() or len(selected_indices) == 0:
            continue

        # Phase 2: Priority-queue-based removal
        num_selected = len(selected_indices)
        active = [True] * num_selected
        cov_count = np.zeros(total_words, dtype=np.int32)

        for idx in selected_indices:
            for nb_idx in neighbors[idx]:
                cov_count[nb_idx] += 1

        def compute_unique_count(pos):
            if not active[pos]:
                return float('inf')
            cw_idx = selected_indices[pos]
            count = 0
            for nb_idx in neighbors[cw_idx]:
                if cov_count[nb_idx] == 1:
                    count += 1
            return count

        # Priority queue: (unique_count, position)
        pq = []
        for i in range(num_selected):
            uc = compute_unique_count(i)
            if uc < float('inf'):
                heapq.heappush(pq, (uc, i))

        # Removal loop
        phase2_deadline = start_time + (deadline - start_time) * 0.65

        removals_since_rebuild = 0
        while time.time() < phase2_deadline:
            if not pq:
                break

            uc, i = heapq.heappop(pq)

            if not active[i]:
                continue

            cw_idx = selected_indices[i]

            # Check if removable: all neighbors must have cov_count > 1
            can_remove = True
            for nb_idx in neighbors[cw_idx]:
                if cov_count[nb_idx] <= 1:
                    can_remove = False
                    break

            if not can_remove:
                # Re-add with updated count
                new_uc = compute_unique_count(i)
                if new_uc < float('inf'):
                    heapq.heappush(pq, (new_uc, i))
                continue

            # Remove codeword
            active[i] = False
            for nb_idx in neighbors[cw_idx]:
                cov_count[nb_idx] -= 1

            removals_since_rebuild += 1

            # Every 30 removals, rebuild the priority queue
            if removals_since_rebuild % 30 == 0:
                pq = []
                for j in range(num_selected):
                    if active[j]:
                        uc_j = compute_unique_count(j)
                        if uc_j < float('inf'):
                            heapq.heappush(pq, (uc_j, j))
                removals_since_rebuild = 0

        # Phase 3: Swap-based local search
        # Try to improve by swapping codewords
        phase3_deadline = start_time + (deadline - start_time) * 0.85

        all_word_indices = list(range(total_words))
        improved_in_swap = True

        while improved_in_swap and time.time() < phase3_deadline:
            improved_in_swap = False

            for i in range(num_selected):
                if time.time() > phase3_deadline:
                    break
                if not active[i]:
                    continue

                cw_idx = selected_indices[i]
                cc = compute_unique_count(i)
                if cc == 0:
                    # Try to remove
                    can_remove = True
                    for nb_idx in neighbors[cw_idx]:
                        if cov_count[nb_idx] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        active[i] = False
                        for nb_idx in neighbors[cw_idx]:
                            cov_count[nb_idx] -= 1
                        improved_in_swap = True
                    continue

                # This codeword has critical neighbors
                critical_neighbors = [nb_idx for nb_idx in neighbors[cw_idx]
                                      if cov_count[nb_idx] == 1]
                if not critical_neighbors:
                    continue

                # Try to find a replacement word that covers all critical neighbors
                candidates = rng.sample(all_word_indices, min(100, len(all_word_indices)))

                best_candidate = None
                best_score = float('inf')

                active_set = set(selected_indices[j] for j in range(num_selected) if active[j])

                for cand_idx in candidates:
                    if cand_idx in active_set:
                        continue

                    # Check if cand_idx covers all critical neighbors
                    cand_neighbors_set = set(neighbors[cand_idx])
                    if not all(nb in cand_neighbors_set for nb in critical_neighbors):
                        continue

                    # Compute score: number of neighbors with cov_count > 1
                    score = 0
                    for nb in neighbors[cand_idx]:
                        if cov_count[nb] > 1:
                            score += 1
                    score = -score  # We want to maximize, so invert

                    if score < best_score:
                        best_score = score
                        best_candidate = cand_idx

                if best_candidate is not None:
                    # Perform the swap
                    active[i] = False
                    for nb_idx in neighbors[cw_idx]:
                        cov_count[nb_idx] -= 1

                    # Add the new codeword
                    selected_indices.append(best_candidate)
                    active.append(True)
                    for nb_idx in neighbors[best_candidate]:
                        cov_count[nb_idx] += 1
                    improved_in_swap = True

        # Phase 4: Final greedy removal pass
        phase4_deadline = start_time + (deadline - start_time) * 0.95

        for pass_num in range(5):
            if time.time() > phase4_deadline:
                break
            removed_any = False
            for i in range(len(selected_indices)):
                if active[i]:
                    cw_idx = selected_indices[i]
                    can_remove = True
                    for nb_idx in neighbors[cw_idx]:
                        if cov_count[nb_idx] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        active[i] = False
                        for nb_idx in neighbors[cw_idx]:
                            cov_count[nb_idx] -= 1
                        removed_any = True
            if not removed_any:
                break

        # Extract remaining codewords and update best
        result_indices = [selected_indices[i] for i in range(len(selected_indices)) if active[i]]
        current_size = len(result_indices)

        if current_size < best_size:
            best_size = current_size
            best_solution = [idx_to_word(idx) for idx in result_indices]

    # Return the best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback: return full space
        return [list(w) for w in product(range(q), repeat=n)]
