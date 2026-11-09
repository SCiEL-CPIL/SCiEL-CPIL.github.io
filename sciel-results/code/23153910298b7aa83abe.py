"""
Ternary covering code K_3(8,1) construction.

Strategy:
1. Phase 1: Greedy set cover - iteratively select the word that covers the most
   currently uncovered words. This breaks free from the product code structure.
2. Phase 2: Priority-queue-based removal - find and remove redundant codewords
   using coverage counts, similar to the current program's refinement.
"""

import time
import numpy as np
from itertools import product
import heapq


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q != 3 or n != 8 or radius != 1:
        # Fallback for general parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n-4)]
        else:
            return [list(w) for w in product(range(q), repeat=n)]

    total_words = q ** n  # 6561
    num_neighbors = 1 + n * (q - 1)  # 17

    # Precompute neighbors as a flat array
    # neighbors[i] contains the 17 indices of words within Hamming distance 1 of word i
    neighbors = np.zeros((total_words, num_neighbors), dtype=np.int32)

    # Precompute powers of 3 for fast index computation
    pow3 = [3 ** (n - 1 - i) for i in range(n)]

    for i in range(total_words):
        # Convert index to word digits
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

    # Phase 1: Greedy Set Cover
    uncovered = np.ones(total_words, dtype=bool)
    selected_indices = []

    # Time budget for Phase 1: use ~60% of available time
    phase1_deadline = start_time + (deadline - start_time) * 0.55

    while time.time() < phase1_deadline:
        # Count uncovered neighbors for each candidate
        counts = uncovered[neighbors].sum(axis=1)
        best_idx = np.argmax(counts)
        best_count = counts[best_idx]

        if best_count == 0:
            break

        selected_indices.append(best_idx)
        uncovered[neighbors[best_idx]] = False

        if not uncovered.any():
            break

    # Phase 2: Priority-queue-based removal
    # Build active set and coverage counts
    num_selected = len(selected_indices)
    active = [True] * num_selected
    cov_count = np.zeros(total_words, dtype=np.int32)

    for idx in selected_indices:
        for nb_idx in neighbors[idx]:
            cov_count[nb_idx] += 1

    # Priority queue: (unique_count, position)
    # unique_count = number of neighbors with cov_count == 1
    pq = []

    def compute_unique_count(pos):
        if not active[pos]:
            return float('inf')
        cw_idx = selected_indices[pos]
        count = 0
        for nb_idx in neighbors[cw_idx]:
            if cov_count[nb_idx] == 1:
                count += 1
        return count

    # Initialize priority queue
    for i in range(num_selected):
        uc = compute_unique_count(i)
        if uc < float('inf'):
            heapq.heappush(pq, (uc, i))

    # Removal loop
    phase2_deadline = deadline - 3.0

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

        # Update priority queue for affected codewords
        # Find codewords that share neighbors with the removed one
        # For efficiency, periodically rebuild the queue
        # Every 50 removals, rebuild
        if len(selected_indices) - sum(1 for a in active if a) % 50 == 0:
            pass  # Just let the existing queue handle it

    # Periodic full re-evaluation passes
    for pass_num in range(5):
        if time.time() > phase2_deadline:
            break

        pq = []
        for i in range(num_selected):
            if active[i]:
                uc = compute_unique_count(i)
                if uc < float('inf'):
                    heapq.heappush(pq, (uc, i))

        while pq and time.time() < phase2_deadline:
            uc, i = heapq.heappop(pq)
            if not active[i]:
                continue

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
            else:
                new_uc = compute_unique_count(i)
                if new_uc < float('inf'):
                    heapq.heappush(pq, (new_uc, i))

    # Extract remaining codewords
    result = []
    for i, idx in enumerate(selected_indices):
        if active[i]:
            # Convert index back to word
            temp = idx
            word = [0] * n
            for j in range(n - 1, -1, -1):
                word[j] = temp % q
                temp //= q
            result.append(word)

    return result
