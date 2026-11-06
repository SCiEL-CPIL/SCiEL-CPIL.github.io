import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin under 150s kill

    N = 1000

    def sum_counts_from_mask(mask):
        m = mask.astype(np.int32)
        c = np.convolve(m, m, mode='full')
        s = np.arange(2 * N + 1)
        diag = np.zeros(2 * N + 1, dtype=np.int32)
        even = (s % 2 == 0)
        half = s // 2
        valid = even & (half <= N)
        diag[valid] = m[half[valid]]
        counts = (c + diag) // 2
        return counts

    def is_valid(mask):
        counts = sum_counts_from_mask(mask)
        return np.max(counts) <= 2

    def greedy_repair(mask):
        """Greedily remove elements to fix violations until valid."""
        mask = mask.copy()
        counts = sum_counts_from_mask(mask)
        while np.max(counts) > 2:
            # Find sums with > 2 representations
            bad_sums = np.where(counts > 2)[0]
            if len(bad_sums) == 0:
                break
            # Pick the worst sum
            worst_sum = bad_sums[np.argmax(counts[bad_sums])]
            # Find all pairs (a, b) with a <= b, a + b = worst_sum, both in mask
            a_min = max(1, worst_sum - N)
            a_max = min(N, worst_sum // 2)
            pairs = []
            for a in range(a_min, a_max + 1):
                b = worst_sum - a
                if a <= b and mask[a] and mask[b]:
                    pairs.append((a, b))
            if not pairs:
                break
            # Remove one element that appears in the most bad-sum pairs
            # Count how many bad sums each element participates in
            elem_bad_count = {}
            for s in bad_sums:
                a_min = max(1, s - N)
                a_max = min(N, s // 2)
                for a in range(a_min, a_max + 1):
                    b = s - a
                    if a <= b and mask[a] and mask[b]:
                        elem_bad_count[a] = elem_bad_count.get(a, 0) + 1
                        if b != a:
                            elem_bad_count[b] = elem_bad_count.get(b, 0) + 1
            # Remove element with highest bad count
            if elem_bad_count:
                worst_elem = max(elem_bad_count, key=elem_bad_count.get)
                mask[worst_elem] = False
                counts = sum_counts_from_mask(mask)
            else:
                break
        return mask

    def probabilistic_construct(p, rng):
        """Randomly select each integer with probability p, then greedily repair."""
        mask = np.zeros(N + 1, dtype=bool)
        for i in range(1, N + 1):
            if rng.random() < p:
                mask[i] = True
        mask = greedy_repair(mask)
        return mask

    def deterministic_greedy():
        mask = np.zeros(N + 1, dtype=bool)
        for x in range(1, N + 1):
            mask[x] = True
            if not is_valid(mask):
                mask[x] = False
        return mask

    def local_search(mask, time_budget, rng):
        """Local search with add/swap moves to maximize size."""
        current_mask = mask.copy()
        current_size = int(np.sum(current_mask))
        current_counts = sum_counts_from_mask(current_mask)
        
        t0 = time.time()
        
        while time.time() < t0 + time_budget:
            # Try add move
            if rng.random() < 0.6:
                b = rng.randint(1, N)
                if not current_mask[b]:
                    current_mask[b] = True
                    new_counts = sum_counts_from_mask(current_mask)
                    if np.max(new_counts) <= 2:
                        current_size += 1
                        current_counts = new_counts
                    else:
                        current_mask[b] = False
            else:
                # Swap move
                present = np.where(current_mask)[0]
                present = present[present >= 1]
                if len(present) == 0:
                    continue
                a = present[rng.randrange(len(present))]
                b = rng.randint(1, N)
                attempts = 0
                while current_mask[b] and attempts < 10:
                    b = rng.randint(1, N)
                    attempts += 1
                if current_mask[b]:
                    continue
                current_mask[a] = False
                current_mask[b] = True
                new_counts = sum_counts_from_mask(current_mask)
                if np.max(new_counts) <= 2:
                    current_counts = new_counts
                else:
                    current_mask[a] = True
                    current_mask[b] = False
        
        return current_mask, int(np.sum(current_mask))

    best_mask = None
    best_size = 0

    # Phase 1: Deterministic greedy baseline
    greedy_mask = deterministic_greedy()
    best_mask = greedy_mask
    best_size = int(np.sum(best_mask))

    # Phase 2: Try multiple probabilistic constructions with different p values
    rng = random.Random(42)
    
    # Theoretical density is ~3/2 per unit, so p should be around 0.3-0.5
    # For N=1000, we want ~600-750 elements
    p_values = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    
    time_per_prob = (deadline - start_time - 30) / (len(p_values) + 1)
    
    for p in p_values:
        if time.time() > deadline - 30:
            break
        prob_mask = probabilistic_construct(p, rng)
        prob_size = int(np.sum(prob_mask))
        
        if prob_size > best_size:
            best_size = prob_size
            best_mask = prob_mask
        
        # Local search on this construction
        remaining = deadline - 30 - time.time()
        if remaining > 5:
            ls_mask, ls_size = local_search(prob_mask, remaining * 0.7, rng)
            if ls_size > best_size:
                best_size = ls_size
                best_mask = ls_mask

    # Phase 3: Final local search on best found so far
    remaining = deadline - 5 - time.time()
    if remaining > 5:
        final_mask, final_size = local_search(best_mask, remaining, rng)
        if final_size > best_size:
            best_size = final_size
            best_mask = final_mask

    result = [int(x) for x in np.where(best_mask)[0] if x >= 1]
    return result
