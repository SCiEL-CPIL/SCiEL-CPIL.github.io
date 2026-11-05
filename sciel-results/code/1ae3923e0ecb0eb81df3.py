import itertools
import numpy as np
import time
import random
from collections import defaultdict

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 118.0

    total_words = q ** n  # 6561

    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)

    # Precompute neighbor lists (Hamming distance <= radius)
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)

    # Precompute neighbor masks as boolean arrays for swap intersection
    neighbor_mask = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        neighbor_mask[i, neighbors[i]] = True

    # Precompute reverse neighbors: for each point, which codewords cover it
    # rev_neighbors[i] = list of points j such that i is in neighbors[j]
    # This is the same as neighbor_mask transposed, but we'll use a set-based approach for speed
    # Actually, since neighbor_mask is symmetric (Hamming distance is symmetric), rev_neighbors[i] = neighbors[i]
    # But let's verify: if i is in neighbors[j], then j is in neighbors[i] (symmetric relation)
    # Yes, Hamming distance is symmetric, so neighbor_mask is symmetric.

    # Helper: greedy construction
    def greedy_construct(seed_offset=0):
        covered = np.zeros(total_words, dtype=bool)
        code_list = []
        while not covered.all() and time.time() < deadline - 40.0:
            # Compute potential for all candidates: count uncovered neighbors
            # Use vectorized: for each uncovered point, add to all its neighbors
            uncovered_idx = np.where(~covered)[0]
            if len(uncovered_idx) == 0:
                break
            # potential[i] = number of uncovered points in neighbors[i]
            # Efficient: sum over uncovered points of neighbor_mask[point]
            potential = np.zeros(total_words, dtype=np.int32)
            # Process in batches to avoid huge temporary arrays
            batch_size = 500
            for s in range(0, len(uncovered_idx), batch_size):
                batch = uncovered_idx[s:s+batch_size]
                potential += neighbor_mask[batch].sum(axis=0).astype(np.int32)
            c = int(np.argmax(potential))
            if potential[c] == 0:
                break
            code_list.append(c)
            covered[neighbors[c]] = True
        return code_list

    # Helper: local search improvement with slice-based coordinate-wise moves
    def local_search(code, deadline_check):
        current_code = code[:]
        current_size = len(current_code)
        best_code = current_code[:]
        best_size = current_size

        current_cov_count = np.zeros(total_words, dtype=np.int32)
        for c in current_code:
            current_cov_count[neighbors[c]] += 1

        in_code = np.zeros(total_words, dtype=bool)
        for c in current_code:
            in_code[c] = True

        random.seed(42)
        stuck_counter = 0

        while time.time() < deadline_check:
            if current_size == 0:
                break

            # Try removal
            idx = random.randint(0, current_size - 1)
            c = current_code[idx]

            if np.all(current_cov_count[neighbors[c]] > 1):
                current_cov_count[neighbors[c]] -= 1
                in_code[c] = False
                current_code.pop(idx)
                current_size -= 1
                stuck_counter = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                stuck_counter += 1

            # Try swap if stuck
            if stuck_counter > 3 and time.time() < deadline_check - 5.0 and current_size > 0:
                c1_idx = random.randint(0, current_size - 1)
                c1 = current_code[c1_idx]

                risky_points = [x for x in neighbors[c1] if current_cov_count[x] == 1]

                if not risky_points:
                    current_cov_count[neighbors[c1]] -= 1
                    in_code[c1] = False
                    current_code.pop(c1_idx)
                    current_size -= 1
                    stuck_counter = 0
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                else:
                    # Find c2 that covers all risky points and is not in code
                    if len(risky_points) == 1:
                        candidate_mask = neighbor_mask[risky_points[0]].copy()
                    else:
                        candidate_mask = neighbor_mask[risky_points[0]].copy()
                        for x in risky_points[1:]:
                            candidate_mask &= neighbor_mask[x]
                            if not candidate_mask.any():
                                break

                    candidate_mask[c1] = False
                    candidate_mask[in_code] = False

                    valid_candidates = np.where(candidate_mask)[0]

                    if len(valid_candidates) > 0:
                        c2 = int(random.choice(valid_candidates))
                        current_cov_count[neighbors[c1]] -= 1
                        in_code[c1] = False
                        current_code[c1_idx] = c2
                        in_code[c2] = True
                        current_cov_count[neighbors[c2]] += 1
                        stuck_counter = 0
                    else:
                        pass

                stuck_counter = 0

            # Slice-based coordinate-wise move: modify one coordinate of a codeword
            # This is a 1-flip within the same "slice" (fixing all other 7 coordinates)
            if stuck_counter > 5 and time.time() < deadline_check - 5.0 and current_size > 0:
                c1_idx = random.randint(0, current_size - 1)
                c1 = current_code[c1_idx]
                d = digits_of[c1]
                
                # Pick a random coordinate to change
                coord = random.randint(0, n - 1)
                orig_val = d[coord]
                
                # Try changing this coordinate to each other value
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    
                    new_d = list(d)
                    new_d[coord] = new_val
                    new_idx = 0
                    for j in range(n):
                        new_idx = new_idx * q + new_d[j]
                    
                    # Check if this swap (remove c1, add new_idx) maintains coverage
                    # c1's neighbors that would lose coverage
                    # new_idx's neighbors that would gain coverage
                    # For the swap to be valid, every point that c1 uniquely covers must be covered by new_idx
                    
                    # Points uniquely covered by c1
                    c1_unique = [x for x in neighbors[c1] if current_cov_count[x] == 1]
                    
                    if not c1_unique:
                        # Can just remove c1
                        current_cov_count[neighbors[c1]] -= 1
                        in_code[c1] = False
                        current_code.pop(c1_idx)
                        current_size -= 1
                        stuck_counter = 0
                        if current_size < best_size:
                            best_size = current_size
                            best_code = current_code[:]
                        break
                    
                    # Check if new_idx covers all of c1_unique
                    all_covered = True
                    for x in c1_unique:
                        if not neighbor_mask[new_idx, x]:
                            all_covered = False
                            break
                    
                    if all_covered and not in_code[new_idx]:
                        # Perform the swap
                        current_cov_count[neighbors[c1]] -= 1
                        in_code[c1] = False
                        current_code[c1_idx] = new_idx
                        in_code[new_idx] = True
                        current_cov_count[neighbors[new_idx]] += 1
                        stuck_counter = 0
                        break

        return best_code, best_size

    # Helper: final polish with multiple passes
    def final_polish(code):
        final_code = code[:]
        final_cov_count = np.zeros(total_words, dtype=np.int32)
        for idx in final_code:
            final_cov_count[neighbors[idx]] += 1

        improved = True
        passes = 0
        while improved and time.time() < deadline - 2.0 and passes < 10:
            improved = False
            passes += 1
            for i in range(len(final_code) - 1, -1, -1):
                c = final_code[i]
                if np.all(final_cov_count[neighbors[c]] > 1):
                    final_cov_count[neighbors[c]] -= 1
                    final_code.pop(i)
                    improved = True

        if not np.all(final_cov_count > 0):
            return code
        return final_code

    # Main: try multiple restarts with different seeds
    best_code_overall = None
    best_size_overall = float('inf')

    # First, do a greedy + local search
    t0 = time.time()
    greedy_code = greedy_construct()
    t1 = time.time()

    if len(greedy_code) < best_size_overall:
        best_code_overall = greedy_code[:]
        best_size_overall = len(greedy_code)

    # Local search on the greedy result
    ls_deadline = deadline - 30.0
    polished, size = local_search(greedy_code, ls_deadline)
    if size < best_size_overall:
        best_code_overall = polished[:]
        best_size_overall = size

    # Try additional restarts if time permits
    attempt = 0
    while time.time() < deadline - 60.0:
        attempt += 1
        random.seed(1000 + attempt)
        # Perturb: take best code, remove some random elements, re-greedy
        if best_code_overall is None:
            break

        perturbed = best_code_overall[:]
        # Remove 5-15 random elements
        num_remove = min(len(perturbed), random.randint(5, 15))
        remove_indices = random.sample(range(len(perturbed)), num_remove)
        for ri in sorted(remove_indices, reverse=True):
            perturbed.pop(ri)

        # Re-greedy to fill gaps
        covered = np.zeros(total_words, dtype=bool)
        for c in perturbed:
            covered[neighbors[c]] = True

        while not covered.all() and time.time() < deadline - 50.0:
            uncovered_idx = np.where(~covered)[0]
            if len(uncovered_idx) == 0:
                break
            potential = np.zeros(total_words, dtype=np.int32)
            batch_size = 500
            for s in range(0, len(uncovered_idx), batch_size):
                batch = uncovered_idx[s:s+batch_size]
                potential += neighbor_mask[batch].sum(axis=0).astype(np.int32)
            c = int(np.argmax(potential))
            if potential[c] == 0:
                break
            perturbed.append(c)
            covered[neighbors[c]] = True

        # Local search on perturbed
        ls_dl = deadline - 40.0
        polished, size = local_search(perturbed, ls_dl)
        if size < best_size_overall:
            best_code_overall = polished[:]
            best_size_overall = size

    # Final polish
    if best_code_overall is not None:
        best_code_overall = final_polish(best_code_overall)

    return [list(digits_of[i]) for i in best_code_overall]
