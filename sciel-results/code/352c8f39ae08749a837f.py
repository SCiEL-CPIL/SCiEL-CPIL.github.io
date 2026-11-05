import numpy as np
import time
import random
import itertools


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists and bitsets
    neighbors_list = []
    neighbor_bitsets = []
    neighbor_sets = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        bs = 0
        for x in nb:
            bs |= (1 << x)
        neighbor_bitsets.append(bs)
        neighbor_sets.append(set(nb))

    # ---------- Phase 1: Greedy Set-Cover with vectorized gain computation ----------
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()

    while np.any(uncovered):
        if time.time() > deadline - 35:
            break

        # Compute gains: for each candidate codeword, how many uncovered points does it cover?
        gains = np.zeros(N, dtype=np.int32)
        uncovered_indices = np.where(uncovered)[0]
        for idx in uncovered_indices:
            gains[neighbors_list[idx]] += 1

        best_idx = int(np.argmax(gains))
        max_gain = gains[best_idx]

        if max_gain == 0:
            break

        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbors_list[best_idx]] = False

    # ---------- Phase 2: Local Search with bitset pruning ----------
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_list[idx]] += 1

    best_size = len(code)
    best_code = list(code)

    def try_remove_redundant():
        """Remove all redundant codewords."""
        removed = False
        for c in list(code):
            if np.min(counts[neighbors_list[c]]) > 1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
                removed = True
        return removed

    def find_replacement(c, crit_points):
        """Find best replacement for c that covers all critical points.
        Uses bitset intersection for fast candidate filtering."""
        # Start with all possible candidates
        # A valid replacement must be in the intersection of neighbor_sets of all crit_points
        # Use bitset intersection: for each crit_point, AND with its neighbor_bitset
        candidate_bitset = (1 << N) - 1  # All ones
        for p in crit_points:
            candidate_bitset &= neighbor_bitsets[p]

        # Remove current code words from candidates
        for cw in code:
            candidate_bitset &= ~(1 << cw)

        if candidate_bitset == 0:
            return -1

        # Score candidates: maximize redundancy creation (points going from count 1 to 2)
        best_w = -1
        best_score = -1

        # Extract candidate indices from bitset
        # For efficiency, only check a subset if there are too many
        candidates = []
        temp = candidate_bitset
        while temp:
            # Find lowest set bit
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            temp ^= lsb

        # If too many candidates, sample
        if len(candidates) > 200:
            random.shuffle(candidates)
            candidates = candidates[:200]

        for w in candidates:
            # Check validity (redundant with bitset check but safer)
            valid = True
            for k in crit_points:
                if k not in neighbor_sets[w]:
                    valid = False
                    break
            if not valid:
                continue

            # Score: count of neighbors of w that have count == 1 (creates redundancy)
            score = int(np.sum(counts[neighbors_list[w]] == 1))
            if score > best_score:
                best_score = score
                best_w = w

        return best_w

    def try_swap(c):
        """Try to swap codeword c with a better-positioned codeword."""
        if np.min(counts[neighbors_list[c]]) > 1:
            return False

        crit_points = [i for i in neighbors_list[c] if counts[i] == 1]
        if not crit_points:
            return False

        # Strategy 1: Find replacement covering all critical points
        best_w = find_replacement(c, crit_points)

        if best_w != -1:
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1

            code.append(best_w)
            code_set.add(best_w)
            counts[neighbors_list[best_w]] += 1
            return True

        # Strategy 2: Relocation - move c to cover would-be-uncovered points
        temp_counts = counts.copy()
        temp_counts[neighbors_list[c]] -= 1
        would_be_uncovered = [i for i in neighbors_list[c] if temp_counts[i] == 0]

        if not would_be_uncovered:
            return False

        all_needed = set(crit_points) | set(would_be_uncovered)

        # Find candidates covering all needed points
        candidate_bitset = (1 << N) - 1
        for p in all_needed:
            candidate_bitset &= neighbor_bitsets[p]

        for cw in code:
            candidate_bitset &= ~(1 << cw)

        if candidate_bitset == 0:
            return False

        candidates = []
        temp = candidate_bitset
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            temp ^= lsb

        if len(candidates) > 200:
            random.shuffle(candidates)
            candidates = candidates[:200]

        best_w = -1
        best_score = -1
        for w in candidates:
            valid = True
            for k in all_needed:
                if k not in neighbor_sets[w]:
                    valid = False
                    break
            if not valid:
                continue
            score = int(np.sum(counts[neighbors_list[w]] == 1))
            if score > best_score:
                best_score = score
                best_w = w

        if best_w != -1:
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1

            code.append(best_w)
            code_set.add(best_w)
            counts[neighbors_list[best_w]] += 1
            return True

        return False

    # Iterative local search with restarts
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False

        # Removal pass
        if try_remove_redundant():
            improved = True

        # Swap pass
        code_indices = list(range(len(code)))
        random.shuffle(code_indices)

        for idx in code_indices:
            if time.time() > deadline - 15:
                break
            if idx >= len(code):
                continue
            c = code[idx]
            if try_swap(c):
                improved = True
                # After a swap, re-check for redundancies
                if try_remove_redundant():
                    pass
                break  # Restart swap pass

    # Track best
    if len(code) < best_size:
        best_size = len(code)
        best_code = list(code)

    # ---------- Phase 3: Simulated Annealing-style polish ----------
    # Try random perturbations (remove + repair) with acceptance criterion
    current_size = len(code)
    best_size_sa = best_size
    best_code_sa = list(best_code)

    # Reset to best
    code = list(best_code)
    code_set = set(code)
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_list[idx]] += 1

    temperature = 2.0  # Start with some tolerance
    cooling = 0.95

    sa_improved = True
    while sa_improved and time.time() < deadline - 5:
        sa_improved = False
        temperature *= cooling

        # Try a random perturbation: remove a random codeword, repair
        if len(code) > 50:
            c = random.choice(code)
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1

            # Find uncovered
            uncovered_mask = counts == 0
            uncovered_set = set(np.where(uncovered_mask)[0].tolist())

            # Greedy repair
            added = []
            temp_uncovered = set(uncovered_set)
            while temp_uncovered and time.time() < deadline - 5:
                # Build candidate pool
                candidate_pool = set()
                for w in temp_uncovered:
                    candidate_pool.update(neighbor_sets[w])
                candidate_pool -= code_set

                if not candidate_pool:
                    break

                best_c = -1
                best_count = 0
                for c2 in candidate_pool:
                    cnt = sum(1 for x in neighbors_list[c2] if x in temp_uncovered)
                    if cnt > best_count:
                        best_count = cnt
                        best_c = c2
                    if best_count == len(temp_uncovered):
                        break

                if best_c == -1:
                    break

                added.append(best_c)
                code_set.add(best_c)
                counts[neighbors_list[best_c]] += 1
                for w in neighbors_list[best_c]:
                    temp_uncovered.discard(w)

            code.extend(added)

            # Check if valid
            if np.any(counts == 0):
                # Invalid - revert
                for c2 in added:
                    code.remove(c2)
                    code_set.remove(c2)
                    counts[neighbors_list[c2]] -= 1
                code.append(c)
                code_set.add(c)
                counts[neighbors_list[c]] += 1
                continue

            # Accept or reject based on size change
            size_change = len(code) - current_size
            if size_change <= 0:
                current_size = len(code)
                if len(code) < best_size_sa:
                    best_size_sa = len(code)
                    best_code_sa = list(code)
                    sa_improved = True
            else:
                # Accept with probability exp(-delta / T)
                if random.random() < np.exp(-size_change / max(temperature, 0.01)):
                    current_size = len(code)
                else:
                    # Revert
                    for c2 in added:
                        code.remove(c2)
                        code_set.remove(c2)
                        counts[neighbors_list[c2]] -= 1
                    code.append(c)
                    code_set.add(c)
                    counts[neighbors_list[c]] += 1

            # Final cleanup
            try_remove_redundant()

    if best_size_sa < best_size:
        best_size = best_size_sa
        best_code = list(best_code_sa)

    return [idx_to_tuple(c) for c in best_code]
