"""Football pool problem for 7 matches: ternary covering code K_3(7,1).

Strategy:
  1. Direct-sum construction with the ternary tetracode (9 * 3**(n-4) words) as initial seed.
  2. Fast greedy set-cover using vectorized gain accumulation from uncovered points.
  3. Intensive local search: removal of redundant codewords, 1-swap and 2-swap moves,
     with multiple restarts and randomized perturbation to escape local optima.
  4. Fallback greedy for general cases.
"""

import time
import itertools
import random
import numpy as np


def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 140  # safety margin before 150s kill

    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # General brute-force small case
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

    # Precompute neighbor lists as numpy arrays for fast operations
    neighbors_list = []
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
        neighbor_sets.append(set(nb))

    # ---------- Phase 1: Fast Greedy Set-Cover ----------
    def greedy_cover(seed_code=None):
        """Greedy set cover. If seed_code is provided, start from it."""
        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()

        if seed_code is not None:
            for c in seed_code:
                if c not in code_set:
                    code.append(c)
                    code_set.add(c)
                    uncovered[neighbors_list[c]] = False

        while np.any(uncovered):
            if time.time() > deadline - 25:
                break

            # Compute gains by accumulating from uncovered points
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

        return code

    # ---------- Phase 2: Local Search ----------
    def compute_counts(code):
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            counts[neighbors_list[c]] += 1
        return counts

    def is_valid(code):
        counts = compute_counts(code)
        return np.all(counts >= 1)

    def try_remove_redundant(code, code_set, counts):
        """Remove all codewords whose neighborhood is covered by >1 codeword."""
        removed = False
        for c in list(code):
            if np.min(counts[neighbors_list[c]]) > 1:
                code.remove(c)
                code_set.discard(c)
                counts[neighbors_list[c]] -= 1
                removed = True
        return removed

    def try_1_swap(code, code_set, counts, randomize=True):
        """Try to replace one codeword with another that covers all critical points.
        Returns True if a swap was made."""
        # Get list of codewords that are critical (have at least one point with count==1)
        critical_cws = []
        for c in code:
            if np.min(counts[neighbors_list[c]]) <= 1:
                critical_cws.append(c)

        if not critical_cws:
            return False

        if randomize:
            random.shuffle(critical_cws)

        for c in critical_cws:
            if time.time() > deadline - 10:
                return False

            crit_points = [i for i in neighbors_list[c] if counts[i] == 1]
            if not crit_points:
                continue

            # Find candidates: codewords not in code that cover all critical points
            first_crit = crit_points[0]
            candidates = [x for x in neighbor_sets[first_crit] if x not in code_set]

            best_w = -1
            best_score = -1

            for w in candidates:
                valid = True
                for k in crit_points[1:]:
                    if k not in neighbor_sets[w]:
                        valid = False
                        break
                if not valid:
                    continue

                # Score: number of points that go from count 1 to count 2 (creates redundancy)
                score = int(np.sum(counts[neighbors_list[w]] == 1))
                # Also prefer codewords that cover more total points with count >= 1
                if score > best_score:
                    best_score = score
                    best_w = w

            if best_w != -1:
                code.remove(c)
                code_set.discard(c)
                counts[neighbors_list[c]] -= 1

                code.append(best_w)
                code_set.add(best_w)
                counts[neighbors_list[best_w]] += 1

                # After swap, try to remove redundants
                try_remove_redundant(code, code_set, counts)
                return True

        return False

    def try_2_swap(code, code_set, counts):
        """Try to replace two codewords with two different codewords.
        Returns True if a swap was made."""
        # Find pairs of critical codewords
        critical_cws = []
        for c in code:
            if np.min(counts[neighbors_list[c]]) <= 1:
                critical_cws.append(c)

        if len(critical_cws) < 2:
            return False

        # Try limited number of pairs
        random.shuffle(critical_cws)
        pairs_tried = 0
        max_pairs = 50

        for i in range(len(critical_cws)):
            if pairs_tried >= max_pairs or time.time() > deadline - 10:
                break
            for j in range(i + 1, len(critical_cws)):
                if pairs_tried >= max_pairs or time.time() > deadline - 10:
                    break
                pairs_tried += 1

                c1, c2 = critical_cws[i], critical_cws[j]

                # Temporarily remove both
                temp_counts = counts.copy()
                temp_counts[neighbors_list[c1]] -= 1
                temp_counts[neighbors_list[c2]] -= 1

                # Points that would be uncovered
                uncovered_by_removal = [k for k in range(N) if temp_counts[k] == 0]
                if not uncovered_by_removal:
                    # Both were redundant, just remove them
                    code.remove(c1)
                    code_set.discard(c1)
                    code.remove(c2)
                    code_set.discard(c2)
                    counts = temp_counts
                    try_remove_redundant(code, code_set, counts)
                    return True

                # Need to find two codewords that cover all uncovered_by_removal
                # and are not in code (excluding c1, c2 which we're removing)
                first_pt = uncovered_by_removal[0]
                cand1 = [x for x in neighbor_sets[first_pt] if x not in code_set or x in (c1, c2)]

                found = False
                for w1 in cand1[:30]:  # Limit candidates for speed
                    if time.time() > deadline - 10:
                        break
                    # Check w1 covers some of the uncovered points
                    covers_w1 = set(neighbors_list[w1]) & set(uncovered_by_removal)
                    remaining = [k for k in uncovered_by_removal if k not in covers_w1]

                    if not remaining:
                        # w1 alone covers everything
                        code.remove(c1)
                        code_set.discard(c1)
                        code.remove(c2)
                        code_set.discard(c2)
                        counts = temp_counts
                        code.append(w1)
                        code_set.add(w1)
                        counts[neighbors_list[w1]] += 1
                        try_remove_redundant(code, code_set, counts)
                        found = True
                        break

                    # Find w2 that covers remaining
                    first_rem = remaining[0]
                    cand2 = [x for x in neighbor_sets[first_rem] if x not in code_set or x in (c1, c2)]
                    cand2 = [x for x in cand2 if x != w1]

                    for w2 in cand2[:20]:
                        valid = True
                        for k in remaining:
                            if k not in neighbor_sets[w2]:
                                valid = False
                                break
                        if valid:
                            code.remove(c1)
                            code_set.discard(c1)
                            code.remove(c2)
                            code_set.discard(c2)
                            counts = temp_counts
                            code.append(w1)
                            code_set.add(w1)
                            counts[neighbors_list[w1]] += 1
                            code.append(w2)
                            code_set.add(w2)
                            counts[neighbors_list[w2]] += 1
                            try_remove_redundant(code, code_set, counts)
                            found = True
                            break
                    if found:
                        break

                if found:
                    return True

        return False

    def local_search(code, max_time=30):
        """Run local search until no more improvements or time limit."""
        code_set = set(code)
        counts = compute_counts(code)
        ls_start = time.time()

        improved = True
        while improved and (time.time() - ls_start) < max_time and time.time() < deadline - 15:
            improved = False
            improved |= try_remove_redundant(code, code_set, counts)
            improved |= try_1_swap(code, code_set, counts, randomize=True)
            if time.time() < deadline - 15:
                improved |= try_2_swap(code, code_set, counts)

        return code

    # ---------- Phase 3: Multi-restart with perturbation ----------
    # Initial greedy from scratch
    initial_code = greedy_cover()
    best_code = local_search(initial_code)
    best_size = len(best_code)

    # Multiple restarts with perturbation
    random.seed(42)
    restarts = 0
    max_restarts = 15

    while time.time() < deadline - 20 and restarts < max_restarts:
        restarts += 1

        # Perturb: remove a random subset and re-greedy
        perturbation_size = max(1, len(best_code) // 8)
        perturbed = list(best_code)
        random.shuffle(perturbed)
        removed = perturbed[:perturbation_size]
        perturbed = perturbed[perturbation_size:]

        new_code = greedy_cover(seed_code=perturbed)
        if len(new_code) >= best_size:
            continue

        new_code = local_search(new_code, max_time=20)
        if len(new_code) < best_size:
            best_code = new_code
            best_size = len(new_code)

    # Final polish pass
    if time.time() < deadline - 5:
        final_code = local_search(best_code, max_time=10)
        if len(final_code) < best_size:
            best_code = final_code
            best_size = len(final_code)

    # Convert indices to word lists
    result = [idx_to_tuple(c) for c in best_code]

    # Ensure distinctness
    seen = set()
    unique_result = []
    for w in result:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            unique_result.append(w)

    return unique_result
