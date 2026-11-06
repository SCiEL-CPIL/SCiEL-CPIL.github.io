"""Ternary covering code K_3(7,1) - optimized greedy + local search with soft-surrogate swaps."""
import time
import random
import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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
    degree = n * (q - 1)  # 15

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

    # Precompute neighbor lists
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

    # Precompute covering_codewords: for each word, which codewords cover it
    covering_codewords = [[] for _ in range(N)]
    for c in range(N):
        for w in neighbors_list[c]:
            covering_codewords[w].append(c)

    # ---------- Phase 1: Optimized Greedy Set-Cover ----------
    # score[c] = number of currently uncovered points that codeword c would cover
    score = np.full(N, degree, dtype=np.int32)
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()

    while np.any(uncovered):
        if time.time() > deadline - 50:
            break

        max_gain = int(np.max(score))
        if max_gain == 0:
            break

        # Find all candidates with max gain, pick one with spectral bias if available
        candidates = np.where(score == max_gain)[0]
        if len(candidates) > 1:
            # Use a simple hash-based tiebreaker for diversity
            best_idx = int(candidates[random.randint(0, len(candidates) - 1)])
        else:
            best_idx = int(candidates[0])

        code.append(best_idx)
        code_set.add(best_idx)

        # Update: for each word covered by best_idx that was uncovered,
        # decrement score for all codewords that cover that word
        for w in neighbors_list[best_idx]:
            if uncovered[w]:
                uncovered[w] = False
                for c in covering_codewords[w]:
                    score[c] -= 1

    # Ensure full coverage
    if np.any(uncovered):
        for w in np.where(uncovered)[0]:
            if w not in code_set:
                code.append(int(w))
                code_set.add(int(w))

    # ---------- Build coverage counts ----------
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_list[idx]] += 1

    best_size = len(code)
    best_code = list(code)

    # ---------- Phase 2: Local Search ----------
    def try_remove():
        """Remove all redundant codewords. Returns True if any removed."""
        removed = False
        i = 0
        while i < len(code):
            c = code[i]
            if np.min(counts[neighbors_list[c]]) > 1:
                code.pop(i)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
                removed = True
            else:
                i += 1
        return removed

    def greedy_repair(uncovered_set):
        """Given a set of uncovered points, greedily add codewords to cover them."""
        if not uncovered_set:
            return []
        added = []
        temp_uncovered = set(uncovered_set)
        temp_mask = np.zeros(N, dtype=bool)
        for w in temp_uncovered:
            temp_mask[w] = True

        # Local score array
        local_score = np.zeros(N, dtype=np.int32)
        for w in temp_uncovered:
            for c in covering_codewords[w]:
                if c not in code_set:
                    local_score[c] += 1

        while np.any(temp_mask):
            if time.time() > deadline - 10:
                break
            max_s = int(np.max(local_score))
            if max_s == 0:
                break
            cands = np.where(local_score == max_s)[0]
            best_c = int(cands[random.randint(0, len(cands) - 1)])

            added.append(best_c)
            for w in neighbors_list[best_c]:
                if temp_mask[w]:
                    temp_mask[w] = False
                    temp_uncovered.discard(w)
                    for c in covering_codewords[w]:
                        local_score[c] -= 1

        return added

    def try_swap_soft(c):
        """
        Try to swap codeword c with a better-positioned codeword using a soft-surrogate score.
        Instead of requiring ALL critical points to be covered, we score candidates by
        (fraction of critical points covered) with a smooth penalty, allowing near-misses.
        """
        # Find critical points (uniquely covered by c)
        crit_mask = (counts == 1)
        crit_points = [i for i in neighbors_list[c] if crit_mask[i]]

        if not crit_points:
            # c is redundant, just remove it
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1
            return True

        # Get candidate pool: neighbors of critical points not in code
        candidate_pool = set()
        for p in crit_points:
            candidate_pool.update(neighbor_sets[p])
        candidate_pool -= code_set

        if not candidate_pool:
            return False

        # Soft-surrogate scoring:
        # For each candidate w, compute:
        #   - covers_crit: how many critical points w covers
        #   - new_unique: how many points w would uniquely cover (counts[w] == 1)
        #   - penalty: for each critical point NOT covered by w, we'd need extra coverage
        # Score = covers_crit * 1.0 - (len(crit_points) - covers_crit) * 0.5 + new_unique * 0.1
        # This is a smooth surrogate that prefers full coverage but tolerates partial.

        best_w = -1
        best_score = -1e9
        crit_set = set(crit_points)

        for w in candidate_pool:
            # How many critical points does w cover?
            covers = 0
            for p in crit_points:
                if p in neighbor_sets[w]:
                    covers += 1

            if covers < len(crit_points):
                # Check if the uncovered critical points can be covered by other existing codewords
                # If not, this swap would leave them uncovered -> invalid
                # Quick check: for each uncovered crit point, is it covered by any other code word?
                valid = True
                for p in crit_points:
                    if p not in neighbor_sets[w]:
                        # p must be covered by some other codeword
                        # counts[p] == 1 means only c covers it, so if w doesn't cover p,
                        # removing c leaves p uncovered. Invalid.
                        valid = False
                        break
                if not valid:
                    continue

            # Compute new_unique: points that would have count 1 after adding w (excluding c's effect)
            # counts[w] currently includes c's contribution if c covers w
            # After removing c and adding w:
            # For points in neighbors[w]: new_count = counts[p] - (1 if p in neighbors[c]) + 1
            new_unique = 0
            for p in neighbors_list[w]:
                old_count = counts[p]
                c_contribution = 1 if p in neighbor_sets[c] else 0
                new_count = old_count - c_contribution + 1
                if new_count == 1:
                    new_unique += 1

            # Soft score
            score_val = covers * 1.0 + new_unique * 0.1 - (len(crit_points) - covers) * 0.5

            if score_val > best_score:
                best_score = score_val
                best_w = w

        if best_w == -1:
            return False

        # Execute swap
        code.remove(c)
        code_set.remove(c)
        counts[neighbors_list[c]] -= 1

        code.append(best_w)
        code_set.add(best_w)
        counts[neighbors_list[best_w]] += 1

        return True

    def kick(k):
        """Remove k random codewords and try to repair. Accept if size decreases."""
        if len(code) < k:
            return False

        old_size = len(code)
        indices = random.sample(range(len(code)), k)
        to_remove = [code[i] for i in indices]

        temp_coverage = counts.copy()
        for c in to_remove:
            temp_coverage[neighbors_list[c]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) == 0:
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
            return True

        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)

        new_size = old_size - k + len(added)

        if new_size < old_size:
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
            for c in added:
                code.append(c)
                code_set.add(c)
                counts[neighbors_list[c]] += 1
            return True
        else:
            return False

    # ---------- Main Local Search Loop ----------
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False

        # 1. Exhaustive removal pass
        while try_remove():
            improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 20:
            break

        # 2. Swap pass with soft surrogate
        shuffled_code = list(code)
        random.shuffle(shuffled_code)

        for c in shuffled_code:
            if time.time() > deadline - 20:
                break
            if c not in code_set:
                continue
            if try_swap_soft(c):
                improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 20:
            break

        # 3. Kick passes
        for k in [1, 2, 3]:
            max_kicks = max(5, 40 // k)
            kicks_tried = 0
            while kicks_tried < max_kicks and time.time() < deadline - 20:
                if kick(k):
                    improved = True
                    kicks_tried = 0
                else:
                    kicks_tried += 1

            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

            if time.time() > deadline - 20:
                break

        # 4. Final polish
        if len(code) > 1 and time.time() < deadline - 20:
            shuffled_code2 = list(code)
            random.shuffle(shuffled_code2)
            for c in shuffled_code2:
                if time.time() > deadline - 20:
                    break
                if c not in code_set:
                    continue
                if try_swap_soft(c):
                    improved = True

            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

    # Restore best code found
    if len(best_code) < len(code):
        code = best_code
        code_set = set(best_code)
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors_list[idx]] += 1

    # Convert back to list of lists
    result = [idx_to_tuple(cw) for cw in code]

    return result
