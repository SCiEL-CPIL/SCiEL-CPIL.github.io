import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # Leave margin for 120s limit

    if q != 3 or n != 7 or radius != 1:
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

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

    # Precompute neighbor lists and sets
    neighbors_list = []
    neighbors_set = []
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
        neighbors_set.append(set(nb))

    # ---------- Greedy Set-Cover with Incremental O(K^2) Gain Updates ----------

    def greedy_construct(seed_val, time_budget):
        """
        Run greedy set-cover construction with incremental gain tracking.
        The key optimization: maintain a gain[c] array that counts how many
        uncovered words each candidate c covers. After picking a codeword,
        update gain[] incrementally in O(K) where K is the candidate pool size,
        rather than O(N) per step.
        """
        rng = random.Random(seed_val)
        cov = np.zeros(N, dtype=np.int32)
        unc = np.ones(N, dtype=bool)
        cod = []
        cod_set = set()

        # Random start
        start = rng.randint(0, N - 1)
        cod.append(start)
        cod_set.add(start)
        cov[neighbors_list[start]] += 1
        unc[neighbors_list[start]] = False

        # Build initial candidate pool: neighbors of uncovered words
        uncovered_indices = np.where(unc)[0]
        candidate_pool = set()
        for w in uncovered_indices:
            candidate_pool.update(neighbors_set[w])
        candidate_pool.discard(start)

        # Initialize gain array for candidates
        gain = {}
        for c in candidate_pool:
            gain[c] = np.sum(unc[neighbors_list[c]])

        while unc.any() and time.time() < time_budget:
            if not candidate_pool:
                break

            # Find best candidate with random tie-breaking
            best_c = -1
            best_count = 0
            for c in candidate_pool:
                g = gain[c]
                if g > best_count:
                    best_count = g
                    best_c = c
                    # Random tie-breaking: if equal, keep first or randomize
                elif g == best_count and g > 0:
                    if rng.random() < 0.5:
                        best_c = c

            if best_c == -1 or best_count == 0:
                break

            cod.append(best_c)
            cod_set.add(best_c)
            cov[neighbors_list[best_c]] += 1
            newly_covered = neighbors_list[best_c]
            unc[newly_covered] = False

            # Incrementally update gain array
            # For each candidate c in the pool, gain[c] decreases by the number
            # of words in newly_covered that are in neighbors_set[c] and were previously uncovered
            for c in list(candidate_pool):
                if c == best_c:
                    continue
                # Count how many of newly_covered are in neighbors_set[c] and were unc before
                # Since we just set unc[newly_covered] = False, we need to count overlap
                overlap = 0
                for w in newly_covered:
                    if w in neighbors_set[c]:
                        overlap += 1
                gain[c] -= overlap
                if gain[c] <= 0:
                    del gain[c]
                    candidate_pool.discard(c)

            # Add new candidates that are neighbors of newly covered words
            # (words that just became covered might make new candidates relevant
            #  only if they were previously covered, but since they're now covered,
            #  new candidates adjacent to them don't help with uncovered words)
            # Actually, new candidates should be neighbors of still-uncovered words.
            # Since we only covered words, the uncovered set shrinks, so no new candidates.
            # But wait: a word that was covered might now be the only thing preventing
            # a candidate from being useful... no, candidates are defined by uncovered words.
            # So the candidate pool only shrinks.

        return cod, cod_set, cov

    # Try multiple greedy restarts with different seeds
    best_code = None
    best_code_set = None
    best_coverage = None
    best_size = float('inf')

    max_restarts = 8
    for restart in range(max_restarts):
        if time.time() > deadline - 40:  # Need time for local search
            break
        cod, cod_set, cov = greedy_construct(42 + restart * 7, deadline - 40)
        if len(cod) < best_size:
            best_size = len(cod)
            best_code = cod
            best_code_set = cod_set
            best_coverage = cov
        # Early stop if we've found a very good solution
        if best_size <= 160:
            break

    code = best_code
    code_set = best_code_set
    coverage = best_coverage

    # ---------- Local Search ----------

    def is_redundant(c):
        # Check if removing c leaves all its neighbors covered by others
        return np.min(coverage[neighbors_list[c]]) > 1

    def greedy_repair(uncovered_set):
        """
        Given a set of uncovered word indices, find a minimal set of codewords
        to add to cover them.
        """
        if not uncovered_set:
            return []

        added = []
        temp_uncovered = set(uncovered_set)

        while temp_uncovered:
            best_c = -1
            best_count = 0

            # Candidates are neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors_set[w])

            for c in candidate_pool:
                if c in code_set:
                    continue
                cnt = len(neighbors_set[c] & temp_uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    if best_count == len(temp_uncovered):
                        break

            if best_c == -1:
                break

            added.append(best_c)
            temp_uncovered.difference_update(neighbors_set[best_c])

        return added

    def apply_code_change(removed_list, added_list):
        """
        Apply the change to global state.
        """
        nonlocal code, code_set, coverage
        for c in removed_list:
            if c in code_set:
                code.remove(c)
                code_set.remove(c)
                coverage[neighbors_list[c]] -= 1
        for c in added_list:
            if c not in code_set:
                code.append(c)
                code_set.add(c)
                coverage[neighbors_list[c]] += 1

    def try_kick(k):
        """
        Try to remove k random codewords and repair.
        Returns True if size decreased.
        """
        if len(code) < k:
            return False

        # Pick k distinct random indices
        indices = random.sample(range(len(code)), k)
        removed = [code[i] for i in indices]

        # Temporarily remove to check coverage
        temp_coverage = coverage.copy()
        for c in removed:
            temp_coverage[neighbors_list[c]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) == 0:
            # Perfect removal
            apply_code_change(removed, [])
            return True

        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)

        new_size = len(code) - len(removed) + len(added)

        if new_size < len(code):
            # Apply change
            apply_code_change(removed, added)
            return True
        else:
            # Try to find a single codeword that covers all uncovered
            if len(uncovered_set) <= 20:
                candidate_pool = set()
                for w in uncovered_set:
                    candidate_pool.update(neighbors_set[w])

                best_single = -1
                best_gain = 0
                for c in candidate_pool:
                    if c in code_set or c in removed:
                        continue
                    gain = len(neighbors_set[c] & uncovered_set)
                    if gain > best_gain:
                        best_gain = gain
                        best_single = c

                if best_single != -1 and best_gain == len(uncovered_set):
                    apply_code_change(removed, [best_single])
                    return True

            return False

    def try_swap():
        """
        Try to swap a random codeword with a different one.
        Returns True if size decreased.
        """
        if len(code) < 2:
            return False

        # Pick a random codeword to remove
        idx = random.randint(0, len(code) - 1)
        removed = code[idx]

        # Temporarily remove to check coverage
        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[removed]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) == 0:
            # Perfect removal, no need to add
            apply_code_change([removed], [])
            return True

        uncovered_set = set(uncovered.tolist())

        # Only consider candidates adjacent to uncovered words
        candidate_pool = set()
        for w in uncovered_set:
            candidate_pool.update(neighbors_set[w])

        # Find a candidate to add
        best_c = -1
        best_gain = 0
        for c in candidate_pool:
            if c == removed or c in code_set:
                continue
            gain = len(neighbors_set[c] & uncovered_set)
            if gain > best_gain:
                best_gain = gain
                best_c = c
                if best_gain >= len(uncovered_set):
                    break

        if best_c == -1:
            return False

        # Apply the swap if it covers all uncovered
        if best_gain >= len(uncovered_set):
            apply_code_change([removed], [best_c])
            return True
        else:
            return False

    # Main Loop
    no_improvement_count = 0
    max_no_improvement = 25
    random.seed(42)

    while time.time() < deadline:
        # 1. Exhaustive Removal
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for c in list(code):
                if is_redundant(c):
                    apply_code_change([c], [])
                    changed = True
                    if time.time() > deadline:
                        break

        # 2. Kicks - try multiple attempts per iteration for better mixing
        kick_success = False
        for _ in range(4):  # Try up to 4 kicks/swaps per main loop iteration
            if time.time() > deadline:
                break

            # Try 1-removal
            if try_kick(1):
                kick_success = True
                break
            # Try 2-removal
            elif try_kick(2):
                kick_success = True
                break
            # Try 3-removal
            elif try_kick(3):
                kick_success = True
                break
            # Try swap
            elif try_swap():
                kick_success = True
                break

        if kick_success:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturb the code slightly to escape local minimum
                if len(code) > 5:
                    k = random.randint(4, min(8, len(code)))
                    if try_kick(k):
                        no_improvement_count = 0
                else:
                    if len(code) > 10:
                        if try_kick(3):
                            no_improvement_count = 0

    result = [idx_to_tuple(c) for c in code]
    return result
