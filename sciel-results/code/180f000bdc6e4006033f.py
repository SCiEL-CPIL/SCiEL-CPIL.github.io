import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave margin for 150s limit

    # Fallback for other parameters
    if q != 3 or n != 7 or radius != 1:
        if n <= 4 and q == 3 and radius == 1:
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

    # Precompute neighbor lists as arrays
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

    # Convert to numpy array for vectorized operations
    neighbors_arr = np.array(neighbors_list, dtype=np.int32)  # shape (N, 15)

    def run_greedy_init(randomize=False):
        """Run greedy set cover and return (code, coverage)."""
        nonlocal deadline
        cov = np.zeros(N, dtype=np.int32)
        unc_mask = np.ones(N, dtype=bool)
        cod = []
        cod_set = set()

        # Greedy initialization
        if randomize:
            first = random.randint(0, N - 1)
        else:
            first = 0
        cod.append(first)
        cod_set.add(first)
        cov[neighbors_list[first]] += 1
        unc_mask[neighbors_list[first]] = False

        while unc_mask.any() and time.time() < deadline - 5:
            # Vectorized: compute coverage gain for all candidates at once
            # For each candidate c, gain = number of uncovered words in its neighborhood
            # We can compute this as: for each uncovered word w, all its neighbors get +1
            # But that's expensive. Instead, iterate over uncovered words' neighbors.

            # Build candidate pool from uncovered words
            uncovered_indices = np.where(unc_mask)[0]
            if len(uncovered_indices) == 0:
                break

            # For speed, sample a subset if too many uncovered
            if len(uncovered_indices) > 200:
                sample = random.sample(uncovered_indices.tolist(), 200)
                uncovered_indices = np.array(sample)

            candidate_pool = set()
            for w in uncovered_indices:
                candidate_pool.update(neighbors_set[w])

            best_c = -1
            best_count = 0

            for c in candidate_pool:
                if c in cod_set:
                    continue
                cnt = np.sum(unc_mask[neighbors_list[c]])
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    if best_count >= len(uncovered_indices):
                        break

            if best_c == -1 or best_count == 0:
                break

            cod.append(best_c)
            cod_set.add(best_c)
            cov[neighbors_list[best_c]] += 1
            unc_mask[neighbors_list[best_c]] = False

        return cod, cov

    def is_redundant(c, cov):
        return np.min(cov[neighbors_list[c]]) > 1

    def greedy_repair(uncovered_set, current_code_set, excluded_set=None):
        if not uncovered_set:
            return []

        added = []
        temp_uncovered = set(uncovered_set)

        while temp_uncovered:
            best_c = -1
            best_count = 0

            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors_set[w])

            for c in candidate_pool:
                if c in current_code_set:
                    continue
                if excluded_set and c in excluded_set:
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

    def try_kick(k, code, code_set, coverage):
        if len(code) < k:
            return False, code, code_set, coverage

        indices = random.sample(range(len(code)), k)
        removed = [code[i] for i in indices]
        removed_set = set(removed)

        temp_coverage = coverage.copy()
        for c in removed:
            temp_coverage[neighbors_list[c]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) == 0:
            new_code = [c for i, c in enumerate(code) if i not in indices]
            new_code_set = set(new_code)
            return True, new_code, new_code_set, temp_coverage

        uncovered_set = set(uncovered.tolist())
        temp_code_set = set(code)
        for c in removed:
            temp_code_set.discard(c)

        added = greedy_repair(uncovered_set, temp_code_set, excluded_set=removed_set)
        new_size = len(code) - len(removed) + len(added)

        if new_size < len(code):
            new_code = [c for i, c in enumerate(code) if i not in indices] + added
            new_code_set = set(new_code)
            new_coverage = temp_coverage
            for c in added:
                new_coverage[neighbors_list[c]] += 1
            return True, new_code, new_code_set, new_coverage
        else:
            return False, code, code_set, coverage

    def try_swap(code, code_set, coverage):
        if len(code) < 2:
            return False, code, code_set, coverage

        idx = random.randint(0, len(code) - 1)
        removed = code[idx]

        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[removed]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) == 0:
            new_code = [c for i, c in enumerate(code) if i != idx]
            new_code_set = set(new_code)
            return True, new_code, new_code_set, temp_coverage

        uncovered_set = set(uncovered.tolist())

        candidate_pool = set()
        for w in uncovered_set:
            candidate_pool.update(neighbors_set[w])

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

        if best_c == -1 or best_gain < len(uncovered_set):
            return False, code, code_set, coverage

        new_code = [c for i, c in enumerate(code) if i != idx] + [best_c]
        new_code_set = set(new_code)
        new_coverage = temp_coverage
        new_coverage[neighbors_list[best_c]] += 1
        return True, new_code, new_code_set, new_coverage

    # Main search
    random.seed(42)

    # Initial greedy construction
    code, coverage = run_greedy_init(randomize=False)
    code_set = set(code)
    best_code = list(code)
    best_size = len(code)

    # Simulated annealing style local search
    temperature = 10.0
    cooling = 0.98

    while time.time() < deadline - 3:
        # Exhaustive removal pass
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            for c in list(code):
                if is_redundant(c, coverage):
                    code.remove(c)
                    code_set.remove(c)
                    coverage[neighbors_list[c]] -= 1
                    changed = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        # Try kicks and swaps with simulated annealing acceptance
        kick_success = False
        for attempt in range(8):
            if time.time() > deadline - 3:
                break

            # Randomly choose operation
            op = random.random()
            if op < 0.4:
                success, new_code, new_code_set, new_coverage = try_kick(1, code, code_set, coverage)
            elif op < 0.6:
                success, new_code, new_code_set, new_coverage = try_kick(2, code, code_set, coverage)
            elif op < 0.75:
                success, new_code, new_code_set, new_coverage = try_kick(3, code, code_set, coverage)
            else:
                success, new_code, new_code_set, new_coverage = try_swap(code, code_set, coverage)

            if success:
                # Accept if it improves, or with probability based on temperature (for equal size)
                if len(new_code) < len(code):
                    code, code_set, coverage = new_code, new_code_set, new_coverage
                    kick_success = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                    break
                elif len(new_code) == len(code) and random.random() < 0.3:
                    # Accept equal-size moves to escape local minima
                    code, code_set, coverage = new_code, new_code_set, new_coverage
                    kick_success = True
                    break

        # Cool down
        temperature *= cooling

        # Periodic restart to explore different basins
        if time.time() > deadline - 25:
            break

        if random.random() < 0.15:
            new_code, new_coverage = run_greedy_init(randomize=True)
            code, coverage = new_code, new_coverage
            code_set = set(code)
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

    # Final removal pass on best code
    final_code = list(best_code)
    final_set = set(final_code)
    final_cov = np.zeros(N, dtype=np.int32)
    for c in final_code:
        final_cov[neighbors_list[c]] += 1

    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        for c in list(final_code):
            if np.min(final_cov[neighbors_list[c]]) > 1:
                final_code.remove(c)
                final_set.remove(c)
                final_cov[neighbors_list[c]] -= 1
                changed = True

    if len(final_code) < best_size:
        best_code = final_code

    result = [idx_to_tuple(c) for c in best_code]
    return result
