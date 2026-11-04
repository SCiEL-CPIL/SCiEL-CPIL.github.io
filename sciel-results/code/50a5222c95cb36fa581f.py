"""Combined ternary covering code K_3(7,1) via greedy + local search with coverage tracking."""
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

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

    # Precompute all words as digit arrays
    all_words = np.zeros((N, n), dtype=np.int8)
    for idx in range(N):
        temp = idx
        for pos in range(n - 1, -1, -1):
            all_words[idx, pos] = temp % 3
            temp //= 3

    # Precompute neighbor lists (words within distance <= 1)
    neighbors_list = []
    for i in range(N):
        w = all_words[i]
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    diff = (new_val - val) * (3 ** (n - 1 - pos))
                    nb.append(i + diff)
        neighbors_list.append(nb)

    # Also store as numpy arrays for fast indexing
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors_list]

    # Precompute neighbor sets for fast intersection
    neighbor_sets = [set(nb) for nb in neighbors_list]

    # ---------- Greedy Set-Cover Construction ----------
    coverage = np.zeros(N, dtype=np.int32)
    uncovered_mask = np.ones(N, dtype=bool)
    code = []

    while uncovered_mask.any() and time.time() < deadline - 30:
        # Find candidates: neighbors of uncovered words
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break

        candidate_set = set()
        for idx in uncovered_indices:
            candidate_set.update(neighbor_sets[idx])

        if not candidate_set:
            break

        best_c = -1
        best_count = 0

        for c in candidate_set:
            cnt = np.sum(uncovered_mask[neighbors_np[c]])
            if cnt > best_count:
                best_count = cnt
                best_c = c
            if best_count == len(uncovered_indices):
                break

        if best_c == -1 or best_count == 0:
            break

        code.append(best_c)
        coverage[neighbors_np[best_c]] += 1
        uncovered_mask[neighbors_np[best_c]] = False

    # ---------- Local Search ----------

    def is_redundant(c):
        """Check if removing codeword c still leaves all its neighbors covered."""
        return np.min(coverage[neighbors_np[c]]) > 1

    def try_remove():
        """Try to remove redundant codewords."""
        nonlocal code, coverage
        removed = False
        for i in range(len(code)):
            c = code[i]
            if is_redundant(c):
                code.pop(i)
                coverage[neighbors_np[c]] -= 1
                removed = True
                break
        return removed

    def greedy_repair(uncovered_set):
        """Greedily add codewords to cover the given uncovered set."""
        if not uncovered_set:
            return []
        added = []
        temp_uncovered = set(uncovered_set)

        while temp_uncovered and time.time() < deadline - 10:
            # Build candidate pool from neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbor_sets[w])

            best_c = -1
            best_count = 0

            for c in candidate_pool:
                cnt = sum(1 for x in neighbors_np[c] if x in temp_uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                if best_count == len(temp_uncovered):
                    break

            if best_c == -1 or best_count == 0:
                break

            added.append(best_c)
            for w in neighbors_np[best_c]:
                temp_uncovered.discard(w)

        return added

    def exchange_move():
        """Try to remove one codeword and add another to improve efficiency."""
        nonlocal code, coverage
        if len(code) < 2:
            return False

        old_size = len(code)

        # Pick a random codeword to try removing
        c_remove = random.choice(code)

        temp_coverage = coverage.copy()
        temp_coverage[neighbors_np[c_remove]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]
        uncovered_set = set(uncovered.tolist())

        if not uncovered_set:
            # Removal is free
            code.remove(c_remove)
            coverage[neighbors_np[c_remove]] -= 1
            return True

        # Try to find a single replacement that covers all uncovered
        candidate_pool = set()
        for w in uncovered_set:
            candidate_pool.update(neighbor_sets[w])

        best_c = -1
        best_count = 0
        for c in candidate_pool:
            if c in code:
                continue
            cnt = sum(1 for x in neighbors_np[c] if x in uncovered_set)
            if cnt > best_count:
                best_count = cnt
                best_c = c
            if best_count == len(uncovered_set):
                break

        if best_c != -1 and best_count == len(uncovered_set):
            # Perfect replacement
            code.remove(c_remove)
            code.append(best_c)
            coverage[neighbors_np[c_remove]] -= 1
            coverage[neighbors_np[best_c]] += 1
            return True

        # Try greedy repair with multiple additions
        added = greedy_repair(uncovered_set)
        new_size = old_size - 1 + len(added)

        if new_size < old_size:
            code.remove(c_remove)
            coverage[neighbors_np[c_remove]] -= 1
            for c in added:
                code.append(c)
                coverage[neighbors_np[c]] += 1
            return True

        return False

    def kick():
        """Randomly remove a codeword and repair."""
        nonlocal code, coverage
        if not code:
            return False

        old_size = len(code)

        idx_to_remove = random.randint(0, len(code) - 1)
        c_remove = code[idx_to_remove]

        temp_coverage = coverage.copy()
        temp_coverage[neighbors_np[c_remove]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) == 0:
            code.remove(c_remove)
            coverage[neighbors_np[c_remove]] -= 1
            return True

        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)

        new_size = old_size - 1 + len(added)

        if new_size < old_size:
            code.remove(c_remove)
            coverage[neighbors_np[c_remove]] -= 1
            for c in added:
                code.append(c)
                coverage[neighbors_np[c]] += 1
            return True

        return False

    # Main local search loop
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False

        # Phase 1: Remove redundant codewords
        while try_remove():
            improved = True

        # Phase 2: Try exchange moves
        for _ in range(50):
            if time.time() > deadline - 20:
                break
            if exchange_move():
                improved = True

        # Phase 3: Kick if stuck
        if not improved and time.time() < deadline - 15:
            if kick():
                improved = True

    # Final removal pass
    while try_remove():
        pass

    # Verify validity
    final_coverage = np.zeros(N, dtype=np.int32)
    for cw in code:
        final_coverage[neighbors_np[cw]] += 1

    if not (final_coverage > 0).all():
        # Fallback to tetracode construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert back to list of lists
    result = [list(all_words[cw]) for cw in code]

    # Ensure distinctness
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)

    return unique_result
