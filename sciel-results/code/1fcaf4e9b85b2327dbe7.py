import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave margin for 120s limit

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
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

    # Precompute neighbor lists and balls
    neighbors_list = []
    neighbors_set = []
    balls = []
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
        balls.append(nb)

    # Precompute adjacency matrix B[c][w] = 1 if w in ball(c)
    B = np.zeros((N, N), dtype=np.int32)
    for c in range(N):
        B[c, balls[c]] = 1

    # Precompute cover_map: cover_map[w] = list of codewords whose ball contains w
    cover_map = [[] for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            cover_map[w].append(c)

    # ---------- Greedy Set-Cover Construction with Multiple Restarts ----------

    def greedy_construct(seed_val):
        """Run greedy set-cover construction with given random seed for tie-breaking."""
        rng = random.Random(seed_val)
        covered = np.zeros(N, dtype=bool)
        cnt = np.array([15] * N, dtype=np.int32)  # Each word is in 15 balls
        code = []
        uncovered_count = N

        while uncovered_count > 0 and time.time() < deadline:
            # Find candidates with max gain
            max_val = int(cnt.max())
            cands = np.flatnonzero(cnt == max_val)
            c = int(rng.choice(cands))
            code.append(c)

            ball_c = balls[c]
            newly = [w for w in ball_c if not covered[w]]
            if newly:
                covered[newly] = True
                uncovered_count -= len(newly)
                for w in newly:
                    cnt[cover_map[w]] -= 1

        return code

    # ---------- Pure Redundancy Removal ----------

    def refine(code):
        """Remove redundant codewords (those whose ball is fully covered by others)."""
        code_list = list(code)
        if not code_list:
            return code_list

        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            cover_count[balls[c]] += 1

        changed = True
        while changed:
            changed = False
            for i in range(len(code_list)):
                c = code_list[i]
                if np.all(cover_count[balls[c]] > 1):
                    cover_count[balls[c]] -= 1
                    code_list.pop(i)
                    changed = True
                    break

        return code_list

    # ---------- Local Search ----------

    def local_search(code, seed, time_limit):
        """
        Local search: try removing one codeword and relocating to cover any uncovered words.
        Returns improved code list.
        """
        rng = random.Random(seed)
        code_list = list(code)
        if not code_list:
            return code_list

        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            cover_count[balls[c]] += 1
        code_set = set(code_list)

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(code_list)))
            rng.shuffle(indices)

            for i in indices:
                if time.time() > time_limit:
                    break
                c_remove = code_list[i]
                ball_r = balls[c_remove]
                cover_count[ball_r] -= 1
                uncovered = [w for w in ball_r if cover_count[w] == 0]

                if not uncovered:
                    # Pure removal
                    code_list.pop(i)
                    code_set.discard(c_remove)
                    improved = True
                    break

                # Relocation: find a codeword whose ball covers all uncovered
                found = -1
                # Only consider candidates adjacent to uncovered words
                candidate_pool = set()
                for w in uncovered:
                    candidate_pool.update(neighbors_set[w])

                for c_add in candidate_pool:
                    if c_add == c_remove or c_add in code_set:
                        continue
                    # Check if c_add covers all uncovered
                    ok = True
                    for w in uncovered:
                        if not B[c_add, w]:
                            ok = False
                            break
                    if ok:
                        found = c_add
                        break

                if found >= 0:
                    # Swap: relocate c_remove -> found
                    cover_count[balls[found]] += 1
                    code_list[i] = found
                    code_set.discard(c_remove)
                    code_set.add(found)
                    improved = True
                    break
                else:
                    # Rollback
                    cover_count[ball_r] += 1

        return code_list

    # ---------- Kick-based Local Search (from current program) ----------

    def is_redundant(c, coverage):
        """Check if removing c leaves all its neighbors covered by others."""
        return np.min(coverage[neighbors_list[c]]) > 1

    def greedy_repair(uncovered_set, code_set, coverage):
        """
        Given a set of uncovered word indices, find codewords to add to cover them.
        Returns list of added codewords.
        """
        if not uncovered_set:
            return []

        added = []
        temp_uncovered = set(uncovered_set)

        while temp_uncovered and time.time() < deadline:
            # Candidates are neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors_set[w])

            best_c = -1
            best_count = 0
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

    def try_kick(k, code, code_set, coverage):
        """
        Try to remove k random codewords and repair.
        Returns (new_code, new_code_set, new_coverage, success) if successful.
        """
        if len(code) < k:
            return None

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
            new_code = [c for i, c in enumerate(code) if i not in indices]
            new_code_set = code_set - set(removed)
            return (new_code, new_code_set, temp_coverage, True)

        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set, code_set, temp_coverage)

        new_size = len(code) - len(removed) + len(added)

        if new_size < len(code):
            # Apply change
            new_code = [c for i, c in enumerate(code) if i not in indices] + added
            new_code_set = (code_set - set(removed)) | set(added)
            final_coverage = temp_coverage.copy()
            for c in added:
                final_coverage[neighbors_list[c]] += 1
            return (new_code, new_code_set, final_coverage, True)
        else:
            return None

    def try_swap(code, code_set, coverage):
        """
        Try to swap a random codeword with a different one.
        Returns (new_code, new_code_set, new_coverage, success) if successful.
        """
        if len(code) < 2:
            return None

        # Pick a random codeword to remove
        idx = random.randint(0, len(code) - 1)
        removed = code[idx]

        # Temporarily remove to check coverage
        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[removed]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) == 0:
            # Perfect removal, no need to add
            new_code = [c for i, c in enumerate(code) if i != idx]
            new_code_set = code_set - {removed}
            return (new_code, new_code_set, temp_coverage, True)

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
            return None

        # Apply the swap if it covers all uncovered
        if best_gain >= len(uncovered_set):
            new_code = code[:]
            new_code[idx] = best_c
            new_code_set = (code_set - {removed}) | {best_c}
            final_coverage = temp_coverage.copy()
            final_coverage[neighbors_list[best_c]] += 1
            return (new_code, new_code_set, final_coverage, True)
        else:
            return None

    # ---------- Main Construction ----------

    # Try multiple greedy restarts with different seeds
    best_code = None
    best_size = float('inf')

    max_restarts = 8
    for restart in range(max_restarts):
        if time.time() > deadline - 40:  # Need time for local search
            break
        seed = 42 + restart * 137
        cod = greedy_construct(seed)
        refined = refine(cod)
        if len(refined) < best_size:
            best_size = len(refined)
            best_code = refined
        # Early stop if we've found a very good solution
        if best_size <= 165:
            break

    if best_code is None:
        # Fallback: just return the first greedy result
        best_code = refine(greedy_construct(42))
        best_size = len(best_code)

    code = best_code
    code_set = set(code)
    coverage = np.zeros(N, dtype=np.int32)
    for c in code:
        coverage[neighbors_list[c]] += 1

    # ---------- Local Search Phase ----------

    # First pass: exhaustive removal of redundant codewords
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for c in list(code):
            if is_redundant(c, coverage):
                code.remove(c)
                code_set.remove(c)
                coverage[neighbors_list[c]] -= 1
                changed = True

    # Second pass: kick-based local search
    no_improvement_count = 0
    max_no_improvement = 25
    random.seed(42)

    while time.time() < deadline - 3:  # Safety margin
        # Try kicks with increasing k
        kick_success = False
        for k in [1, 2, 3]:
            if time.time() > deadline - 3:
                break
            result = try_kick(k, code, code_set, coverage)
            if result:
                new_code, new_code_set, new_coverage, _ = result
                code = new_code
                code_set = new_code_set
                coverage = new_coverage
                kick_success = True
                break

        # Try swap
        if not kick_success and time.time() < deadline - 3:
            result = try_swap(code, code_set, coverage)
            if result:
                new_code, new_code_set, new_coverage, _ = result
                code = new_code
                code_set = new_code_set
                coverage = new_coverage
                kick_success = True

        if kick_success:
            no_improvement_count = 0
            # After improvement, do another exhaustive removal pass
            changed = True
            while changed and time.time() < deadline - 3:
                changed = False
                for c in list(code):
                    if is_redundant(c, coverage):
                        code.remove(c)
                        code_set.remove(c)
                        coverage[neighbors_list[c]] -= 1
                        changed = True
        else:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturb: try a larger kick to escape local minimum
                if len(code) > 5:
                    k = random.randint(4, min(6, len(code)))
                    result = try_kick(k, code, code_set, coverage)
                    if result:
                        new_code, new_code_set, new_coverage, _ = result
                        code = new_code
                        code_set = new_code_set
                        coverage = new_coverage
                        no_improvement_count = 0
                else:
                    break

    # Final refinement pass
    final_code = refine(code)

    result = [idx_to_tuple(c) for c in final_code]
    return result
