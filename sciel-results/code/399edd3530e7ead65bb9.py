"""Ternary covering code via coverage-count multi-exchange local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin under 150s kill

    N = q ** n
    if N <= 0:
        return []

    # Precompute all words
    all_words = np.zeros((N, n), dtype=np.int32)
    for idx in range(N):
        rem = idx
        for j in range(n - 1, -1, -1):
            all_words[idx, j] = rem % q
            rem //= q

    # Precompute balls: for each codeword, list of words within `radius`
    # Use a compact representation: ball[c] = list of word indices within radius
    # Also build ball_size and a boolean mask for quick coverage checks
    ball_size = 1 + n * (q - 1) * radius  # exact for radius=1
    # For general radius, compute properly
    balls = []
    ball_masks = []  # boolean array per codeword (N x N total, but we store as list of arrays)
    for c in range(N):
        diff_count = np.sum(all_words != all_words[c], axis=1)
        mask = diff_count <= radius
        balls.append(np.where(mask)[0])
        ball_masks.append(mask)

    # Greedy initialization
    def greedy_init():
        uncovered = np.ones(N, dtype=bool)
        code = []
        while np.any(uncovered):
            # Find codeword covering most uncovered
            best_c = -1
            best_count = -1
            for c in range(N):
                cnt = np.sum(ball_masks[c] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
            if best_count <= 0:
                break
            code.append(best_c)
            uncovered[ball_masks[best_c]] = False
        return code

    def compute_coverage_counts(code):
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            counts[balls[c]] += 1
        return counts

    def is_covered(counts):
        return np.all(counts > 0)

    # Multi-exchange local search
    def local_search(code, time_limit):
        """Run multi-exchange local search starting from code."""
        code = list(code)
        counts = compute_coverage_counts(code)
        covered_all = is_covered(counts)
        best_code = code.copy()
        best_len = len(code)
        best_covered = covered_all

        def try_exchange(removes, adds):
            """Try removing `removes` and adding `adds`. Return (accepted, new_counts)."""
            nonlocal counts, code, covered_all, best_code, best_len, best_covered
            if len(removes) == 0 and len(adds) == 0:
                return False, counts
            # Check no duplicates
            add_set = set(adds)
            if any(r in add_set for r in removes):
                return False, counts
            # Check adds not already in code
            code_set = set(code)
            for a in adds:
                if a in code_set:
                    return False, counts
            # Compute new counts
            new_counts = counts.copy()
            for r in removes:
                new_counts[balls[r]] -= 1
            for a in adds:
                new_counts[balls[a]] += 1
            new_covered = np.all(new_counts > 0)
            new_len = len(code) - len(removes) + len(adds)

            # Acceptance: prefer valid solutions with fewer codewords
            if new_covered and not covered_all:
                # Any valid is better than invalid
                code = [c for c in code if c not in set(removes)]
                code.extend(adds)
                counts = new_counts
                covered_all = True
                if new_len < best_len:
                    best_code = code.copy()
                    best_len = new_len
                    best_covered = True
                return True, new_counts
            elif covered_all and new_covered:
                if new_len < len(code):
                    code = [c for c in code if c not in set(removes)]
                    code.extend(adds)
                    counts = new_counts
                    if new_len < best_len:
                        best_code = code.copy()
                        best_len = new_len
                        best_covered = True
                    return True, new_counts
                elif new_len == len(code):
                    # Same size, same coverage: accept with some probability for diversity
                    if random.random() < 0.1:
                        code = [c for c in code if c not in set(removes)]
                        code.extend(adds)
                        counts = new_counts
                        return True, new_counts
            elif not covered_all and new_covered:
                code = [c for c in code if c not in set(removes)]
                code.extend(adds)
                counts = new_counts
                covered_all = True
                if new_len < best_len:
                    best_code = code.copy()
                    best_len = new_len
                    best_covered = True
                return True, new_counts
            # Invalid -> invalid: only accept if reduces uncovered count
            if not covered_all and not new_covered:
                old_uncov = np.sum(counts == 0)
                new_uncov = np.sum(new_counts == 0)
                if new_uncov < old_uncov - 2:  # significant improvement
                    code = [c for c in code if c not in set(removes)]
                    code.extend(adds)
                    counts = new_counts
                    return True, new_counts
            return False, counts

        # Main loop: try different exchange types
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # 1-for-1: remove one, add one
            for i in range(len(code)):
                if time.time() > time_limit:
                    break
                # Try removing code[i] and adding a random word
                for attempt in range(5):
                    candidate = random.randint(0, N - 1)
                    if candidate in code:
                        continue
                    accepted, _ = try_exchange([code[i]], [candidate])
                    if accepted:
                        improved = True
                        break

            # 2-for-1: remove two, add one
            if len(code) >= 2:
                for i in range(len(code)):
                    if time.time() > time_limit:
                        break
                    for j in range(i + 1, len(code)):
                        if time.time() > time_limit:
                            break
                        for attempt in range(3):
                            candidate = random.randint(0, N - 1)
                            if candidate in code:
                                continue
                            accepted, _ = try_exchange([code[i], code[j]], [candidate])
                            if accepted:
                                improved = True
                                break

            # 1-for-2: remove one, add two (useful when we need to split coverage)
            if not covered_all:
                for i in range(len(code)):
                    if time.time() > time_limit:
                        break
                    for attempt in range(3):
                        c1 = random.randint(0, N - 1)
                        c2 = random.randint(0, N - 1)
                        if c1 in code or c2 in code or c1 == c2:
                            continue
                        accepted, _ = try_exchange([code[i]], [c1, c2])
                        if accepted:
                            improved = True
                            break

            # 2-for-2: remove two, add two
            if len(code) >= 2:
                for i in range(len(code)):
                    if time.time() > time_limit:
                        break
                    for j in range(i + 1, len(code)):
                        if time.time() > time_limit:
                            break
                        for attempt in range(2):
                            c1 = random.randint(0, N - 1)
                            c2 = random.randint(0, N - 1)
                            if c1 in code or c2 in code or c1 == c2:
                                continue
                            accepted, _ = try_exchange([code[i], code[j]], [c1, c2])
                            if accepted:
                                improved = True
                                break

        # Final: try to remove any redundant codeword
        for i in range(len(code) - 1, -1, -1):
            if time.time() > time_limit:
                break
            accepted, _ = try_exchange([code[i]], [])
            if accepted:
                pass  # already updated

        return best_code if best_covered else code

    # Multi-start
    best_valid = None
    best_len = N + 1

    # Start 1: greedy
    code0 = greedy_init()
    if time.time() < deadline - 20:
        result = local_search(code0, deadline - 20)
        if is_covered(compute_coverage_counts(result)):
            if len(result) < best_len:
                best_len = len(result)
                best_valid = result

    # Additional random starts
    ball_sz = 1 + n * (q - 1) * radius
    heuristic_size = max(1, int(N / ball_sz) + 15)

    num_starts = 4
    for s in range(num_starts):
        if time.time() > deadline - 15:
            break
        # Random initialization near heuristic size
        size = heuristic_size + random.randint(-10, 20)
        size = max(5, min(size, N))
        code_s = random.sample(range(N), size)
        # Quick coverage check; if not covered, pad with greedy
        counts = compute_coverage_counts(code_s)
        if not is_covered(counts):
            uncovered = counts == 0
            while np.any(uncovered) and time.time() < deadline - 15:
                best_c = -1
                best_cnt = -1
                for c in range(N):
                    cnt = np.sum(ball_masks[c] & uncovered)
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_c = c
                if best_cnt <= 0:
                    break
                code_s.append(best_c)
                uncovered[ball_masks[best_c]] = False
                counts = compute_coverage_counts(code_s)
                uncovered = counts == 0

        if is_covered(compute_coverage_counts(code_s)) and time.time() < deadline - 10:
            result = local_search(code_s, deadline - 10)
            if is_covered(compute_coverage_counts(result)):
                if len(result) < best_len:
                    best_len = len(result)
                    best_valid = result

    # Fallback
    if best_valid is None:
        best_valid = code0

    # Final validation
    final_counts = compute_coverage_counts(best_valid)
    if not is_covered(final_counts):
        # Greedy fallback
        uncovered = np.ones(N, dtype=bool)
        code_set = []
        while np.any(uncovered):
            best_c = -1
            best_cnt = -1
            for c in range(N):
                cnt = np.sum(ball_masks[c] & uncovered)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_cnt <= 0:
                break
            code_set.append(best_c)
            uncovered[ball_masks[best_c]] = False
        best_valid = code_set

    return [list(all_words[i]) for i in best_valid]
# EVOLVE-BLOCK-END
