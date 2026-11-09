import time
import random
import numpy as np

N = 256
MAX_EACH = 64

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    def is_sidon(lst):
        """Check if list is a Sidon set (pair sums including doubles are unique)."""
        if not lst:
            return True
        sums = set()
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                s = lst[i] + lst[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_diffs(lst):
        """Get nonzero absolute differences."""
        diffs = set()
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                d = abs(lst[i] - lst[j])
                if d > 0:
                    diffs.add(d)
        return diffs

    def validate(A, B):
        """Validate both sets are Sidon and have disjoint differences."""
        if not A or not B:
            return False
        if len(A) > MAX_EACH or len(B) > MAX_EACH:
            return False
        if not is_sidon(A):
            return False
        if not is_sidon(B):
            return False
        diffs_A = get_diffs(A)
        diffs_B = get_diffs(B)
        if diffs_A & diffs_B:
            return False
        return True

    best_A = None
    best_B = None
    best_score = 0

    def update_best(A, B):
        nonlocal best_A, best_B, best_score
        if validate(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = A[:]
                best_B = B[:]

    def greedy_build(candidates, blocked_diffs, max_size, rng):
        """
        Build a Sidon set greedily from candidates, avoiding blocked differences.
        Returns (result, used_sums, used_diffs).
        """
        result = []
        used_sums = set()
        used_diffs = set()

        for x in candidates:
            if len(result) >= max_size:
                break
            if elapsed() > deadline:
                break

            # Check Sidon property: all pair sums (including x+x) must be new
            valid = True
            for a in result:
                if (x + a) in used_sums:
                    valid = False
                    break
            if valid and (2 * x) in used_sums:
                valid = False
            if not valid:
                continue

            # Check difference constraints
            valid_diff = True
            for a in result:
                d = abs(x - a)
                if d > 0 and (d in blocked_diffs or d in used_diffs):
                    valid_diff = False
                    break
            if not valid_diff:
                continue

            # Accept
            result.append(x)
            for a in result:
                used_sums.add(x + a)
            used_sums.add(2 * x)
            for a in result:
                d = abs(x - a)
                if d > 0:
                    used_diffs.add(d)

        return result, used_sums, used_diffs

    def local_search_polish(A, B, rng, max_iters=200):
        """
        Polish a solution by trying to add elements and perform swaps.
        Returns improved (A, B).
        """
        A = A[:]
        B = B[:]
        improved = True
        iters = 0

        while improved and iters < max_iters and elapsed() < deadline:
            improved = False
            iters += 1

            # Try adding to A
            if len(A) < MAX_EACH:
                candidates = [x for x in range(1, N + 1) if x not in A]
                rng.shuffle(candidates)
                for x in candidates[:30]:
                    new_A = A + [x]
                    if is_sidon(new_A):
                        if not (get_diffs(new_A) & get_diffs(B)):
                            A = new_A
                            improved = True
                            break

            # Try adding to B
            if len(B) < MAX_EACH:
                candidates = [x for x in range(1, N + 1) if x not in B]
                rng.shuffle(candidates)
                for x in candidates[:30]:
                    new_B = B + [x]
                    if is_sidon(new_B):
                        if not (get_diffs(A) & get_diffs(new_B)):
                            B = new_B
                            improved = True
                            break

            # Try swapping in A: remove one, add one
            if len(A) > 1:
                idx = rng.randint(0, len(A) - 1)
                old = A[idx]
                candidates = [x for x in range(1, N + 1) if x not in A]
                rng.shuffle(candidates)
                for x in candidates[:20]:
                    new_A = [a for i, a in enumerate(A) if i != idx] + [x]
                    new_A.sort()
                    if is_sidon(new_A):
                        if not (get_diffs(new_A) & get_diffs(B)):
                            A = new_A
                            improved = True
                            break

            # Try swapping in B
            if len(B) > 1:
                idx = rng.randint(0, len(B) - 1)
                old = B[idx]
                candidates = [x for x in range(1, N + 1) if x not in B]
                rng.shuffle(candidates)
                for x in candidates[:20]:
                    new_B = [b for i, b in enumerate(B) if i != idx] + [x]
                    new_B.sort()
                    if is_sidon(new_B):
                        if not (get_diffs(A) & get_diffs(new_B)):
                            B = new_B
                            improved = True
                            break

        return A, B

    def build_A_then_B(seed):
        """
        Strategy: Build A first with a structured greedy, then build B avoiding A's diffs.
        """
        rng = random.Random(seed)

        # Build A with a mix of structured and random ordering
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)

        A, sums_A, diffs_A = greedy_build(candidates, set(), MAX_EACH, rng)

        # Build B avoiding A's differences
        candidates_B = list(range(1, N + 1))
        rng.shuffle(candidates_B)
        B, sums_B, diffs_B = greedy_build(candidates_B, diffs_A, MAX_EACH, rng)

        return A, B

    def interleaved_build(seed):
        """
        Strategy: Interleave building A and B, tracking both difference sets.
        """
        rng = random.Random(seed)

        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()

        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)

        for x in candidates:
            if elapsed() > deadline:
                break

            # Decide which set to try adding to
            if len(A) < len(B) - 2:
                order = [0]
            elif len(B) < len(A) - 2:
                order = [1]
            else:
                order = [0, 1] if rng.random() < 0.5 else [1, 0]

            for side in order:
                if side == 0 and len(A) < MAX_EACH:
                    valid = True
                    for a in A:
                        if (x + a) in sums_A:
                            valid = False
                            break
                    if valid and (2 * x) in sums_A:
                        valid = False
                    if not valid:
                        continue
                    valid_diff = True
                    for a in A:
                        d = abs(x - a)
                        if d > 0 and (d in diffs_A or d in diffs_B):
                            valid_diff = False
                            break
                    if not valid_diff:
                        continue
                    A.append(x)
                    for a in A:
                        sums_A.add(x + a)
                    sums_A.add(2 * x)
                    for a in A:
                        d = abs(x - a)
                        if d > 0:
                            diffs_A.add(d)
                    break

                elif side == 1 and len(B) < MAX_EACH:
                    valid = True
                    for b in B:
                        if (x + b) in sums_B:
                            valid = False
                            break
                    if valid and (2 * x) in sums_B:
                        valid = False
                    if not valid:
                        continue
                    valid_diff = True
                    for b in B:
                        d = abs(x - b)
                        if d > 0 and (d in diffs_A or d in diffs_B):
                            valid_diff = False
                            break
                    if not valid_diff:
                        continue
                    B.append(x)
                    for b in B:
                        sums_B.add(x + b)
                    sums_B.add(2 * x)
                    for b in B:
                        d = abs(x - b)
                        if d > 0:
                            diffs_B.add(d)
                    break

        return A, B

    def build_with_complement(seed):
        """
        Strategy: Build A, then try to build B from the complement of A's differences.
        Use a different candidate ordering for B.
        """
        rng = random.Random(seed)

        candidates_A = list(range(1, N + 1))
        rng.shuffle(candidates_A)
        A, sums_A, diffs_A = greedy_build(candidates_A, set(), MAX_EACH, rng)

        # For B, prefer elements that are far apart from each other and from A's diffs
        candidates_B = list(range(1, N + 1))
        # Sort by how many of B's potential differences would conflict with A's diffs
        def conflict_score(x):
            count = 0
            for y in range(1, N + 1):
                if y != x and abs(x - y) in diffs_A:
                    count += 1
            return count

        candidates_B.sort(key=lambda x: conflict_score(x))
        rng.shuffle(candidates_B)  # Add some randomness

        B, sums_B, diffs_B = greedy_build(candidates_B, diffs_A, MAX_EACH, rng)

        return A, B

    # Run multiple strategies with different seeds
    strategies = [build_A_then_B, interleaved_build, build_with_complement]

    for seed in range(500):
        if elapsed() > deadline:
            break

        strategy = strategies[seed % len(strategies)]
        A, B = strategy(seed)

        if A and B:
            update_best(A, B)

        # Apply local search polish to promising solutions
        if A and B and (len(A) + len(B)) >= best_score - 5:
            A_ls, B_ls = local_search_polish(A, B, random.Random(seed + 10000), max_iters=100)
            update_best(A_ls, B_ls)

    # Final local search on the best solution found
    if best_A and best_B and elapsed() < deadline:
        A_final, B_final = local_search_polish(best_A, best_B, random.Random(99999), max_iters=300)
        update_best(A_final, B_final)

    # Fallback
    if best_A is None or best_B is None:
        best_A = [1]
        best_B = [4]

    return [best_A, best_B]
