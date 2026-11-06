import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    n = 128
    TARGET_START = 128
    TARGET_END = 256
    TARGETS = list(range(TARGET_START, TARGET_END + 1))
    NUM_TARGETS = TARGET_END - TARGET_START + 1  # 129 targets

    def build_unordered_counts(A_set):
        """Build the unordered pair count for each target t in [128,256]."""
        counts = np.zeros(NUM_TARGETS, dtype=np.int32)
        A_list = sorted(A_set)
        for i, a in enumerate(A_list):
            for b in A_list[i:]:  # b >= a, unordered
                s = a + b
                if TARGET_START <= s <= TARGET_END:
                    counts[s - TARGET_START] += 1
        return counts

    def evaluate(A_set):
        """Return (max_multiplicity, covered) for the given set A."""
        counts = build_unordered_counts(A_set)
        max_mult = int(np.max(counts))
        covered = bool(np.all(counts > 0))
        return max_mult, covered

    def update_counts_add(counts, A_set, x):
        """Update counts after adding x to A. Returns new counts (copy)."""
        new_counts = counts.copy()
        for a in A_set:
            s = a + x
            if TARGET_START <= s <= TARGET_END:
                new_counts[s - TARGET_START] += 1
        # Also (x, x) if x+x is in range
        s = 2 * x
        if TARGET_START <= s <= TARGET_END:
            new_counts[s - TARGET_START] += 1
        return new_counts

    def update_counts_remove(counts, A_set, x):
        """Update counts after removing x from A. Returns new counts (copy)."""
        new_counts = counts.copy()
        for a in A_set:
            if a == x:
                continue
            s = a + x
            if TARGET_START <= s <= TARGET_END:
                new_counts[s - TARGET_START] -= 1
        # Remove (x, x)
        s = 2 * x
        if TARGET_START <= s <= TARGET_END:
            new_counts[s - TARGET_START] -= 1
        return new_counts

    def compute_cost(counts):
        """Compute a smooth cost function for the SA search."""
        max_mult = int(np.max(counts))
        uncovered = int(np.sum(counts == 0))
        var = float(np.var(counts))
        return max_mult + 100 * uncovered + 0.5 * var

    def local_search(A_set, time_budget, seed=None):
        """Local search with incremental evaluation and swap moves."""
        if seed is not None:
            random.seed(seed)

        current_A = set(A_set)
        current_counts = build_unordered_counts(current_A)

        # Best tracking
        best_A = set(current_A)
        best_counts = current_counts.copy()
        best_max = int(np.max(best_counts))
        best_covered = bool(np.all(best_counts > 0))
        best_score = best_max if best_covered else float('inf')

        T = 50.0
        T_min = 1.0
        cooling = 0.9995

        while time.time() < time_budget and T > T_min:
            current_cost = compute_cost(current_counts)

            move_type = random.random()
            new_A = None
            new_counts = None
            new_cost = None

            if move_type < 0.4:
                # Add move
                x = random.randint(0, 128)
                if x not in current_A:
                    new_A = current_A | {x}
                    new_counts = update_counts_add(current_counts, current_A, x)
                    new_cost = compute_cost(new_counts)
            elif move_type < 0.7:
                # Remove move
                if len(current_A) > 10:
                    x = random.choice(list(current_A))
                    new_A = current_A - {x}
                    new_counts = update_counts_remove(current_counts, current_A, x)
                    new_cost = compute_cost(new_counts)
            else:
                # Swap move
                if len(current_A) > 10:
                    x = random.choice(list(current_A))
                    y = random.randint(0, 128)
                    if y not in current_A:
                        new_A = (current_A - {x}) | {y}
                        temp_counts = update_counts_remove(current_counts, current_A, x)
                        new_counts = update_counts_add(temp_counts, current_A - {x}, y)
                        new_cost = compute_cost(new_counts)

            if new_A is not None:
                delta = new_cost - current_cost
                if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                    current_A = new_A
                    current_counts = new_counts

                    new_max = int(np.max(new_counts))
                    new_covered = bool(np.all(new_counts > 0))
                    new_score = new_max if new_covered else float('inf')
                    if new_score < best_score:
                        best_score = new_score
                        best_A = set(new_A)
                        best_counts = new_counts.copy()

            T *= cooling

        # Final Polish: Try to reduce max multiplicity further
        if best_score < float('inf'):
            best_A_list = sorted(best_A)
            improved = True
            while improved and time.time() < time_budget:
                improved = False
                best_max = int(np.max(best_counts))
                for x in best_A_list[:]:
                    if len(best_A) <= 10:
                        break
                    test_counts = update_counts_remove(best_counts, best_A, x)
                    test_max = int(np.max(test_counts))
                    test_covered = bool(np.all(test_counts > 0))

                    if test_covered and test_max < best_max:
                        best_A = best_A - {x}
                        best_counts = test_counts
                        best_max = test_max
                        best_score = best_max
                        improved = True
                        best_A_list = sorted(best_A)
                        break
                    elif not test_covered:
                        uncovered = np.where(test_counts == 0)[0]
                        if len(uncovered) == 0:
                            continue

                        best_y = None
                        best_y_cost = float('inf')
                        for _ in range(5):
                            y = random.randint(0, 128)
                            if y in best_A:
                                continue
                            candidate_counts = update_counts_add(test_counts, best_A - {x}, y)
                            cand_max = int(np.max(candidate_counts))
                            cand_covered = bool(np.all(candidate_counts > 0))
                            if cand_covered:
                                cost = cand_max
                                if cost < best_y_cost:
                                    best_y_cost = cost
                                    best_y = y
                                    best_y_counts = candidate_counts
                        if best_y is not None:
                            best_A = (best_A - {x}) | {best_y}
                            best_counts = best_y_counts
                            best_max = int(np.max(best_counts))
                            best_score = best_max
                            improved = True
                            best_A_list = sorted(best_A)
                            break

        return best_A, best_score

    # Multi-start: try each initial solution with local search
    best_A = None
    best_score = float('inf')

    def make_initials():
        initials = []
        # Full set
        initials.append(set(range(0, 129)))
        # Half: [64, 128]
        initials.append(set(range(64, 129)))
        # Every other in [0,128]: even numbers
        initials.append(set(range(0, 129, 2)))
        # Odd numbers
        initials.append(set(range(1, 129, 2)))
        # Random sets of various sizes
        random.seed(123)
        for size in [40, 50, 60, 70, 80, 90, 100]:
            s = set(random.sample(range(0, 129), size))
            initials.append(s)
        # Structured: multiples of k
        for k in [2, 3, 4]:
            s = set(range(0, 129, k))
            initials.append(s)
        # Complementary pairs
        s = set()
        for i in range(0, 129, 2):
            s.add(i)
            s.add(128 - i)
        initials.append(s)
        return initials

    def fix_coverage(A_set):
        """Greedily add elements to ensure all targets are covered."""
        A = set(A_set)
        for t in TARGETS:
            found = False
            for a in A:
                b = t - a
                if b >= a and b in A:
                    found = True
                    break
            if not found:
                mid = t // 2
                if 0 <= mid <= 128:
                    A.add(mid)
                if t % 2 == 1 and mid + 1 <= 128:
                    A.add(mid + 1)
        return A

    initials = make_initials()

    # Shuffle for diversity
    random.seed(42)
    random.shuffle(initials)

    # Phase 1: Quick evaluation of all initials
    scored_initials = []
    for init in initials:
        if time.time() > start_time + 10:
            break
        s, c = evaluate(init)
        cost = s if c else s + 10000
        scored_initials.append((cost, init))

    # Sort by cost (best first)
    scored_initials.sort(key=lambda x: x[0])

    # Phase 2: Local search from best initials
    for i, (cost, init) in enumerate(scored_initials):
        if time.time() >= deadline - 5:
            break
        remaining_starts = len(scored_initials) - i
        remaining_time = deadline - 5 - time.time()
        if remaining_time <= 0:
            break
        time_alloc = max(2, min(remaining_time / remaining_starts, 25))

        seed = 1000 + i
        ls_result, ls_score = local_search(init, time.time() + time_alloc, seed=seed)

        if ls_score < best_score:
            best_score = ls_score
            best_A = ls_result

    # Phase 3: Additional random restarts if time remains
    if time.time() < deadline - 10 and best_A is not None:
        num_extra = 3
        for j in range(num_extra):
            if time.time() >= deadline - 5:
                break
            random.seed(2000 + j)
            size = random.randint(40, 90)
            init = set(random.sample(range(0, 129), size))
            init = fix_coverage(init)

            time_alloc = max(2, (deadline - 5 - time.time()) / (num_extra - j))
            ls_result, ls_score = local_search(init, time.time() + time_alloc, seed=3000 + j)

            if ls_score < best_score:
                best_score = ls_score
                best_A = ls_result

    # Fallback: ensure we have a valid solution
    if best_A is None or best_score == float('inf'):
        best_A = set(range(0, 129))
        best_score, best_covered = evaluate(best_A)
        if not best_covered:
            best_A = set(range(64, 129))

    # Final verification
    final_score, final_covered = evaluate(best_A)
    if not final_covered:
        best_A = set(range(0, 129))
        final_score, final_covered = evaluate(best_A)

    return sorted(best_A)
