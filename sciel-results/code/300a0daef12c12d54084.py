import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    n = 4096
    max_sum = 3 * n  # 12288
    best_set = []

    def check_valid(S):
        """Verify that all unordered triple sums (with repetition) are distinct."""
        sums = set()
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    s = S[i] + S[j] + S[l]
                    if s in sums:
                        return False
                    sums.add(s)
        return True

    def get_new_sums(x, S):
        """Generate all triple sums involving x and elements of S."""
        new_sums = set()
        new_sums.add(3 * x)
        for y in S:
            new_sums.add(2 * x + y)
        for i in range(len(S)):
            for j in range(i, len(S)):
                new_sums.add(x + S[i] + S[j])
        return new_sums

    def can_add(x, S, existing_sums):
        """Check if x can be added to S without creating duplicate triple sums."""
        new_sums = get_new_sums(x, S)
        k = len(S)
        expected = 1 + k + k * (k + 1) // 2
        if len(new_sums) != expected:
            return False, set()
        if new_sums & existing_sums:
            return False, set()
        return True, new_sums

    def compute_all_sums(S):
        """Compute all triple sums of set S."""
        sums = set()
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    sums.add(S[i] + S[j] + S[l])
        return sums

    def greedy_build(start_set, order):
        """Greedily build a set starting from start_set, trying elements in given order."""
        S = list(start_set)
        existing_sums = compute_all_sums(S)

        for x in order:
            if x in S:
                continue
            valid, new_sums = can_add(x, S, existing_sums)
            if valid:
                S.append(x)
                existing_sums.update(new_sums)
        return S

    # Strategy 1: Power-of-3 base, then greedy extend (smallest first)
    base = [1, 3, 9, 27, 81, 243, 729, 2187]
    remaining = [x for x in range(1, n + 1) if x not in base]
    S1 = greedy_build(base, remaining)
    if len(S1) > len(best_set):
        best_set = S1

    # Strategy 2: Pure greedy smallest-first
    S2 = []
    sums2 = set()
    for x in range(1, n + 1):
        valid, new_sums = can_add(x, S2, sums2)
        if valid:
            S2.append(x)
            sums2.update(new_sums)
    if len(S2) > len(best_set):
        best_set = S2

    # Strategy 3: Randomized greedy with multiple seeds
    for seed in range(300):
        if time.time() > deadline:
            break
        random.seed(seed)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)

        S = []
        existing_sums = set()
        for x in candidates:
            if time.time() > deadline:
                break
            valid, new_sums = can_add(x, S, existing_sums)
            if valid:
                S.append(x)
                existing_sums.update(new_sums)

        if len(S) > len(best_set):
            best_set = S

    # Strategy 4: Start from power-of-3, try randomized extension
    for seed in range(300):
        if time.time() > deadline:
            break
        random.seed(seed + 10000)
        base = [1, 3, 9, 27, 81, 243, 729, 2187]
        remaining = [x for x in range(1, n + 1) if x not in base]
        random.shuffle(remaining)

        S = list(base)
        existing_sums = compute_all_sums(S)

        for x in remaining:
            if time.time() > deadline:
                break
            valid, new_sums = can_add(x, S, existing_sums)
            if valid:
                S.append(x)
                existing_sums.update(new_sums)

        if len(S) > len(best_set):
            best_set = S

    # Strategy 5: Larger geometric base sequences
    # Try different bases to find better starting points
    bases_to_try = [
        [1, 4, 16, 64, 256, 1024, 4096],  # powers of 4
        [1, 5, 25, 125, 625, 3125],  # powers of 5
        [1, 7, 49, 343, 2401],  # powers of 7
    ]
    for base in bases_to_try:
        if time.time() > deadline:
            break
        remaining = [x for x in range(1, n + 1) if x not in base]
        S = greedy_build(base, remaining)
        if len(S) > len(best_set):
            best_set = S

        # Also try randomized extension from this base
        for seed in range(50):
            if time.time() > deadline:
                break
            random.seed(seed + 20000)
            remaining = [x for x in range(1, n + 1) if x not in base]
            random.shuffle(remaining)

            S = list(base)
            existing_sums = compute_all_sums(S)

            for x in remaining:
                if time.time() > deadline:
                    break
                valid, new_sums = can_add(x, S, existing_sums)
                if valid:
                    S.append(x)
                    existing_sums.update(new_sums)

            if len(S) > len(best_set):
                best_set = S

    # Strategy 6: Backtracking with limited depth to fix conflicts
    # Try to improve the best set by removing one element and trying alternatives
    if best_set and time.time() < deadline - 5.0:
        S = list(best_set)
        improved = False
        for remove_idx in range(len(S)):
            if time.time() > deadline - 1.0:
                break
            # Try removing element at remove_idx
            S_trial = S[:remove_idx] + S[remove_idx+1:]
            existing_sums = compute_all_sums(S_trial)

            # Try to add a different element in its place
            for candidate in range(1, n + 1):
                if candidate in S_trial:
                    continue
                valid, new_sums = can_add(candidate, S_trial, existing_sums)
                if valid:
                    S_new = S_trial + [candidate]
                    # Verify this is actually valid
                    if check_valid(S_new):
                        S = S_new
                        improved = True
                        break

        if improved:
            best_set = S

    # Final verification
    if check_valid(best_set):
        return best_set

    # Fallback: return a known valid small set
    return [1, 3, 9, 27, 81, 243, 729, 2187]
