import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    n = 512
    k = 4

    # Precompute all valid 4-term APs in [1, n]
    # An AP is defined by (start, diff).
    # We will represent the set as a boolean array `S` of length n+1 (index 0 unused).
    # A set is invalid if there exists an AP such that S[start] & S[start+d] & S[start+2d] & S[start+3d] is True.

    # List of all (start, d) pairs
    ap_starts = []
    ap_diffs = []
    for d in range(1, (n - 1) // 3 + 1):
        max_start = n - 3 * d
        if max_start >= 1:
            starts = np.arange(1, max_start + 1)
            ap_starts.append(starts)
            ap_diffs.append(np.full(len(starts), d))

    # Flatten into a single array of indices for the 4 terms
    all_t1 = np.concatenate(ap_starts)
    all_t2 = np.concatenate([s + d for s, d in zip(ap_starts, ap_diffs)])
    all_t3 = np.concatenate([s + 2*d for s, d in zip(ap_starts, ap_diffs)])
    all_t4 = np.concatenate([s + 3*d for s, d in zip(ap_starts, ap_diffs)])

    num_aps = len(all_t1)

    def is_valid(S):
        """
        S is a boolean array of length n+1.
        Returns True if no 4-term AP is fully contained in S.
        """
        if not S.any():
            return True
        check = S[all_t1] & S[all_t2] & S[all_t3] & S[all_t4]
        return not check.any()

    def count_aps(S):
        """Count the number of 4-APs in S."""
        check = S[all_t1] & S[all_t2] & S[all_t3] & S[all_t4]
        return np.sum(check)

    # Strategy: Start with a structured base-3 Cantor-like set, then refine with local search.
    # The Cantor set (numbers with no '1' in base 3) is known to be 3-AP free, and thus 4-AP free.
    # This provides a much denser starting point than random greedy.
    
    def cantor_init():
        S = np.zeros(n + 1, dtype=bool)
        for x in range(1, n + 1):
            # Check if x has no '1' in base 3
            y = x
            valid = True
            while y > 0:
                if y % 3 == 1:
                    valid = False
                    break
                y //= 3
            if valid:
                S[x] = True
        return S

    def greedy_init():
        S = np.zeros(n + 1, dtype=bool)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        for x in candidates:
            S[x] = True
            if not is_valid(S):
                S[x] = False
        return S

    # Use Cantor set as initial state
    S = cantor_init()
    best_S = S.copy()
    best_size = np.sum(S)

    # Local search to add more elements
    # We will use a simple hill-climbing with random restarts/kicks
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01

    while time.time() < deadline - 5:
        if not is_valid(S):
            while not is_valid(S) and np.sum(S) > 0:
                idxs = np.where(S)[0]
                if len(idxs) == 0: break
                rem = random.choice(idxs)
                S[rem] = False

        improved = False

        # Phase 1: Try to add random elements
        unselected = np.where(~S)[0]
        unselected = unselected[unselected > 0]

        if len(unselected) > 0:
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                x = random.choice(unselected)
                S[x] = True
                if is_valid(S):
                    best_S = S.copy()
                    best_size = np.sum(S)
                    improved = True
                    break
                else:
                    S[x] = False

        if improved:
            continue

        # Phase 2: Try swaps (Remove 1, Add 2) to increase size
        selected = np.where(S)[0]
        selected = selected[selected > 0]

        if len(selected) > 0 and len(unselected) >= 2:
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                y = random.choice(selected)
                S[y] = False

                candidates = [x for x in unselected if x != y]
                if len(candidates) < 2:
                    S[y] = True
                    continue

                x1, x2 = random.sample(candidates, 2)
                S[x1] = True
                S[x2] = True

                if is_valid(S):
                    best_S = S.copy()
                    best_size = np.sum(S)
                    improved = True
                    break
                else:
                    S[x1] = False
                    S[x2] = False
                    S[y] = True

        if improved:
            continue

        # Phase 3: Neutral moves (Remove 1, Add 1)
        if len(selected) > 0 and len(unselected) > 0:
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                y = random.choice(selected)
                x = random.choice([u for u in unselected if u != y])

                S[y] = False
                S[x] = True

                if is_valid(S):
                    best_S = S.copy()
                    break
                else:
                    S[x] = False
                    S[y] = True

        if not improved:
            if len(selected) >= 5:
                kicks = random.sample(selected, 5)
                for k in kicks:
                    S[k] = False

        T *= cooling
        if T < min_T:
            T = min_T

    if not is_valid(best_S):
        while not is_valid(best_S) and np.sum(best_S) > 0:
            idxs = np.where(best_S)[0]
            if len(idxs) == 0: break
            rem = random.choice(idxs)
            best_S[rem] = False

    result = np.where(best_S)[0]
    result = result[result > 0]
    return sorted(result.tolist())

# Test run
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    S = set(res)
    valid = True
    for a in S:
        for d in range(1, (512 - a) // 3 + 1):
            if (a+d) in S and (a+2*d) in S and (a+3*d) in S:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
