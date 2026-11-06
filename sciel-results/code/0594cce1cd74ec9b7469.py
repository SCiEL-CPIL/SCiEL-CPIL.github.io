import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 87.0

    T_min = 128
    T_max = 256
    N_max = 129
    R = T_max - T_min + 1  # 129

    best_A = None
    best_max = float('inf')

    def compute_counts(ind):
        """Compute unordered pair-sum representation counts for [T_min, T_max]."""
        conv = np.convolve(ind, ind, mode='full')
        counts = np.empty(R, dtype=np.int32)
        for i, s in enumerate(range(T_min, T_max + 1)):
            c_ord = conv[s]
            if s % 2 == 0:
                half = s // 2
                n_eq = 1 if (half < N_max and ind[half] == 1) else 0
            else:
                n_eq = 0
            counts[i] = (c_ord + n_eq) // 2
        return counts

    def max_count(counts):
        return int(np.max(counts))

    def is_valid(counts):
        return bool(np.all(counts > 0))

    def repair(A_set, ind, counts):
        """Repair gaps by adding elements with best gain."""
        for _ in range(100):
            if is_valid(counts):
                break
            gaps = np.where(counts == 0)[0]
            best_y = None
            best_gain = -1
            for y in range(N_max):
                if ind[y] == 1:
                    continue
                gain = 0
                for gi in gaps:
                    s = T_min + gi
                    a = s - y
                    if 0 <= a < N_max and (ind[a] == 1 or a == y):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_y = y
            if best_y is not None:
                A_set.add(best_y)
                ind[best_y] = 1
                counts = compute_counts(ind)
            else:
                break
        return is_valid(counts)

    def try_improve(A_initial, time_budget):
        nonlocal best_A, best_max
        A_set = set(A_initial)
        ind = np.zeros(N_max, dtype=np.int32)
        for x in A_set:
            ind[x] = 1
        counts = compute_counts(ind)

        if not is_valid(counts):
            if not repair(A_set, ind, counts):
                return

        current_max = max_count(counts)
        if current_max < best_max:
            best_max = current_max
            best_A = set(A_set)

        # Phase 1: Random swaps to reduce max multiplicity
        A_cur = set(A_set)
        ind_cur = ind.copy()
        counts_cur = counts.copy()
        max_cur = current_max
        no_improve = 0
        phase1_deadline = start_time + time_budget * 0.6

        while time.time() < phase1_deadline:
            if len(A_cur) == 0:
                break
            A_list_cur = list(A_cur)
            x = random.choice(A_list_cur)
            y = random.randint(0, N_max - 1)
            if y == x or ind_cur[y] == 1:
                continue

            ind_new = ind_cur.copy()
            ind_new[x] = 0
            ind_new[y] = 1
            c_new = compute_counts(ind_new)

            if not is_valid(c_new):
                no_improve += 1
                if no_improve > 800:
                    break
                continue

            m_new = max_count(c_new)

            if m_new < max_cur:
                A_cur = A_cur - {x}
                A_cur.add(y)
                ind_cur = ind_new
                counts_cur = c_new
                max_cur = m_new
                no_improve = 0
                if max_cur < best_max:
                    best_max = max_cur
                    best_A = set(A_cur)
            elif m_new == max_cur:
                if random.random() < 0.2:
                    A_cur = A_cur - {x}
                    A_cur.add(y)
                    ind_cur = ind_new
                    counts_cur = c_new
            else:
                no_improve += 1
                if no_improve > 800:
                    break

        # Phase 2: Targeted swaps to reduce peak multiplicity
        # Focus on sums with high counts
        phase2_deadline = start_time + time_budget * 0.95
        no_improve2 = 0

        while time.time() < phase2_deadline:
            if len(A_cur) == 0:
                break

            # Identify sums with maximum count
            max_c = max_count(counts_cur)
            high_sums = np.where(counts_cur >= max_c)[0]

            # Try to find a swap that reduces the count at a high-sum
            # by removing an element that contributes to many high-sums
            A_list_cur = list(A_cur)
            x = random.choice(A_list_cur)

            # Find candidate y that would reduce peak
            best_y = None
            best_score = None

            # Sample a few random y candidates
            for _ in range(5):
                y = random.randint(0, N_max - 1)
                if y == x or ind_cur[y] == 1:
                    continue

                ind_new = ind_cur.copy()
                ind_new[x] = 0
                ind_new[y] = 1
                c_new = compute_counts(ind_new)

                if not is_valid(c_new):
                    continue

                m_new = max_count(c_new)
                # Score: prefer lower max, then fewer high-count sums
                n_high = int(np.sum(c_new >= m_new))
                score = (m_new, n_high)

                if best_score is None or score < best_score:
                    best_score = score
                    best_y = y

            if best_y is None:
                no_improve2 += 1
                if no_improve2 > 300:
                    break
                continue

            ind_new = ind_cur.copy()
            ind_new[x] = 0
            ind_new[best_y] = 1
            c_new = compute_counts(ind_new)

            if not is_valid(c_new):
                no_improve2 += 1
                if no_improve2 > 300:
                    break
                continue

            m_new = max_count(c_new)

            if m_new < max_cur:
                A_cur = A_cur - {x}
                A_cur.add(best_y)
                ind_cur = ind_new
                counts_cur = c_new
                max_cur = m_new
                no_improve2 = 0
                if max_cur < best_max:
                    best_max = max_cur
                    best_A = set(A_cur)
            elif m_new == max_cur:
                # Accept if it reduces the number of peak sums
                old_peak_count = int(np.sum(counts_cur >= max_cur))
                new_peak_count = int(np.sum(c_new >= max_cur))
                if new_peak_count < old_peak_count or random.random() < 0.1:
                    A_cur = A_cur - {x}
                    A_cur.add(best_y)
                    ind_cur = ind_new
                    counts_cur = c_new
            else:
                no_improve2 += 1
                if no_improve2 > 300:
                    break

    # Try multiple diverse starting points
    starts = []
    # Seed: upper half
    starts.append(list(range(64, 129)))
    # Random sizes with fixed seeds for reproducibility
    for sz in [35, 40, 45, 50, 55, 60]:
        for trial in range(3):
            random.seed(42 + trial * 1000 + sz)
            starts.append(random.sample(range(0, N_max), sz))
    # Structured starts
    starts.append(list(range(0, 129, 2)))  # evens
    starts.append(list(range(0, 129, 2)) + [128])
    starts.append(list(range(1, 129, 2)))  # odds
    starts.append(list(range(0, 129, 3)) + list(range(2, 129, 3)))
    # Additional structured starts
    starts.append(list(range(32, 129, 2)))  # evens in upper portion
    starts.append(list(range(33, 129, 2)))  # odds in upper portion

    for i, s in enumerate(starts):
        remaining = deadline - time.time() - 3.0
        if remaining < 2.0:
            break
        # Allocate more time to earlier starts, less to later ones
        time_budget = min(remaining * 0.8, 15.0)
        if i > 5:
            time_budget = min(time_budget, 8.0)
        try_improve(s, time_budget)

    if best_A is None:
        return list(range(64, 129))
    return list(best_A)
