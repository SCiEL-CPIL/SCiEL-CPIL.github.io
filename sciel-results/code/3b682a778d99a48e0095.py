import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85

    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    N_ELEMENTS = 129

    # Precompute for each sum s in [128, 256], the list of pairs (i, j) with i <= j
    # such that i + j = s and 0 <= i, j <= 128
    pairs_for_sum = [None] * (N_HIGH + 1)
    for s in range(N_LOW, N_HIGH + 1):
        pairs = []
        i_min = max(0, s - 128)
        i_max = min(128, s // 2)
        for i in range(i_min, i_max + 1):
            j = s - i
            if j <= 128 and j >= 0:
                pairs.append((i, j))
        if pairs:
            arr = np.array(pairs, dtype=np.int32).flatten()
            pairs_for_sum[s] = arr
        else:
            pairs_for_sum[s] = np.array([], dtype=np.int32)

    # Precompute the index arrays for fast numpy operations
    # For each sum, we have pairs as a flattened array of length 2*k
    # We'll create separate arrays for the first and second elements of each pair
    pair_first = [None] * (N_HIGH + 1)
    pair_second = [None] * (N_HIGH + 1)
    for s in range(N_LOW, N_HIGH + 1):
        if len(pairs_for_sum[s]) == 0:
            pair_first[s] = np.array([], dtype=np.int32)
            pair_second[s] = np.array([], dtype=np.int32)
        else:
            p = pairs_for_sum[s].reshape(-1, 2)
            pair_first[s] = p[:, 0]
            pair_second[s] = p[:, 1]

    def calc_reps(mask):
        """
        Calculate representation counts for all sums in [128, 256].
        mask: numpy array of 0s and 1s of length 129.
        Returns: numpy array of length 129 (indices 0..128 corresponding to sums 128..256).
        """
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        for idx, s in enumerate(range(N_LOW, N_HIGH + 1)):
            if len(pair_first[s]) == 0:
                continue
            p1 = pair_first[s]
            p2 = pair_second[s]
            # Get mask values for both elements in each pair
            m1 = mask[p1]
            m2 = mask[p2]
            # Product is 1 only if both are in A
            reps[idx] = np.sum(m1 * m2)
        return reps

    def get_objective(reps):
        """
        Objective: Minimize peak multiplicity, but heavily penalize uncovered sums.
        If any sum is uncovered (rep == 0), return a large penalty.
        Otherwise, return the max rep.
        """
        uncovered_count = np.sum(reps == 0)
        if uncovered_count > 0:
            return 1000 + uncovered_count * 100 + int(np.max(reps))
        else:
            return int(np.max(reps))

    def mask_to_list(mask):
        return [i for i in range(N_ELEMENTS) if mask[i] == 1]

    # --- Phase 1: Greedy Balanced Set Cover ---
    def greedy_balanced_cover():
        mask = np.zeros(N_ELEMENTS, dtype=np.int32)
        reps = calc_reps(mask)

        for step in range(N_ELEMENTS):
            if time.time() > deadline - 10:
                break

            best_elem = -1
            best_score = -1

            for c in range(N_ELEMENTS):
                if mask[c] == 1:
                    continue

                trial_mask = mask.copy()
                trial_mask[c] = 1
                trial_reps = calc_reps(trial_mask)

                new_covered = np.sum((trial_reps > 0) & (reps == 0))
                max_rep = int(np.max(trial_reps))

                score = new_covered * 1000 - max_rep

                if score > best_score:
                    best_score = score
                    best_elem = c

            if best_elem == -1:
                break

            mask[best_elem] = 1
            reps = calc_reps(mask)

            if np.all(reps > 0):
                break

        return mask

    # --- Phase 2: Local Search Refinement ---
    def local_search_refine(mask, time_limit=10.0):
        current_mask = mask.copy()
        current_reps = calc_reps(current_mask)
        current_obj = get_objective(current_reps)

        if np.any(current_reps == 0):
            while np.any(current_reps == 0) and time.time() < time_limit:
                uncovered = [s for s in range(RANGE_SIZE) if current_reps[s] == 0]
                if not uncovered:
                    break
                s_idx = random.choice(uncovered)
                s_val = N_LOW + s_idx
                p1 = pair_first[s_val]
                p2 = pair_second[s_val]

                added = False
                candidates = set()
                for i, j in zip(p1, p2):
                    if current_mask[i] == 1 and current_mask[j] == 0:
                        candidates.add(j)
                    elif current_mask[j] == 1 and current_mask[i] == 0:
                        candidates.add(i)

                if candidates:
                    best_c = None
                    best_max_rep = float('inf')
                    for c in candidates:
                        trial_mask = current_mask.copy()
                        trial_mask[c] = 1
                        trial_reps = calc_reps(trial_mask)
                        if np.all(trial_reps > 0):
                            mr = int(np.max(trial_reps))
                            if mr < best_max_rep:
                                best_max_rep = mr
                                best_c = c
                    if best_c is not None:
                        current_mask[best_c] = 1
                        current_reps = calc_reps(current_mask)
                        added = True

                if not added:
                    all_cands = set()
                    for s in uncovered:
                        s_val = N_LOW + s
                        all_cands.update(pair_first[s_val].tolist())
                        all_cands.update(pair_second[s_val].tolist())
                    if all_cands:
                        c = random.choice(list(all_cands))
                        current_mask[c] = 1
                        current_reps = calc_reps(current_mask)

        best_mask = current_mask.copy()
        best_obj = get_objective(current_reps)
        best_reps = current_reps.copy()

        T = 5.0
        cooling = 0.995
        no_improve = 0

        while time.time() < time_limit:
            if T < 0.1:
                break

            new_mask = current_mask.copy()
            move_type = random.random()

            if move_type < 0.5:
                present = np.where(current_mask == 1)[0]
                absent = np.where(current_mask == 0)[0]
                if len(present) == 0 or len(absent) == 0:
                    continue
                i = random.choice(present)
                j = random.choice(absent)
                new_mask[i] = 0
                new_mask[j] = 1
            else:
                if random.random() < 0.7:
                    present = np.where(current_mask == 1)[0]
                    if len(present) > 1:
                        i = random.choice(present)
                        new_mask[i] = 0
                else:
                    absent = np.where(current_mask == 0)[0]
                    if len(absent) > 0:
                        j = random.choice(absent)
                        new_mask[j] = 1

            new_reps = calc_reps(new_mask)
            new_obj = get_objective(new_reps)

            delta = new_obj - current_obj

            accept = False
            if delta < 0:
                accept = True
                no_improve = 0
            else:
                if delta > 100:
                    prob = 0.0
                else:
                    prob = np.exp(-delta / T)
                if random.random() < prob:
                    accept = True
                    no_improve += 1

            if accept:
                current_mask = new_mask
                current_reps = new_reps
                current_obj = new_obj

                if current_obj < best_obj:
                    best_obj = current_obj
                    best_mask = current_mask.copy()
                    best_reps = current_reps.copy()
                    no_improve = 0

            T *= cooling

            if no_improve > 500:
                current_mask = best_mask.copy()
                current_reps = best_reps.copy()
                current_obj = best_obj
                T = 5.0
                no_improve = 0

        return best_mask

    # Main Execution
    best_mask = None
    best_obj = float('inf')

    num_rounds = 5
    time_per_round = (deadline - start_time - 5) / num_rounds

    for r in range(num_rounds):
        if time.time() > deadline - 5:
            break

        mask = greedy_balanced_cover()

        round_deadline = time.time() + time_per_round
        refined_mask = local_search_refine(mask, time_limit=round_deadline - time.time())

        reps = calc_reps(refined_mask)
        obj = get_objective(reps)

        if obj < best_obj:
            best_obj = obj
            best_mask = refined_mask.copy()

    if best_mask is None:
        mask = np.zeros(N_ELEMENTS, dtype=np.int32)
        for i in range(64, 129):
            mask[i] = 1
        best_mask = mask
    else:
        reps = calc_reps(best_mask)
        if np.any(reps == 0):
            while np.any(reps == 0) and time.time() < deadline - 1:
                uncovered = [s for s in range(RANGE_SIZE) if reps[s] == 0]
                if not uncovered:
                    break
                s_idx = random.choice(uncovered)
                s_val = N_LOW + s_idx
                p1 = pair_first[s_val]
                p2 = pair_second[s_val]

                added = False
                for i, j in zip(p1, p2):
                    if best_mask[i] == 1 and best_mask[j] == 0:
                        best_mask[j] = 1
                        added = True
                        break
                    elif best_mask[j] == 1 and best_mask[i] == 0:
                        best_mask[i] = 1
                        added = True
                        break

                if not added:
                    best_mask[p1[0]] = 1
                    best_mask[p2[0]] = 1

                reps = calc_reps(best_mask)

            if np.any(reps == 0):
                mask = np.zeros(N_ELEMENTS, dtype=np.int32)
                for i in range(64, 129):
                    mask[i] = 1
                best_mask = mask

    return mask_to_list(best_mask)
