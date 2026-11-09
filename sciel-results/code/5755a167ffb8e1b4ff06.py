import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for 90s limit

    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    N_ELEMENTS = 129

    # Precompute pair indices for unordered pairs (i <= j)
    pairs_for_sum = [None] * (N_HIGH + 1)
    for s in range(N_LOW, N_HIGH + 1):
        pairs = []
        i_min = max(0, s - 128)
        i_max = min(128, s // 2)
        for i in range(i_min, i_max + 1):
            j = s - i
            if j <= 128 and j >= 0:
                pairs.append((i, j))
        pairs_for_sum[s] = np.array(pairs, dtype=np.int32).flatten() if pairs else np.array([], dtype=np.int32)

    def calc_reps(mask):
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        for idx, s in enumerate(range(N_LOW, N_HIGH + 1)):
            pairs = pairs_for_sum[s]
            if len(pairs) == 0:
                continue
            p = pairs.reshape(-1, 2)
            m_vals = mask[p]
            prods = m_vals[:, 0] * m_vals[:, 1]
            reps[idx] = np.sum(prods)
        return reps

    def get_objective(reps):
        uncovered_count = np.sum(reps == 0)
        if uncovered_count == 0:
            peak = int(np.max(reps))
            return peak
        else:
            return 10000 + uncovered_count * 100

    def mask_to_list(mask):
        return [i for i in range(129) if mask[i] == 1]

    # Phase 1: Initialization
    best_A = None
    best_obj = float('inf')
    best_mask = None

    # Try multiple initializations with different strategies
    for init_round in range(20):
        mask = np.zeros(129, dtype=np.int32)

        if init_round < 5:
            # Strategy 1: Random with forced 0 and 128
            mask[0] = 1
            mask[128] = 1
            k = random.randint(35, 55)
            candidates = list(range(1, 128))
            random.shuffle(candidates)
            for i in range(k):
                mask[candidates[i]] = 1
        elif init_round < 10:
            # Strategy 2: Symmetric set around 64
            # Include 0 and 128, and symmetric pairs
            mask[0] = 1
            mask[128] = 1
            k_pairs = random.randint(15, 30)
            candidates = list(range(1, 64))
            random.shuffle(candidates)
            for i in range(k_pairs):
                v = candidates[i]
                mask[v] = 1
                mask[128 - v] = 1
        else:
            # Strategy 3: Greedy coverage
            # Start with 0, 128, and iteratively add elements that cover most uncovered sums
            mask[0] = 1
            mask[128] = 1
            reps = calc_reps(mask)
            uncovered = set(range(N_LOW, N_HIGH + 1))
            for s in range(N_LOW, N_HIGH + 1):
                if reps[s - N_LOW] > 0:
                    uncovered.discard(s)

            for _ in range(30):
                if len(uncovered) == 0:
                    break
                best_elem = -1
                best_coverage = 0
                for c in range(1, 128):
                    if mask[c] == 1:
                        continue
                    # Check how many uncovered sums this element would cover
                    # An element c covers sum s if there exists a in A such that c + a = s, i.e., a = s - c is in A
                    coverage = 0
                    for s in list(uncovered):
                        a = s - c
                        if 0 <= a <= 128 and mask[a] == 1:
                            coverage += 1
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_elem = c
                if best_elem == -1 or best_coverage == 0:
                    # Pick a random element
                    best_elem = random.randint(1, 127)
                mask[best_elem] = 1
                # Update reps and uncovered
                reps = calc_reps(mask)
                uncovered = set()
                for s in range(N_LOW, N_HIGH + 1):
                    if reps[s - N_LOW] == 0:
                        uncovered.add(s)

        reps = calc_reps(mask)
        obj = get_objective(reps)

        if obj < best_obj:
            best_obj = obj
            best_mask = mask.copy()
            best_A = mask_to_list(mask)

    if best_mask is None:
        mask = np.zeros(129, dtype=np.int32)
        for i in range(64, 129):
            mask[i] = 1
        return mask_to_list(mask)

    current_mask = best_mask.copy()
    current_reps = calc_reps(current_mask)
    current_obj = get_objective(current_reps)

    # Simulated Annealing with adaptive temperature
    T = 100.0
    cooling_rate = 0.998
    min_T = 0.1
    no_improve_count = 0
    stagnation_limit = 500

    while time.time() < deadline:
        if T < min_T:
            break

        # Check for stagnation
        if no_improve_count > stagnation_limit:
            # Restart with a new random initialization
            mask = np.zeros(129, dtype=np.int32)
            mask[0] = 1
            mask[128] = 1
            k = random.randint(35, 55)
            candidates = list(range(1, 128))
            random.shuffle(candidates)
            for i in range(k):
                mask[candidates[i]] = 1
            current_mask = mask.copy()
            current_reps = calc_reps(current_mask)
            current_obj = get_objective(current_reps)
            T = 50.0
            no_improve_count = 0
            stagnation_limit = random.randint(300, 800)
            continue

        # Generate move
        new_mask = current_mask.copy()
        move_type = random.random()

        if move_type < 0.4:
            # Flip
            i = random.randint(0, 128)
            new_mask[i] = 1 - new_mask[i]
        elif move_type < 0.7:
            # Swap: pick one present, one absent, swap
            present = np.where(current_mask == 1)[0]
            absent = np.where(current_mask == 0)[0]
            if len(present) == 0 or len(absent) == 0:
                continue
            i = random.choice(present)
            j = random.choice(absent)
            new_mask[i] = 0
            new_mask[j] = 1
        elif move_type < 0.85:
            # Add/Remove
            if random.random() < 0.5:
                absent = np.where(current_mask == 0)[0]
                if len(absent) > 0:
                    j = random.choice(absent)
                    new_mask[j] = 1
            else:
                present = np.where(current_mask == 1)[0]
                if len(present) > 1:
                    i = random.choice(present)
                    new_mask[i] = 0
        else:
            # Multi-flip: flip 2-4 random bits
            n_flips = random.randint(2, 4)
            for _ in range(n_flips):
                i = random.randint(0, 128)
                new_mask[i] = 1 - new_mask[i]

        if np.array_equal(new_mask, current_mask):
            continue

        new_reps = calc_reps(new_mask)
        new_obj = get_objective(new_reps)

        delta_obj = new_obj - current_obj

        accept = False
        if delta_obj < 0:
            accept = True
            no_improve_count = 0
        else:
            if delta_obj / T > 100:
                prob = 0.0
            else:
                prob = np.exp(-delta_obj / T)
            if random.random() < prob:
                accept = True
                no_improve_count += 1

        if accept:
            current_mask = new_mask
            current_reps = new_reps
            current_obj = new_obj

            if current_obj < best_obj:
                best_obj = current_obj
                best_mask = current_mask.copy()
                best_A = mask_to_list(best_mask)
                no_improve_count = 0

        T *= cooling_rate

    # Final validation and repair if necessary
    final_reps = calc_reps(best_mask)
    if np.any(final_reps == 0):
        # Greedy repair
        missing_sums = [s for s in range(N_LOW, N_HIGH + 1) if final_reps[s - N_LOW] == 0]

        repair_mask = best_mask.copy()
        repair_reps = final_reps.copy()

        for _ in range(100):  # Max 100 repair steps
            if not np.any(repair_reps == 0):
                break

            # Find the most "difficult" uncovered sum (one with fewest candidate pairs)
            uncovered = [s for s in range(N_LOW, N_HIGH + 1) if repair_reps[s - N_LOW] == 0]
            if not uncovered:
                break

            # Pick a random uncovered sum
            s = random.choice(uncovered)
            pairs = pairs_for_sum[s]
            if len(pairs) == 0:
                continue

            # Find a pair where one element is already in A
            best_add_elem = -1
            best_new_uncovered = int(np.sum(repair_reps == 0))

            candidates = set()
            for idx in range(len(pairs) // 2):
                i = pairs[2 * idx]
                j = pairs[2 * idx + 1]
                candidates.add(i)
                candidates.add(j)

            for c in candidates:
                if repair_mask[c] == 1:
                    continue
                # Simulate adding c
                trial_mask = repair_mask.copy()
                trial_mask[c] = 1
                trial_reps = calc_reps(trial_mask)
                new_uncovered = int(np.sum(trial_reps == 0))
                if new_uncovered < best_new_uncovered:
                    best_new_uncovered = new_uncovered
                    best_add_elem = c
                    if new_uncovered == 0:
                        break

            if best_add_elem != -1:
                repair_mask[best_add_elem] = 1
                repair_reps = calc_reps(repair_mask)
            else:
                # Add a random element from candidates
                if candidates:
                    best_add_elem = random.choice(list(candidates))
                    repair_mask[best_add_elem] = 1
                    repair_reps = calc_reps(repair_mask)

        if not np.any(repair_reps == 0):
            best_mask = repair_mask
            best_A = mask_to_list(best_mask)
        else:
            # If repair failed, return the dense set [64..128] which is guaranteed valid
            mask = np.zeros(N_ELEMENTS, dtype=np.int32)
            for i in range(64, 129):
                mask[i] = 1
            best_A = mask_to_list(mask)

    return best_A
