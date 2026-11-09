import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (evaluator kills at 90s)
    random.seed(42)  # Reproducibility

    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    MAX_VAL = 128
    N_DOMAIN = MAX_VAL + 1  # 129

    # Precompute FFT sizes
    # We need convolution of two vectors of length N_DOMAIN
    # Result length is 2*N_DOMAIN - 1 = 257
    # Next power of 2: 512
    fft_size = 1
    while fft_size < 2 * N_DOMAIN - 1:
        fft_size *= 2

    def calc_reps(A):
        """Returns array of counts for sums N_LOW..N_HIGH using unordered pairs (i<=j).
        Uses FFT convolution for speed."""
        if not A:
            return np.zeros(RANGE_SIZE, dtype=np.int32)

        # Create indicator vector
        f = np.zeros(N_DOMAIN, dtype=np.float64)
        for x in A:
            f[x] = 1.0

        # FFT convolution: f * f gives ordered pair counts
        F = np.fft.fft(f, n=fft_size)
        conv = np.fft.ifft(F * F).real
        # conv[s] = number of ordered pairs (a,b) with a+b=s, a,b in A
        # For unordered with i<=j: count = (ordered + diagonal) / 2
        # diagonal at sum s: 1 if s/2 is integer and s/2 in A
        # So unordered = (conv[s] + diag[s]) / 2 where diag[s] = 1 if s even and s//2 in A

        # Extract relevant range
        lo_idx = N_LOW
        hi_idx = N_HIGH + 1  # exclusive
        ordered = conv[lo_idx:hi_idx]

        # Compute diagonal contribution
        # For sum s, diagonal exists if s is even and s//2 is in A
        sums = np.arange(N_LOW, N_HIGH + 1)
        diag = np.zeros(RANGE_SIZE, dtype=np.float64)
        even_mask = (sums % 2 == 0)
        if np.any(even_mask):
            half_sums = sums[even_mask] // 2
            # Check if half_sums are in A and in [0, 128]
            in_range = (half_sums >= 0) & (half_sums <= MAX_VAL)
            if np.any(in_range):
                half_valid = half_sums[in_range]
                in_A = np.isin(half_valid, list(A))
                diag[even_mask][in_range][in_A] = 1.0

        # Unordered count = (ordered + diag) / 2
        unordered = (ordered + diag) / 2.0

        # Round to integer (FFT numerical error)
        counts = np.round(unordered).astype(np.int32)
        return counts

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    def get_total(reps):
        return int(np.sum(reps))

    def objective(reps):
        """Primary: peak. Secondary: total representations (prefer sparser)."""
        return (get_peak(reps), get_total(reps))

    def element_contribution(x, A, reps):
        """Return (peak_contrib, coverage_risk) for element x."""
        A_arr = np.array(sorted(A), dtype=np.int32)
        peak = get_peak(reps)

        sums = x + A_arr
        mask = (sums >= N_LOW) & (sums <= N_HIGH)
        if not np.any(mask):
            return 0, 0

        valid_sums = sums[mask]
        indices = valid_sums - N_LOW
        rep_vals = reps[indices]

        peak_count = int(np.sum(rep_vals == peak))
        coverage_risk = int(np.sum(rep_vals == 1))

        return peak_count, coverage_risk

    def find_peak_elements(A, reps):
        """Find elements that contribute most to peak sums."""
        peak = get_peak(reps)
        if peak == 0:
            return []

        A_list = list(A)
        scores = {}
        for x in A_list:
            pc, _ = element_contribution(x, A, reps)
            scores[x] = pc

        sorted_elems = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_elems

    best_A = None
    best_obj = (float('inf'), float('inf'))

    def update_best(A, reps):
        nonlocal best_A, best_obj
        if is_covered(reps):
            obj = objective(reps)
            if obj < best_obj:
                best_obj = obj
                best_A = list(A)

    def random_init():
        A = set()
        # Include structural elements for good coverage
        for x in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            A.add(x)
        for x in [120, 121, 122, 123, 124, 125, 126, 127, 128]:
            A.add(x)
        # Mid-range
        for x in [60, 61, 62, 63, 64, 65, 66, 67]:
            A.add(x)
        # Fill with random
        while len(A) < 35:
            A.add(random.randint(0, 128))
        return A

    def repair_coverage(A):
        """Add elements until all sums in [128,256] are covered."""
        A = set(A)
        reps = calc_reps(A)
        tries = 0
        while not is_covered(reps) and tries < 200:
            missing = np.where(reps == 0)[0] + N_LOW
            # For each missing sum, find a good element to add
            best_candidate = None
            best_gain = 0
            for s in missing:
                # Find elements x such that s-x is in [0,128] and x not in A
                candidates = set()
                for x in range(0, 129):
                    y = s - x
                    if 0 <= y <= 128:
                        candidates.add(x)
                candidates -= A
                # Only evaluate a subset of candidates for speed
                for c in list(candidates)[:15]:
                    new_A = A | {c}
                    new_reps = calc_reps(new_A)
                    new_missing = np.sum(new_reps == 0)
                    gain = len(missing) - new_missing
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = c
            if best_candidate is not None:
                A.add(best_candidate)
                reps = calc_reps(A)
            else:
                A.add(random.randint(0, 128))
                reps = calc_reps(A)
            tries += 1
        return A, reps

    # Phase 1: Multi-start Local Search with peak-aware moves
    num_restarts = 80

    for r in range(num_restarts):
        if time.time() > deadline:
            break

        current_A = random_init()
        current_A, current_reps = repair_coverage(current_A)

        if not is_covered(current_reps):
            continue

        update_best(current_A, current_reps)

        # Local Search: Simulated Annealing with peak-aware moves
        T = 2.0
        cooling = 0.998
        min_T = 0.02
        stuck_counter = 0
        max_stuck = 300
        iterations = 0

        while time.time() < deadline and T > min_T:
            if stuck_counter > max_stuck:
                break

            current_peak = get_peak(current_reps)
            current_total = get_total(current_reps)
            move_type = random.random()

            if len(current_A) > 20 and move_type < 0.5:
                # Try Remove: prioritize elements with high peak contribution
                peak_elems = find_peak_elements(current_A, current_reps)
                if peak_elems:
                    # Weighted selection: higher peak contribution -> higher chance
                    weights = []
                    for x in peak_elems:
                        pc, _ = element_contribution(x, current_A, current_reps)
                        weights.append(pc + 1)
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cum = 0
                    item_to_remove = peak_elems[-1]
                    for i, x in enumerate(peak_elems):
                        cum += weights[i]
                        if r <= cum:
                            item_to_remove = x
                            break
                else:
                    item_to_remove = random.choice(list(current_A))

                new_A = set(current_A)
                new_A.remove(item_to_remove)
                new_reps = calc_reps(new_A)

                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    new_total = get_total(new_reps)

                    if new_peak < current_peak or (new_peak == current_peak and new_total < current_total):
                        current_A = new_A
                        current_reps = new_reps
                        update_best(current_A, current_reps)
                        stuck_counter = 0
                    elif new_peak == current_peak:
                        # Accept with some probability to explore
                        if random.random() < 0.2:
                            current_A = new_A
                            current_reps = new_reps
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                    else:
                        # Peak increased, accept with low probability
                        prob = np.exp(-(new_peak - current_peak) / T)
                        if random.random() < prob:
                            current_A = new_A
                            current_reps = new_reps
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            elif move_type < 0.8:
                # Try Swap: remove a high-peak element, add a random one
                peak_elems = find_peak_elements(current_A, current_reps)
                if peak_elems:
                    item_to_remove = peak_elems[0]  # Highest peak contributor
                else:
                    item_to_remove = random.choice(list(current_A))

                new_item = random.randint(0, 128)
                if new_item == item_to_remove:
                    continue

                new_A = set(current_A)
                new_A.remove(item_to_remove)
                new_A.add(new_item)

                new_reps = calc_reps(new_A)

                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    new_total = get_total(new_reps)

                    if new_peak < current_peak or (new_peak == current_peak and new_total < current_total):
                        current_A = new_A
                        current_reps = new_reps
                        update_best(current_A, current_reps)
                        stuck_counter = 0
                    else:
                        prob = np.exp(-(new_peak - current_peak) / T)
                        if random.random() < prob:
                            current_A = new_A
                            current_reps = new_reps
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            else:
                # Try targeted swap: replace a peak element with one that helps low-coverage sums
                peak_elems = find_peak_elements(current_A, current_reps)
                if not peak_elems:
                    stuck_counter += 1
                    T *= cooling
                    continue

                item_to_remove = peak_elems[0]

                # Find sums with low coverage (count == 1 or 2)
                low_cov_sums = np.where(current_reps <= 2)[0] + N_LOW
                if len(low_cov_sums) == 0:
                    stuck_counter += 1
                    T *= cooling
                    continue

                # Try candidates that help cover low-coverage sums
                found = False
                for _ in range(15):
                    s = random.choice(low_cov_sums.tolist())
                    # Find x such that s - x is in current_A
                    candidates = []
                    for y in current_A:
                        x = s - y
                        if 0 <= x <= 128 and x != item_to_remove:
                            candidates.append(x)
                    if candidates:
                        new_item = random.choice(candidates)
                        if new_item not in current_A:
                            new_A = set(current_A)
                            new_A.remove(item_to_remove)
                            new_A.add(new_item)
                            new_reps = calc_reps(new_A)
                            if is_covered(new_reps):
                                new_peak = get_peak(new_reps)
                                new_total = get_total(new_reps)
                                if new_peak < current_peak or (new_peak == current_peak and new_total < current_total):
                                    current_A = new_A
                                    current_reps = new_reps
                                    update_best(current_A, current_reps)
                                    stuck_counter = 0
                                    found = True
                                    break
                if not found:
                    stuck_counter += 1

            T *= cooling
            iterations += 1

    # Final Polish: Peak-targeted refinement
    if best_A:
        A = set(best_A)
        reps = calc_reps(A)

        # Phase 2a: Greedy removal of redundant elements
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Sort by peak contribution descending
            elems_sorted = sorted(A, key=lambda e: -element_contribution(e, A, reps)[0])
            for x in elems_sorted:
                new_A = set(A)
                new_A.remove(x)
                new_reps = calc_reps(new_A)
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    old_peak = get_peak(reps)
                    new_total = get_total(new_reps)
                    old_total = get_total(reps)
                    if new_peak < old_peak or (new_peak == old_peak and new_total <= old_total):
                        A = new_A
                        reps = new_reps
                        changed = True
                        update_best(A, reps)
                        break

        # Phase 2b: Targeted swaps to reduce peak
        if time.time() < deadline:
            for _ in range(2000):
                if time.time() > deadline:
                    break

                peak = get_peak(reps)
                total = get_total(reps)
                peak_elems = find_peak_elements(A, reps)
                if not peak_elems:
                    break

                improved = False
                for x in peak_elems[:8]:  # Try top 8 peak contributors
                    if time.time() > deadline:
                        break
                    # Try swapping x with various candidates
                    for _ in range(20):
                        new_item = random.randint(0, 128)
                        if new_item == x:
                            continue
                        new_A = set(A)
                        new_A.remove(x)
                        new_A.add(new_item)
                        new_reps = calc_reps(new_A)
                        if is_covered(new_reps):
                            new_peak = get_peak(new_reps)
                            new_total = get_total(new_reps)
                            if new_peak < peak or (new_peak == peak and new_total < total):
                                A = new_A
                                reps = new_reps
                                peak = new_peak
                                total = new_total
                                update_best(A, reps)
                                improved = True
                                break
                    if improved:
                        break

                if not improved:
                    # Try removing low-contribution elements
                    if len(A) > 25:
                        low_contrib = []
                        for x in A:
                            pc, cr = element_contribution(x, A, reps)
                            if pc == 0 and cr == 0:
                                low_contrib.append(x)
                        if low_contrib:
                            x_to_remove = random.choice(low_contrib)
                            new_A = set(A)
                            new_A.remove(x_to_remove)
                            new_reps = calc_reps(new_A)
                            if is_covered(new_reps):
                                new_peak = get_peak(new_reps)
                                new_total = get_total(new_reps)
                                old_peak = get_peak(reps)
                                old_total = get_total(reps)
                                if new_peak < old_peak or (new_peak == old_peak and new_total <= old_total):
                                    A = new_A
                                    reps = new_reps
                                    update_best(A, reps)
                        else:
                            break
                    else:
                        break

    if best_A is None:
        return list(range(64, 129))

    return best_A
