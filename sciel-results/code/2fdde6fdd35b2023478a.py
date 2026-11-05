import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))

    def compute_peak(A_set):
        """Compute peak representation multiplicity using vectorized convolution."""
        A_list = sorted(A_set)
        if not A_list:
            return float('inf')
        # Create indicator vector
        vec = np.zeros(257, dtype=np.int32)
        for a in A_list:
            vec[a] = 1
        # Autocorrelation: counts[s] = sum over a of vec[a]*vec[s-a]
        # For unordered pairs with equal summands allowed:
        # We compute full convolution, then adjust for the diagonal
        conv = np.convolve(vec, vec, mode='full')  # length 513, index s gives sum of vec[i]*vec[s-i]
        # conv[s] counts ordered pairs (a,b) with a+b=s, a,b in [0,256]
        # But we want unordered pairs from A (a <= b)
        # For a != b: ordered count = 2 * unordered count
        # For a == b: ordered count = unordered count = 1
        # So: unordered_count[s] = (conv[s] + num_diagonal[s]) / 2
        # where num_diagonal[s] = 1 if s is even and s/2 in A, else 0
        # Actually simpler: just compute directly for target range
        counts = np.zeros(257, dtype=np.int32)
        A_arr = np.array(A_list)
        m = len(A_arr)
        # Vectorized: for each i, add contributions from j >= i
        for i in range(m):
            s_vals = A_arr[i] + A_arr[i:]
            mask = (s_vals >= 128) & (s_vals <= 256)
            if np.any(mask):
                # Use bincount for efficiency
                vals = s_vals[mask]
                counts += np.bincount(vals, minlength=257)
        
        for s in target_sums:
            if counts[s] == 0:
                return float('inf')
        return int(np.max(counts[128:257]))

    def fast_peak_and_coverage(A_set):
        """Return (peak, all_covered) using efficient computation."""
        A_list = sorted(A_set)
        if not A_list:
            return float('inf'), False
        A_arr = np.array(A_list)
        m = len(A_arr)
        counts = np.zeros(257, dtype=np.int32)
        for i in range(m):
            s_vals = A_arr[i] + A_arr[i:]
            mask = (s_vals >= 128) & (s_vals <= 256)
            if np.any(mask):
                vals = s_vals[mask]
                counts += np.bincount(vals, minlength=257)
        
        target_counts = counts[128:257]
        if np.any(target_counts == 0):
            return float('inf'), False
        return int(np.max(target_counts)), True

    def local_search_deterministic(A_set, max_time=80.0):
        """
        Deterministic multi-start local search with vectorized updates.
        Uses structured moves: swap, add, remove with systematic exploration.
        """
        current_A = set(A_set)
        current_peak, covered = fast_peak_and_coverage(current_A)
        if not covered:
            # Need to fix coverage first
            return current_A, current_peak

        best_A = set(current_A)
        best_peak = current_peak
        start = time.time()
        max_time = min(max_time, deadline - start_time - 2.0)

        # Phase 1: Try all single removals
        improved = True
        while improved and (time.time() - start) < max_time * 0.3:
            improved = False
            for x in sorted(current_A):
                new_A = current_A - {x}
                if not new_A:
                    continue
                new_peak, new_covered = fast_peak_and_coverage(new_A)
                if new_covered and new_peak < current_peak:
                    current_A = new_A
                    current_peak = new_peak
                    improved = True
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(current_A)
                    break  # Restart scan after improvement

        # Phase 2: Try all single additions and swaps
        improved = True
        while improved and (time.time() - start) < max_time * 0.6:
            improved = False

            # Try removals again (after additions may have changed landscape)
            for x in sorted(current_A):
                new_A = current_A - {x}
                if not new_A:
                    continue
                new_peak, new_covered = fast_peak_and_coverage(new_A)
                if new_covered and new_peak < current_peak:
                    current_A = new_A
                    current_peak = new_peak
                    improved = True
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(current_A)
                    break

            if improved:
                continue

            # Try swaps: remove x, add y
            for x in sorted(current_A):
                if (time.time() - start) > max_time * 0.6:
                    break
                for y in range(0, 129):
                    if y in current_A:
                        continue
                    new_A = (current_A - {x}) | {y}
                    if not new_A:
                        continue
                    new_peak, new_covered = fast_peak_and_coverage(new_A)
                    if new_covered and new_peak < current_peak:
                        current_A = new_A
                        current_peak = new_peak
                        improved = True
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = set(current_A)
                        break
                if improved:
                    break

        # Phase 3: Randomized local search with hill climbing
        if (time.time() - start) < max_time * 0.9:
            rng = random.Random(42)
            no_improve = 0
            max_no_improve = 2000
            while (time.time() - start) < max_time * 0.9 and no_improve < max_no_improve:
                move_type = rng.random()
                if move_type < 0.3:
                    # Add
                    candidates = [x for x in range(129) if x not in current_A]
                    if not candidates:
                        no_improve += 1
                        continue
                    new_elem = rng.choice(candidates)
                    new_A = current_A | {new_elem}
                elif move_type < 0.6:
                    # Remove
                    if len(current_A) <= 5:
                        no_improve += 1
                        continue
                    new_A = current_A - {rng.choice(list(current_A))}
                    if not new_A:
                        no_improve += 1
                        continue
                else:
                    # Swap
                    if len(current_A) <= 5:
                        no_improve += 1
                        continue
                    x = rng.choice(list(current_A))
                    candidates = [y for y in range(129) if y not in current_A]
                    if not candidates:
                        no_improve += 1
                        continue
                    y = rng.choice(candidates)
                    new_A = (current_A - {x}) | {y}

                new_peak, new_covered = fast_peak_and_coverage(new_A)
                if new_covered and new_peak < current_peak:
                    current_A = new_A
                    current_peak = new_peak
                    no_improve = 0
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(current_A)
                else:
                    no_improve += 1

        return best_A, best_peak

    # --- Initialization: structured candidates ---
    candidates = []

    # Symmetric pair construction: A = {a, 256-a} style
    # If a in A and 256-a in A, then a + (256-a) = 256
    # More generally, for sum s, we need a and s-a both in A

    # Candidate 1: Full interval [64, 128]
    candidates.append(set(range(64, 129)))

    # Candidate 2: Two symmetric blocks
    for lo in range(32, 65, 4):
        for hi in range(96, 129, 4):
            A = set(range(lo, hi + 1))
            A = A & set(range(0, 129))
            if len(A) >= 10:
                candidates.append(A)

    # Candidate 3: Arithmetic progressions
    for start in range(0, 20):
        for step in range(1, 5):
            A = set()
            for i in range(40):
                val = start + i * step
                if 0 <= val <= 128:
                    A.add(val)
            if len(A) >= 10:
                candidates.append(A)

    # Candidate 4: Random sets of various sizes
    rng_init = random.Random(123)
    for size in [20, 30, 40, 50, 60]:
        for trial in range(5):
            A = set(rng_init.sample(range(0, 129), size))
            candidates.append(A)

    # Evaluate candidates
    best_A = None
    best_peak = float('inf')

    for A in candidates:
        peak, covered = fast_peak_and_coverage(A)
        if covered and peak < best_peak:
            best_peak = peak
            best_A = set(A)

    # If no valid candidate found, build one greedily
    if best_A is None:
        # Greedy: start with [64,128] and try to reduce
        best_A = set(range(64, 129))
        best_peak, _ = fast_peak_and_coverage(best_A)

    # Run deterministic local search
    best_A, best_peak = local_search_deterministic(best_A, max_time=70.0)

    # Final safety check
    if best_A is None or len(best_A) == 0:
        return list(range(64, 129))

    return sorted(best_A)
