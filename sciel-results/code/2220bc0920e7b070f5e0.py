import time
import random
import numpy as np
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)

    def compute_peak(A_set):
        """Compute the peak representation multiplicity for all target sums."""
        A_list = sorted(A_set)
        m = len(A_list)
        if m == 0:
            return float('inf')

        counts = np.zeros(max_sum + 1, dtype=np.int32)
        A_arr = np.array(A_list)
        for i in range(m):
            for j in range(i, m):
                s = A_arr[i] + A_arr[j]
                if min_sum <= s <= max_sum:
                    counts[s] += 1

        for s in target_sums:
            if counts[s] == 0:
                return float('inf')

        return int(np.max(counts[min_sum:max_sum + 1]))

    def peak_from_binary(x):
        """
        Compute peak from a binary vector x of length 129.
        x[i] = 1 means i is in A.
        Returns (peak, is_valid)
        """
        A_set = set(np.where(x > 0.5)[0].tolist())
        if len(A_set) == 0:
            return float('inf'), False
        peak = compute_peak(A_set)
        return peak, peak != float('inf')

    def objective_de(x):
        """
        Objective for differential evolution.
        x is a binary-ish vector of length 129.
        We want to minimize the peak, but also ensure coverage.
        If coverage fails, return a large penalty.
        """
        # Round to binary
        x_bin = (x > 0.5).astype(np.float64)
        A_set = set(np.where(x_bin > 0.5)[0].tolist())
        if len(A_set) == 0:
            return 1000.0

        # Compute counts
        A_list = sorted(A_set)
        m = len(A_list)
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        A_arr = np.array(A_list)
        for i in range(m):
            for j in range(i, m):
                s = A_arr[i] + A_arr[j]
                if min_sum <= s <= max_sum:
                    counts[s] += 1

        # Check coverage
        uncovered = 0
        for s in target_sums:
            if counts[s] == 0:
                uncovered += 1

        if uncovered > 0:
            return 1000.0 + uncovered

        peak = int(np.max(counts[min_sum:max_sum + 1]))
        # Also penalize large set size slightly to encourage sparsity?
        # Actually, we just want to minimize peak.
        return float(peak)

    # Initialize with a reasonable structure
    # Try to find a good initial guess using greedy or structured approach
    # Structure: two intervals
    # A = [0, a] U [b, 128]
    best_initial_peak = float('inf')
    best_initial_A = None

    for a in range(0, 65):
        for b in range(64, 129):
            if a >= b:
                continue
            A = set(range(0, a + 1)) | set(range(b, 129))
            peak = compute_peak(A)
            if peak < best_initial_peak:
                best_initial_peak = peak
                best_initial_A = A.copy()

    if best_initial_A is None:
        # Fallback
        best_initial_A = set(range(0, 65))
        best_initial_peak = compute_peak(best_initial_A)

    # Create initial binary vector
    x0 = np.zeros(129)
    for x in best_initial_A:
        x0[x] = 1.0

    # Define bounds for DE
    bounds = [(0, 1) for _ in range(129)]

    # Run Differential Evolution
    # We use a small population and limited iterations to fit within time
    # DE is stochastic, so we run it for a few seconds
    de_start = time.time()
    de_deadline = de_start + 30.0  # Give DE 30 seconds

    # We need a custom callback to stop DE if time is up
    # scipy's DE doesn't have a direct time-based stop, so we use maxiter and poll
    # Or we can just run it with a reasonable maxiter and popsize

    # Let's try a few different configurations
    # Configuration 1: Moderate population
    try:
        result = differential_evolution(
            objective_de,
            bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=0.01,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=False,
            init='latinhypercube'
        )
        if time.time() > de_deadline:
            # If it took too long, we might have missed the deadline, but DE usually returns
            pass

        # Convert result to binary
        x_best = (result.x > 0.5).astype(np.float64)
        A_de = set(np.where(x_best > 0.5)[0].tolist())
        if len(A_de) > 0:
            peak_de = compute_peak(A_de)
            if peak_de < best_initial_peak:
                best_initial_peak = peak_de
                best_initial_A = A_de.copy()
    except Exception as e:
        print(f"DE failed: {e}")

    # If DE didn't improve, or we have time, do local search
    if time.time() < deadline - 10:
        current_A = best_initial_A.copy()
        current_peak = best_initial_peak

        random.seed(42)
        max_iters = 50000
        iter_count = 0
        no_improve = 0
        max_no_improve = 2000

        while time.time() < deadline and iter_count < max_iters and no_improve < max_no_improve:
            iter_count += 1

            move_type = random.random()

            if move_type < 0.3:
                candidates_add = [x for x in range(0, 129) if x not in current_A]
                if not candidates_add:
                    continue
                new_elem = random.choice(candidates_add)
                new_A = current_A.copy()
                new_A.add(new_elem)

            elif move_type < 0.6:
                if len(current_A) <= 5:
                    continue
                elems = list(current_A)
                new_elem = random.choice(elems)
                new_A = current_A.copy()
                new_A.discard(new_elem)

            else:
                if len(current_A) <= 5:
                    continue
                elems = list(current_A)
                remove_elem = random.choice(elems)
                candidates_add = [x for x in range(0, 129) if x not in current_A]
                if not candidates_add:
                    continue
                add_elem = random.choice(candidates_add)
                new_A = current_A.copy()
                new_A.discard(remove_elem)
                new_A.add(add_elem)

            new_peak = compute_peak(new_A)

            if new_peak < current_peak:
                current_A = new_A
                current_peak = new_peak
                no_improve = 0
                best_initial_A = current_A.copy()
                best_initial_peak = current_peak
            else:
                no_improve += 1

    # Final check
    if best_initial_A is None or len(best_initial_A) == 0:
        return list(range(64, 129))

    return sorted(best_initial_A)
