import time
import random
import numpy as np
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 128
    min_sum = 128
    max_sum = 256

    def get_counts_fast(A):
        """Returns array of counts for sums in [0, 256] using numpy approach"""
        A_arr = np.array(sorted(A), dtype=np.int64)
        k = len(A_arr)
        if k == 0:
            return np.zeros(max_sum + 1, dtype=np.int64)
        # Compute all pair sums (unordered, with self-pairs)
        A_i = A_arr[:, None]
        A_j = A_arr[None, :]
        sums = A_i + A_j
        mask = np.triu(np.ones((k, k), dtype=bool), k=0)
        valid_sums = sums[mask]
        valid_sums = valid_sums[valid_sums <= max_sum]
        if len(valid_sums) > 0:
            counts = np.bincount(valid_sums, minlength=max_sum + 1)
        else:
            counts = np.zeros(max_sum + 1, dtype=np.int64)
        return counts

    def get_cost_detailed(A):
        """Returns (max_mult, uncovered_count, counts_array)"""
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            return (1000 + int(uncovered), int(uncovered), counts)
        return (int(np.max(target_counts)), 0, counts)

    def smooth_cost(counts, t=3.0):
        """
        Smooth surrogate for max multiplicity using log-sum-exp.
        Also penalizes uncovered sums heavily.
        """
        target_counts = counts[min_sum:max_sum + 1].astype(np.float64)
        # Penalize uncovered
        uncovered_penalty = np.sum(target_counts == 0) * 1000.0
        # Soft max
        shifted = target_counts / t
        max_val = np.max(shifted)
        if max_val > 500:
            return 10000.0 + uncovered_penalty
        soft_max = max_val + np.log(np.sum(np.exp(shifted - max_val)))
        return soft_max * t + uncovered_penalty

    def incremental_swap(A, x_in, x_out, current_counts):
        """Compute new counts after swapping x_in with x_out in A."""
        new_counts = current_counts.copy()
        
        # Subtract contributions involving x_in
        s = 2 * x_in
        if s <= max_sum:
            new_counts[s] -= 1
        for y in A:
            if y == x_in:
                continue
            s = x_in + y
            if s <= max_sum:
                new_counts[s] -= 1

        # Add contributions involving x_out
        s = 2 * x_out
        if s <= max_sum:
            new_counts[s] += 1
        for y in A:
            if y == x_in:
                continue
            s = x_out + y
            if s <= max_sum:
                new_counts[s] += 1
        
        return new_counts

    # Direct construction: try several structured initial sets
    best_A = None
    best_cost = float('inf')

    candidates = []

    # Candidate 1: k=14-24, spread out evenly
    for k in [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]:
        base_points = np.linspace(0, 128, k, dtype=int)
        base_set = set(int(x) for x in base_points)
        if len(base_set) < k:
            base_set = set()
            used = set()
            for i in range(k):
                x = round(i * 128 / (k - 1)) if k > 1 else 0
                while x in used:
                    x += 1
                    if x > 128:
                        x = 0
                base_set.add(x)
                used.add(x)
            base_set = set(x for x in base_set if 0 <= x <= 128)
        if len(base_set) >= 10:
            candidates.append(set(base_set))

    # Candidate: Two clusters approach (small + large)
    for a in range(8, 20):
        for b in range(110, 128):
            if b > a + 1:
                clus = set(range(0, a + 1)) | set(range(b, 129))
                if len(clus) >= 10:
                    candidates.append(clus)

    # Candidate: Mix of small and large with some middle
    for m in range(5, 15):
        small = set(range(0, m + 1))
        large = set(range(128 - m, 129))
        # Add a few middle points
        mid_count = max(1, m // 2)
        mid = set()
        for i in range(mid_count):
            mid.add(64 + (i - mid_count // 2) * 8)
        mixed = small | large | mid
        mixed = set(x for x in mixed if 0 <= x <= 128)
        if len(mixed) >= 10:
            candidates.append(mixed)

    # Candidate: Arithmetic progression with step 2 or 3
    for step in [2, 3]:
        ap_set = set(range(0, 129, step))
        candidates.append(ap_set)

    # Candidate: Symmetric set around 64
    for half in range(10, 25):
        sym_set = set(range(0, half + 1)) | set(range(128 - half, 129))
        candidates.append(sym_set)

    # Candidate: Every other point (step 2) with some adjustments
    step2 = set(range(0, 129, 2))
    candidates.append(step2)
    step2_odd = set(range(1, 129, 2))
    candidates.append(step2_odd)

    # Evaluate all candidates
    for cand in candidates:
        cand = set(x for x in cand if 0 <= x <= 128)
        if len(cand) == 0:
            continue
        cost, uncovered, _ = get_cost_detailed(cand)
        if cost < best_cost:
            best_cost = cost
            best_A = set(cand)

    # Use differential evolution on binary vector representation
    # x[i] = 1 if i is in A, 0 otherwise
    # Objective: minimize max multiplicity in [128, 256], penalize uncovered
    
    def de_objective(x):
        """
        x: array of 129 floats in [0, 1]
        Convert to binary set, compute cost
        """
        # Convert to binary: threshold at 0.5
        binary = (x > 0.5).astype(np.int64)
        
        # Ensure at least 10 elements
        if np.sum(binary) < 10:
            return 10000.0 + np.sum(binary) * 10
        
        A = set(int(i) for i in range(129) if binary[i] == 1)
        
        if len(A) == 0:
            return 10000.0
        
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        
        if uncovered > 0:
            return 1000.0 + float(uncovered) * 10.0 + float(np.max(target_counts))
        
        return float(np.max(target_counts))

    # Run differential evolution
    if time.time() < deadline - 30.0:
        try:
            # Bounds for each of the 129 positions
            bounds = [(0.0, 1.0)] * 129
            
            # Use moderate population size and max iterations
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                popsize=20,
                tol=0.01,
                seed=42,
                polish=False,  # We'll do our own polish
                init='latinhypercube',
                time_limit=deadline - time.time() - 25.0
            )
            
            if de_result.success or de_result.fun < best_cost:
                binary = (de_result.x > 0.5).astype(np.int64)
                if np.sum(binary) >= 10:
                    de_A = set(int(i) for i in range(129) if binary[i] == 1)
                    de_cost, de_uncovered, _ = get_cost_detailed(de_A)
                    if de_uncovered == 0 and de_cost < best_cost:
                        best_cost = de_cost
                        best_A = de_A
        except Exception:
            pass  # Fall through to local search

    # Now optimize with simulated annealing using smooth cost
    if best_A is not None:
        A = set(best_A)
        current_counts = get_counts_fast(A)
        current_smooth = smooth_cost(current_counts)
        current_max_mult, current_uncovered, _ = get_cost_detailed(A)

        rng = random.Random(42)
        outside = list(set(range(129)) - A)

        # Simulated annealing parameters
        T_start = 8.0
        T_end = 0.05
        cooling_rate = 0.997
        max_iterations = 120000

        T = T_start
        best_smooth = current_smooth
        best_A_sa = set(A)
        best_max_mult_sa = current_max_mult

        for iteration in range(max_iterations):
            if time.time() > deadline - 5.0:
                break

            move_type = rng.random()

            if move_type < 0.7 and len(outside) > 0 and len(A) > 0:
                # Swap
                x_in = rng.choice(list(A))
                x_out = rng.choice(outside)

                if x_in == x_out:
                    continue

                new_counts = incremental_swap(A, x_in, x_out, current_counts)
                new_smooth = smooth_cost(new_counts)
                delta = new_smooth - current_smooth

                # Metropolis acceptance
                if delta < 0 or (T > 0.01 and rng.random() < np.exp(-delta / T)):
                    # Apply move
                    A.remove(x_in)
                    A.add(x_out)
                    outside.remove(x_out)
                    outside.append(x_in)
                    current_counts = new_counts
                    current_smooth = new_smooth

                    new_max_mult, new_uncovered, _ = get_cost_detailed(A)
                    if new_uncovered == 0 and new_max_mult < best_max_mult_sa:
                        best_max_mult_sa = new_max_mult
                        best_A_sa = set(A)

            elif move_type < 0.85 and len(A) > 10:
                # Removal
                x = rng.choice(list(A))

                new_counts = current_counts.copy()

                # Subtract contributions involving x
                s = 2 * x
                if s <= max_sum:
                    new_counts[s] -= 1
                for y in A:
                    if y == x:
                        continue
                    s = x + y
                    if s <= max_sum:
                        new_counts[s] -= 1

                new_smooth = smooth_cost(new_counts)
                delta = new_smooth - current_smooth

                if delta < 0 or (T > 0.01 and rng.random() < np.exp(-delta / T)):
                    # Apply move
                    A.remove(x)
                    outside.append(x)
                    current_counts = new_counts
                    current_smooth = new_smooth

                    new_max_mult, new_uncovered, _ = get_cost_detailed(A)
                    if new_uncovered == 0 and new_max_mult < best_max_mult_sa:
                        best_max_mult_sa = new_max_mult
                        best_A_sa = set(A)

            elif move_type < 1.0 and len(A) < 40:
                # Addition
                x = rng.choice(outside)

                new_counts = current_counts.copy()

                # Add contributions involving x
                s = 2 * x
                if s <= max_sum:
                    new_counts[s] += 1
                for y in A:
                    s = x + y
                    if s <= max_sum:
                        new_counts[s] += 1

                new_smooth = smooth_cost(new_counts)
                delta = new_smooth - current_smooth

                if delta < 0 or (T > 0.01 and rng.random() < np.exp(-delta / T)):
                    # Apply move
                    A.add(x)
                    outside.remove(x)
                    current_counts = new_counts
                    current_smooth = new_smooth

                    new_max_mult, new_uncovered, _ = get_cost_detailed(A)
                    if new_uncovered == 0 and new_max_mult < best_max_mult_sa:
                        best_max_mult_sa = new_max_mult
                        best_A_sa = set(A)

            # Cool down
            T *= cooling_rate

            # Early exit if we reach a very good cost
            if best_max_mult_sa <= 2:
                break

        # Use the best result from SA
        if best_max_mult_sa < best_cost:
            best_cost = best_max_mult_sa
            best_A = best_A_sa

    # Final polish pass: targeted swaps to reduce the most congested sums
    if best_A is not None and time.time() < deadline - 3.0:
        A = set(best_A)
        current_counts = get_counts_fast(A)
        current_max_mult, _, _ = get_cost_detailed(A)

        # Try targeted swaps to reduce the most congested sums
        for polish_iter in range(2000):
            if time.time() > deadline - 2.0:
                break

            # Find the sum with highest multiplicity in target range
            target_counts = current_counts[min_sum:max_sum + 1]
            max_idx = int(np.argmax(target_counts))
            max_sum_val = min_sum + max_idx
            max_mult = int(target_counts[max_idx])

            if max_mult <= 2:
                break

            # Try to find a swap that reduces this particular sum's count
            found_improvement = False
            outside_list = list(set(range(129)) - A)
            for x_in in list(A):
                if found_improvement:
                    break
                # Check if x_in contributes to the congested sum
                partner = max_sum_val - x_in
                if 0 <= partner <= 128 and partner in A and partner != x_in:
                    # Try swapping x_in with something else
                    for x_out in outside_list:
                        new_counts = incremental_swap(A, x_in, x_out, current_counts)
                        new_target = new_counts[min_sum:max_sum + 1]
                        new_uncovered = np.sum(new_target == 0)
                        if new_uncovered > 0:
                            continue

                        new_max = int(np.max(new_target))
                        if new_max < max_mult:
                            # Apply the swap
                            A.remove(x_in)
                            A.add(x_out)
                            current_counts = new_counts
                            current_max_mult = new_max
                            if current_max_mult < best_cost:
                                best_cost = current_max_mult
                                best_A = set(A)
                            found_improvement = True
                            break

            if not found_improvement:
                break

    # Final verification and cleanup
    if best_A is None:
        # Fallback: return a valid set
        best_A = set(range(0, 129, 8))

    return list(best_A)
