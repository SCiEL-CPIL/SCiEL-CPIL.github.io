"""Additive bases with few representations – optimized search."""
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin before 90s kill

    n = 128
    target_lo, target_hi = 128, 256
    n_targets = target_hi - target_lo + 1  # 129

    # Precompute all pair sums for quick reference
    all_pairs = []
    for i in range(129):
        for j in range(i, 129):
            s = i + j
            if target_lo <= s <= target_hi:
                all_pairs.append((i, j, s - target_lo))
    all_pairs = np.array(all_pairs, dtype=np.int64)
    pair_i = all_pairs[:, 0]
    pair_j = all_pairs[:, 1]
    pair_s = all_pairs[:, 2]

    # Precompute for each element x in [0,128] the list of (other, target_idx) pairs
    # that x participates in. This allows fast delta-based count updates.
    # For each x, store: list of (y, t) where y >= x, x+y in [128,256], t = x+y-128
    elem_pairs = [[] for _ in range(129)]
    for idx in range(len(pair_i)):
        i_val = int(pair_i[idx])
        j_val = int(pair_j[idx])
        t_val = int(pair_s[idx])
        elem_pairs[i_val].append((j_val, t_val))
        if i_val != j_val:
            elem_pairs[j_val].append((i_val, t_val))

    def compute_counts_fast(A_set):
        """Compute multiplicity using numpy boolean indexing on precomputed pairs."""
        in_A = np.zeros(129, dtype=bool)
        for x in A_set:
            in_A[x] = True
        mask = in_A[pair_i] & in_A[pair_j]
        counts = np.bincount(pair_s[mask], minlength=n_targets)
        return counts

    def compute_counts_from_indicator(indicator):
        """Compute counts from a boolean indicator array."""
        mask = indicator[pair_i] & indicator[pair_j]
        counts = np.bincount(pair_s[mask], minlength=n_targets)
        return counts

    def get_indicator(A):
        """Get boolean indicator array for set A."""
        ind = np.zeros(129, dtype=bool)
        for x in A:
            ind[x] = True
        return ind

    def objective_exact(A):
        """Exact objective: peak + 100*uncovered."""
        ind = get_indicator(A)
        counts = compute_counts_from_indicator(ind)
        uncovered = int(np.sum(counts == 0))
        peak = int(counts.max())
        return peak + 100 * uncovered, counts

    def objective_smooth(A, beta=3.0):
        """Smooth objective using soft-max."""
        ind = get_indicator(A)
        counts = compute_counts_from_indicator(ind)
        c = counts.astype(np.float64)
        uncovered = int(np.sum(counts == 0))
        c_max = c.max()
        if c_max == 0:
            peak_smooth = 0.0
        else:
            shifted = c - c_max
            peak_smooth = c_max + np.log(np.sum(np.exp(beta * shifted)) / len(c)) / beta
        return peak_smooth + 100.0 * uncovered

    def delta_objective(indicator, counts, add_x, remove_x=None):
        """Compute objective delta when adding add_x and optionally removing remove_x.
        Returns (new_peak, new_uncovered) estimated via count updates."""
        new_counts = counts.copy()
        
        if remove_x is not None:
            # Remove contributions from pairs involving remove_x
            for (y, t) in elem_pairs[remove_x]:
                if indicator[y]:
                    new_counts[t] -= 1
            indicator[remove_x] = False
        
        # Add contributions from pairs involving add_x
        for (y, t) in elem_pairs[add_x]:
            if indicator[y]:
                new_counts[t] += 1
        indicator[add_x] = True
        
        uncovered = int(np.sum(new_counts == 0))
        peak = int(new_counts.max())
        return peak + 100 * uncovered

    def smart_init(k):
        """Generate a structured initial set with good coverage properties."""
        base = list(range(0, 129, max(1, 129 // k)))
        random.shuffle(base)
        A = sorted(base[:min(k, len(base))])
        remaining = k - len(A)
        if remaining > 0:
            candidates = [x for x in range(129) if x not in A]
            if candidates:
                A.extend(random.sample(candidates, min(remaining, len(candidates))))
        return sorted(set(A))

    def repair(A):
        """Add elements to A to ensure full coverage of [128, 256]."""
        A = sorted(set(A))
        ind = get_indicator(A)
        counts = compute_counts_from_indicator(ind)
        for _ in range(200):
            uncovered = np.where(counts == 0)[0]
            if len(uncovered) == 0:
                break
            target = random.choice(list(uncovered)) + target_lo
            best_candidate = None
            best_gain = 0
            old_uncovered = int(np.sum(counts == 0))
            for _ in range(40):
                candidate = random.randint(0, 128)
                if not ind[candidate]:
                    # Count how many uncovered targets this would fix
                    gain = 0
                    for (y, t) in elem_pairs[candidate]:
                        if ind[y] and counts[t] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = candidate
            if best_candidate is not None:
                A.append(best_candidate)
                ind[best_candidate] = True
                for (y, t) in elem_pairs[best_candidate]:
                    if ind[y]:
                        counts[t] += 1
            else:
                # Try all candidates
                for candidate in range(129):
                    if not ind[candidate]:
                        gain = 0
                        for (y, t) in elem_pairs[candidate]:
                            if ind[y] and counts[t] == 0:
                                gain += 1
                        if gain > 0:
                            A.append(candidate)
                            ind[candidate] = True
                            for (y, t) in elem_pairs[candidate]:
                                if ind[y]:
                                    counts[t] += 1
                            break
        return A

    def local_search(A, max_iters=500, use_smooth=False):
        """Improve A with additions, removals, and pair swaps."""
        A = sorted(set(A))
        ind = get_indicator(A)
        counts = compute_counts_from_indicator(ind)
        
        if use_smooth:
            c = counts.astype(np.float64)
            c_max = c.max()
            if c_max == 0:
                best_obj = 0.0
            else:
                shifted = c - c_max
                best_obj = c_max + np.log(np.sum(np.exp(3.0 * shifted)) / len(c)) / 3.0
            best_obj += 100.0 * int(np.sum(counts == 0))
        else:
            best_obj = int(counts.max()) + 100 * int(np.sum(counts == 0))
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1

            # Try additions
            if not improved:
                for _ in range(30):
                    candidate = random.randint(0, 128)
                    if not ind[candidate]:
                        # Compute delta
                        new_counts = counts.copy()
                        for (y, t) in elem_pairs[candidate]:
                            if ind[y]:
                                new_counts[t] += 1
                        if use_smooth:
                            c = new_counts.astype(np.float64)
                            c_max = c.max()
                            if c_max == 0:
                                new_obj = 0.0
                            else:
                                shifted = c - c_max
                                new_obj = c_max + np.log(np.sum(np.exp(3.0 * shifted)) / len(c)) / 3.0
                            new_obj += 100.0 * int(np.sum(new_counts == 0))
                        else:
                            new_obj = int(new_counts.max()) + 100 * int(np.sum(new_counts == 0))
                        if new_obj < best_obj:
                            A.append(candidate)
                            ind[candidate] = True
                            counts = new_counts
                            best_obj = new_obj
                            improved = True
                            break

            # Try removals
            if not improved and len(A) > 12:
                for _ in range(20):
                    idx = random.randint(0, len(A) - 1)
                    x = A[idx]
                    new_counts = counts.copy()
                    for (y, t) in elem_pairs[x]:
                        if ind[y]:
                            new_counts[t] -= 1
                    if use_smooth:
                        c = new_counts.astype(np.float64)
                        c_max = c.max()
                        if c_max == 0:
                            new_obj = 0.0
                        else:
                            shifted = c - c_max
                            new_obj = c_max + np.log(np.sum(np.exp(3.0 * shifted)) / len(c)) / 3.0
                        new_obj += 100.0 * int(np.sum(new_counts == 0))
                    else:
                        new_obj = int(new_counts.max()) + 100 * int(np.sum(new_counts == 0))
                    if new_obj < best_obj:
                        A.pop(idx)
                        ind[x] = False
                        counts = new_counts
                        best_obj = new_obj
                        improved = True
                        break

            # Try pair swaps
            if not improved and len(A) >= 2:
                for _ in range(30):
                    idx = random.randint(0, len(A) - 1)
                    x = A[idx]
                    candidate = random.randint(0, 128)
                    if not ind[candidate]:
                        new_counts = counts.copy()
                        # Remove x's contributions
                        for (y, t) in elem_pairs[x]:
                            if y != candidate and ind[y]:
                                new_counts[t] -= 1
                        # Add candidate's contributions
                        for (y, t) in elem_pairs[candidate]:
                            if ind[y]:
                                new_counts[t] += 1
                        if use_smooth:
                            c = new_counts.astype(np.float64)
                            c_max = c.max()
                            if c_max == 0:
                                new_obj = 0.0
                            else:
                                shifted = c - c_max
                                new_obj = c_max + np.log(np.sum(np.exp(3.0 * shifted)) / len(c)) / 3.0
                            new_obj += 100.0 * int(np.sum(new_counts == 0))
                        else:
                            new_obj = int(new_counts.max()) + 100 * int(np.sum(new_counts == 0))
                        if new_obj < best_obj:
                            A[idx] = candidate
                            ind[x] = False
                            ind[candidate] = True
                            counts = new_counts
                            best_obj = new_obj
                            improved = True
                            break

            # Try double swaps
            if not improved and len(A) >= 4:
                for _ in range(15):
                    idx1, idx2 = random.sample(range(len(A)), 2)
                    x1, x2 = A[idx1], A[idx2]
                    c1 = random.randint(0, 128)
                    c2 = random.randint(0, 128)
                    if c1 != c2 and not ind[c1] and not ind[c2]:
                        new_counts = counts.copy()
                        # Remove x1 and x2
                        for (y, t) in elem_pairs[x1]:
                            if ind[y] and y != x2:
                                new_counts[t] -= 1
                        for (y, t) in elem_pairs[x2]:
                            if ind[y] and y != x1:
                                new_counts[t] -= 1
                        # Add c1 and c2
                        for (y, t) in elem_pairs[c1]:
                            if ind[y]:
                                new_counts[t] += 1
                        for (y, t) in elem_pairs[c2]:
                            if ind[y]:
                                new_counts[t] += 1
                        if use_smooth:
                            c = new_counts.astype(np.float64)
                            c_max = c.max()
                            if c_max == 0:
                                new_obj = 0.0
                            else:
                                shifted = c - c_max
                                new_obj = c_max + np.log(np.sum(np.exp(3.0 * shifted)) / len(c)) / 3.0
                            new_obj += 100.0 * int(np.sum(new_counts == 0))
                        else:
                            new_obj = int(new_counts.max()) + 100 * int(np.sum(new_counts == 0))
                        if new_obj < best_obj:
                            A[idx1] = c1
                            A[idx2] = c2
                            ind[x1] = False
                            ind[x2] = False
                            ind[c1] = True
                            ind[c2] = True
                            counts = new_counts
                            best_obj = new_obj
                            improved = True
                            break

        return sorted(set(A))

    def simulated_annealing(A, time_limit):
        """SA using smooth objective for faster evaluation."""
        A = sorted(set(A))
        ind = get_indicator(A)
        counts = compute_counts_from_indicator(ind)
        
        c = counts.astype(np.float64)
        c_max = c.max()
        if c_max == 0:
            best_obj = 0.0
        else:
            shifted = c - c_max
            best_obj = c_max + np.log(np.sum(np.exp(3.0 * shifted)) / len(c)) / 3.0
        best_obj += 100.0 * int(np.sum(counts == 0))
        
        best_A = A[:]
        current_obj = best_obj
        T = 12.0
        cooling = 0.999
        min_T = 0.05

        while time.time() < time_limit and T > min_T:
            move_type = random.random()
            new_A = None
            
            if move_type < 0.25 and len(A) > 12:
                idx = random.randint(0, len(A) - 1)
                new_A = A[:idx] + A[idx+1:]
            elif move_type < 0.55:
                candidate = random.randint(0, 128)
                if candidate not in A:
                    new_A = A + [candidate]
            elif move_type < 0.85:
                if len(A) >= 1:
                    idx = random.randint(0, len(A) - 1)
                    candidate = random.randint(0, 128)
                    if candidate not in A:
                        new_A = A[:idx] + [candidate] + A[idx+1:]
                        new_A = sorted(new_A)
            else:
                if len(A) >= 4:
                    idx1, idx2 = random.sample(range(len(A)), 2)
                    c1 = random.randint(0, 128)
                    c2 = random.randint(0, 128)
                    if c1 not in A and c2 not in A and c1 != c2:
                        new_A = [x for i, x in enumerate(A) if i not in (idx1, idx2)]
                        new_A.extend([c1, c2])
                        new_A = sorted(new_A)

            if new_A is None:
                T *= cooling
                continue

            new_ind = get_indicator(new_A)
            new_counts = compute_counts_from_indicator(new_ind)
            c = new_counts.astype(np.float64)
            c_max = c.max()
            if c_max == 0:
                new_obj = 0.0
            else:
                shifted = c - c_max
                new_obj = c_max + np.log(np.sum(np.exp(3.0 * shifted)) / len(c)) / 3.0
            new_obj += 100.0 * int(np.sum(new_counts == 0))
            
            delta = new_obj - current_obj

            if delta <= 0 or random.random() < np.exp(-delta / T):
                A = new_A
                ind = new_ind
                counts = new_counts
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_A = A[:]

            T *= cooling

        return best_A, best_obj

    def final_polish(A):
        """High-precision polish: iteratively try all single-element swaps to reduce peak."""
        A = sorted(set(A))
        ind = get_indicator(A)
        counts = compute_counts_from_indicator(ind)
        peak = int(counts.max())
        
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try removing each element and see if peak decreases
            for idx in range(len(A)):
                x = A[idx]
                new_counts = counts.copy()
                for (y, t) in elem_pairs[x]:
                    if ind[y]:
                        new_counts[t] -= 1
                new_peak = int(new_counts.max())
                new_uncovered = int(np.sum(new_counts == 0))
                if new_uncovered == 0 and new_peak < peak:
                    A.pop(idx)
                    ind[x] = False
                    counts = new_counts
                    peak = new_peak
                    improved = True
                    break
            if improved:
                continue
            
            # Try swapping each element with each candidate
            for idx in range(len(A)):
                x = A[idx]
                for candidate in range(129):
                    if ind[candidate]:
                        continue
                    new_counts = counts.copy()
                    for (y, t) in elem_pairs[x]:
                        if ind[y] and y != candidate:
                            new_counts[t] -= 1
                    for (y, t) in elem_pairs[candidate]:
                        if ind[y]:
                            new_counts[t] += 1
                    new_peak = int(new_counts.max())
                    new_uncovered = int(np.sum(new_counts == 0))
                    if new_uncovered == 0 and new_peak < peak:
                        A[idx] = candidate
                        ind[x] = False
                        ind[candidate] = True
                        counts = new_counts
                        peak = new_peak
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return sorted(set(A)), peak

    best_A = None
    best_score = float('inf')

    # Try different sizes and restarts
    for k in range(16, 34):
        if time.time() > deadline - 25:
            break

        for restart in range(3):
            if time.time() > deadline - 20:
                break

            A = smart_init(k)
            A = repair(A)
            if not A:
                continue

            # Check validity
            ind = get_indicator(A)
            counts = compute_counts_from_indicator(ind)
            if not np.all(counts > 0):
                continue

            A = local_search(A, max_iters=300, use_smooth=True)
            ind = get_indicator(A)
            counts = compute_counts_from_indicator(ind)
            if not np.all(counts > 0):
                continue
            obj = int(counts.max()) + 100 * int(np.sum(counts == 0))

            sa_time = min(6, deadline - time.time() - 10)
            if sa_time > 2:
                A, sa_obj = simulated_annealing(A, time.time() + sa_time)
                ind = get_indicator(A)
                counts = compute_counts_from_indicator(ind)
                if not np.all(counts > 0):
                    continue
                obj = int(counts.max()) + 100 * int(np.sum(counts == 0))

            if obj < best_score:
                best_score = obj
                best_A = A[:]

    # Final polish with exact objective
    if best_A is not None and time.time() < deadline - 10:
        best_A, final_peak = final_polish(best_A)
        if final_peak < best_score:
            best_score = final_peak
            best_A = sorted(best_A)

    # One more polish pass if time allows
    if best_A is not None and time.time() < deadline - 5:
        best_A, final_peak = final_polish(best_A)
        if final_peak < best_score:
            best_score = final_peak
            best_A = sorted(best_A)

    if best_A is None or not best_A:
        return list(range(64, 129))

    # Final validity check
    ind = get_indicator(best_A)
    counts = compute_counts_from_indicator(ind)
    if not np.all(counts > 0):
        return list(range(64, 129))

    return sorted(best_A)
