import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    num_targets = max_sum - min_sum + 1  # 129

    def compute_sum_counts(A_set):
        """Compute unordered pair sum counts for all target sums."""
        counts = np.zeros(num_targets, dtype=np.int32)
        A_list = sorted(A_set)
        for i, a in enumerate(A_list):
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if min_sum <= s <= max_sum:
                    idx = s - min_sum
                    counts[idx] += 1
        return counts

    def fast_counts_from_binary(x_bin):
        """Vectorized computation of unordered pair sum counts."""
        conv = np.convolve(x_bin.astype(np.int32), x_bin.astype(np.int32), mode='full')
        counts = conv[min_sum:max_sum+1].astype(np.float64)
        diag = np.zeros(num_targets)
        for idx, s in enumerate(range(min_sum, max_sum + 1)):
            if s % 2 == 0:
                mid = s // 2
                if 0 <= mid <= n:
                    diag[idx] = x_bin[mid]
        unordered = (counts + diag) / 2.0
        return np.round(unordered).astype(np.int32)

    def evaluate(A_set):
        """Return (peak, uncovered) for a set A."""
        x_bin = np.zeros(n + 1, dtype=np.int8)
        for x in A_set:
            x_bin[x] = 1
        counts = fast_counts_from_binary(x_bin)
        uncovered = int(np.sum(counts == 0))
        if uncovered > 0:
            return float('inf'), uncovered
        peak = int(np.max(counts))
        return peak, 0

    # Phase 1: Greedy construction
    # Start with a reasonable initial set
    random.seed(42)
    np.random.seed(42)

    best_peak = float('inf')
    best_A = None

    # Try several initial seeds based on different structures
    initial_candidates = []

    # Arithmetic progression based
    for start in range(0, 40, 5):
        for step in [2, 3, 4, 5]:
            A = set()
            x = start
            while x <= n:
                A.add(x)
                x += step
            initial_candidates.append(A)

    # Two-interval sets
    for a in range(30, 80, 5):
        for b in range(60, 129, 5):
            if a < b:
                A = set(range(0, a + 1)) | set(range(b, n + 1))
                initial_candidates.append(A)

    # Random sets of various sizes
    for size in [30, 40, 50, 60, 70]:
        for _ in range(5):
            A = set(random.sample(range(0, n + 1), size))
            initial_candidates.append(A)

    # Evaluate and pick best initial
    for A_init in initial_candidates:
        peak, unc = evaluate(A_init)
        if unc == 0 and peak < best_peak:
            best_peak = peak
            best_A = A_init.copy()

    if best_A is None:
        # Fallback: use a dense set
        best_A = set(range(0, n + 1, 2))
        best_peak, _ = evaluate(best_A)

    # Phase 2: Greedy improvement - add/remove elements to reduce peak
    A_current = best_A.copy()
    current_peak, _ = evaluate(A_current)

    no_improve = 0
    max_no_improve = 200

    while time.time() < deadline - 5 and no_improve < max_no_improve:
        # Get current counts
        x_bin = np.zeros(n + 1, dtype=np.int8)
        for x in A_current:
            x_bin[x] = 1
        counts = fast_counts_from_binary(x_bin)
        peak = int(np.max(counts))

        if peak < current_peak:
            current_peak = peak
            no_improve = 0
            best_peak = peak
            best_A = A_current.copy()
        else:
            no_improve += 1

        # Try to improve by removing an element that's in high-count sums
        if len(A_current) > 20:
            # Find elements that appear in the most high-count sums
            element_score = {}
            for elem in A_current:
                score = 0
                for other in A_current:
                    if other < elem:
                        continue
                    s = elem + other
                    if min_sum <= s <= max_sum:
                        score += counts[s - min_sum]
                element_score[elem] = score

            # Remove the element with highest score (contributes most to peak)
            if element_score:
                worst_elem = max(element_score, key=element_score.get)
                A_trial = A_current.copy()
                A_trial.discard(worst_elem)
                trial_peak, trial_unc = evaluate(A_trial)
                if trial_unc == 0 and trial_peak <= peak:
                    A_current = A_trial
                    no_improve = 0
                    continue

        # Try adding an element that covers uncovered or low-count sums
        # Focus on sums that are currently at or near peak
        high_count_sums = []
        for idx in range(num_targets):
            if counts[idx] >= peak - 1:
                high_count_sums.append(min_sum + idx)

        # Try adding elements that help with coverage
        if len(A_current) < n:
            # Find which sums are poorly covered
            low_count_sums = [min_sum + idx for idx in range(num_targets) if counts[idx] <= 1]

            best_add = None
            best_add_peak = peak + 1000

            # Sample some candidate additions
            candidates = [x for x in range(n + 1) if x not in A_current]
            random.shuffle(candidates)
            for cand in candidates[:50]:
                A_trial = A_current.copy()
                A_trial.add(cand)
                trial_peak, trial_unc = evaluate(A_trial)
                if trial_unc == 0 and trial_peak < best_add_peak:
                    best_add_peak = trial_peak
                    best_add = cand

            if best_add is not None and best_add_peak < peak:
                A_current.add(best_add)
                no_improve = 0
                continue

        # Random perturbation to escape local optima
        if no_improve > 50 and len(A_current) > 20:
            # Remove a random element
            rand_elem = random.choice(list(A_current))
            A_current.discard(rand_elem)
            # Add a random element
            candidates = [x for x in range(n + 1) if x not in A_current]
            if candidates:
                A_current.add(random.choice(candidates))

    # Phase 3: Final local search
    A_final = best_A.copy()
    final_peak, _ = evaluate(A_final)

    # Try single-element swaps
    for _ in range(100):
        if time.time() > deadline - 2:
            break

        # Try removing one element
        for elem in list(A_final):
            A_trial = A_final.copy()
            A_trial.discard(elem)
            trial_peak, trial_unc = evaluate(A_trial)
            if trial_unc == 0 and trial_peak < final_peak:
                A_final = A_trial
                final_peak = trial_peak
                break

        # Try adding one element
        candidates = [x for x in range(n + 1) if x not in A_final]
        random.shuffle(candidates)
        for cand in candidates[:20]:
            A_trial = A_final.copy()
            A_trial.add(cand)
            trial_peak, trial_unc = evaluate(A_trial)
            if trial_unc == 0 and trial_peak < final_peak:
                A_final = A_trial
                final_peak = trial_peak
                break

    if len(A_final) == 0:
        A_final = best_A

    return sorted(A_final)
