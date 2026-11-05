import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 128
    min_sum = 128
    max_sum = 256
    S = max_sum - min_sum + 1  # 129 sums

    def get_counts(mask):
        """Fast count of unordered pair sums (a<=b) in [128,256] using FFT."""
        m = len(mask)
        N = 1 << (2 * m - 1).bit_length()
        f = np.fft.rfft(mask.astype(np.float64), n=N)
        conv = np.fft.irfft(f * f, n=N)[:2 * m - 1]
        conv = np.rint(conv)
        counts = np.zeros(S, dtype=np.int64)
        for i, s in enumerate(range(min_sum, max_sum + 1)):
            ordered = int(conv[s])
            if s % 2 == 0:
                half = s // 2
                diag = 1 if (0 <= half <= n and mask[half]) else 0
            else:
                diag = 0
            counts[i] = (ordered + diag) // 2
        return counts

    def mask_to_list(mask):
        return [i for i in range(n + 1) if mask[i]]

    def list_to_mask(A_list):
        mask = np.zeros(n + 1, dtype=np.int64)
        for x in A_list:
            mask[x] = 1
        return mask

    def get_cost(mask):
        counts = get_counts(mask)
        if np.any(counts == 0):
            return float('inf')
        return int(np.max(counts))

    best_score = float('inf')
    best_mask = None

    def update_best(mask):
        nonlocal best_score, best_mask
        if not np.any(mask):
            return
        cost = get_cost(mask)
        if cost < best_score:
            best_score = cost
            best_mask = mask.copy()

    # 1. Heuristic Initializations
    for offset in range(0, 40):
        lo = max(0, 64 - offset)
        hi = min(128, 64 + offset)
        mask = np.zeros(n + 1, dtype=np.int64)
        mask[lo:hi + 1] = 1
        update_best(mask)

    for mod in [3, 4, 5, 6, 7, 8, 9, 10]:
        for keep in range(1, mod):
            mask = np.zeros(n + 1, dtype=np.int64)
            for x in range(n + 1):
                if x % mod != mod - keep:
                    mask[x] = 1
            update_best(mask)

    random.seed(42)
    for _ in range(80):
        if time.time() > deadline - 40:
            break
        size = random.randint(25, 65)
        A = random.sample(range(n + 1), size)
        mask = list_to_mask(A)
        update_best(mask)

    # 3. Local Search Refinement with Simulated Annealing and Incremental Updates
    if best_mask is None:
        best_mask = np.zeros(n + 1, dtype=np.int64)
        best_mask[64:129] = 1
        best_score = get_cost(best_mask)

    current_mask = best_mask.copy()
    current_counts = get_counts(current_mask)
    current_cost = best_score

    def try_local_search(mask, cost, time_limit):
        nonlocal best_score, best_mask
        current_mask = mask.copy()
        current_counts = get_counts(current_mask)
        current_cost = cost
        
        # Simulated Annealing parameters
        T = 5.0
        cooling = 0.95
        min_T = 0.1

        def energy(counts):
            if np.any(counts == 0):
                return float('inf')
            # Use a smooth surrogate to guide moves: log-sum-exp of counts
            # This penalizes high counts more smoothly than just max
            c = counts.astype(np.float64)
            m = np.max(c)
            if m > 100: # Avoid overflow
                return m + np.log(np.sum(np.exp(c - m)))
            return m + np.log(np.sum(np.exp(c)))

        current_energy = energy(current_counts)

        while time.time() < time_limit and T > min_T:
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.4: # Remove
                ones = [i for i in range(n + 1) if current_mask[i] == 1]
                if not ones: 
                    T *= cooling
                    continue
                i = random.choice(ones)
                trial_mask = current_mask.copy()
                trial_mask[i] = 0
                if not np.any(trial_mask): 
                    T *= cooling
                    continue
                
                # Incremental update for counts
                trial_counts = current_counts.copy()
                # Remove contributions of i
                for j in range(n + 1):
                    if current_mask[j] == 0: continue
                    s = i + j
                    if s < min_sum or s > max_sum: continue
                    idx = s - min_sum
                    trial_counts[idx] -= 1

                tc = int(np.max(trial_counts)) if not np.any(trial_counts == 0) else float('inf')
                te = energy(trial_counts)
                
                delta_e = te - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    current_mask = trial_mask
                    current_counts = trial_counts
                    current_cost = tc
                    current_energy = te
                    if tc < best_score:
                        best_score = tc
                        best_mask = current_mask.copy()

            elif move_type < 0.8: # Add
                zeros = [i for i in range(n + 1) if current_mask[i] == 0]
                if not zeros: 
                    T *= cooling
                    continue
                i = random.choice(zeros)
                trial_mask = current_mask.copy()
                trial_mask[i] = 1
                
                trial_counts = current_counts.copy()
                for j in range(n + 1):
                    if current_mask[j] == 0: continue
                    s = i + j
                    if s < min_sum or s > max_sum: continue
                    idx = s - min_sum
                    trial_counts[idx] += 1

                tc = int(np.max(trial_counts)) if not np.any(trial_counts == 0) else float('inf')
                te = energy(trial_counts)
                
                delta_e = te - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    current_mask = trial_mask
                    current_counts = trial_counts
                    current_cost = tc
                    current_energy = te
                    if tc < best_score:
                        best_score = tc
                        best_mask = current_mask.copy()
            else: # Swap
                ones = [i for i in range(n + 1) if current_mask[i] == 1]
                zeros = [i for i in range(n + 1) if current_mask[i] == 0]
                if not ones or not zeros: 
                    T *= cooling
                    continue
                i = random.choice(ones)
                j = random.choice(zeros)
                
                trial_mask = current_mask.copy()
                trial_mask[i] = 0
                trial_mask[j] = 1
                
                trial_counts = current_counts.copy()
                # Remove i
                for k in range(n + 1):
                    if current_mask[k] == 0: continue
                    s = i + k
                    if s < min_sum or s > max_sum: continue
                    idx = s - min_sum
                    trial_counts[idx] -= 1
                # Add j
                for k in range(n + 1):
                    if current_mask[k] == 0: continue
                    if k == i: continue # i is removed
                    s = j + k
                    if s < min_sum or s > max_sum: continue
                    idx = s - min_sum
                    trial_counts[idx] += 1

                tc = int(np.max(trial_counts)) if not np.any(trial_counts == 0) else float('inf')
                te = energy(trial_counts)
                
                delta_e = te - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    current_mask = trial_mask
                    current_counts = trial_counts
                    current_cost = tc
                    current_energy = te
                    if tc < best_score:
                        best_score = tc
                        best_mask = current_mask.copy()

            T *= cooling

    try_local_search(best_mask, best_score, deadline - 3)

    # Final check
    if best_mask is None or not np.any(best_mask):
        return list(range(64, 129))

    return mask_to_list(best_mask)
