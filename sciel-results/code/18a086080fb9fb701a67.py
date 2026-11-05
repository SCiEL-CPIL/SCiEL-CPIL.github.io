import time
import random
import numpy as np

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
        conv = np.rint(conv).astype(np.int64)
        
        sums = np.arange(min_sum, max_sum + 1)
        ordered_counts = conv[sums]
        
        even_mask = (sums % 2 == 0)
        half_sums = sums // 2
        diag = np.where(even_mask, 
                        np.where((half_sums >= 0) & (half_sums <= n), 
                                mask[half_sums], 0), 
                        0)
        
        counts = (ordered_counts + diag) // 2
        return counts

    def get_cost(mask):
        counts = get_counts(mask)
        if np.any(counts == 0):
            return float('inf')
        return int(np.max(counts))

    def mask_to_list(mask):
        return [i for i in range(n + 1) if mask[i]]

    def list_to_mask(A_list):
        mask = np.zeros(n + 1, dtype=np.int64)
        for x in A_list:
            mask[x] = 1
        return mask

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
    # Symmetric ranges
    for offset in range(0, 40):
        lo = max(0, 64 - offset)
        hi = min(128, 64 + offset)
        mask = np.zeros(n + 1, dtype=np.int64)
        mask[lo:hi + 1] = 1
        update_best(mask)

    # Modular
    for mod in [3, 4, 5, 6, 7, 8, 9, 10]:
        for keep in range(1, mod):
            mask = np.zeros(n + 1, dtype=np.int64)
            for x in range(n + 1):
                if x % mod != mod - keep:
                    mask[x] = 1
            update_best(mask)

    # Random initializations
    random.seed(42)
    for _ in range(80):
        if time.time() > deadline - 40:
            break
        size = random.randint(25, 65)
        A = random.sample(range(n + 1), size)
        mask = list_to_mask(A)
        update_best(mask)

    # 2. Multiplicative Weights Update (Mwu) / Iterative Reweighting
    # This is the new core strategy suggested in the directive.
    # We maintain a continuous weight vector w.
    # At each step, we compute the convolution of the thresholded mask (or a soft mask) 
    # to get counts. Then we update weights based on the "pressure" of the sums.
    
    if best_mask is not None and time.time() < deadline - 30:
        # Start from best mask
        w = best_mask.astype(np.float64)
        # Initialize weights to 1 for included, 0 for excluded, or small positive for all?
        # Let's start with the binary mask but allow continuous updates.
        # Actually, for MWU, we usually start with uniform or based on current state.
        # Let's try starting from the current best binary solution as a soft start.
        
        # We will iterate:
        # 1. Threshold w to get mask (or use soft counts if we implement soft convolution? 
        #    Standard FFT convolution requires linear inputs. 
        #    If we use soft w, the convolution gives "soft counts".
        #    Let's use soft counts directly to guide the weights.)
        
        def get_soft_counts(w):
            """Compute convolution of soft weights w with itself."""
            m = len(w)
            N = 1 << (2 * m - 1).bit_length()
            f = np.fft.rfft(w, n=N)
            conv = np.fft.irfft(f * f, n=N)[:2 * m - 1]
            return conv

        def soft_max(x, beta=50.0):
            x = np.asarray(x, dtype=np.float64)
            m = np.max(x)
            return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

        current_w = w.copy()
        
        # Precompute index mapping for sums
        sums = np.arange(min_sum, max_sum + 1)
        
        # We need to map element i to its contribution to each sum s.
        # Contribution of element i to sum s is:
        # - 1 if i < s-i and s-i is in A (i.e., j = s-i > i)
        # - 0.5 if i == s-i (i.e., s is even and i = s/2)
        # Wait, the convolution approach gives ordered counts.
        # Ordered count for sum s is sum_{i+j=s} w_i w_j.
        # Unordered count is (Ordered + Diag) / 2.
        # Diag is w_{s/2} if s even.
        
        # Let's use the gradient of the objective with respect to w_i.
        # Objective: Maximize min count? No, Minimize Max Count.
        # Let C_s be the count for sum s.
        # We want to minimize T = max_s C_s.
        # Surrogate: f(w) = sum_s exp(beta * C_s) ? Or just soft_max.
        # Let's use soft_max of C_s.
        
        # Gradient of soft_max(C) wrt w_i:
        # d/dw_i soft_max(C) = sum_s (exp(beta * C_s) / sum exp(beta * C_k)) * (d C_s / d w_i)
        
        # d C_s / d w_i:
        # C_s = (1/2) * (sum_{j} w_{s-j} w_j + (w_{s/2} if s even else 0))
        # This is getting complicated to compute analytically for every i,s.
        
        # Alternative: Finite difference or just discrete updates based on "pressure".
        # Let's stick to the discrete "add/remove" logic but guided by a global pressure map.
        
        # Compute current counts
        counts = get_counts(current_w > 0.5)
        
        # Define pressure: how much "excess" load is on each sum?
        # If max count is M, sums with count M are "hot", sums with count 0 are "cold".
        # We want to move weight from hot sums to cold sums.
        
        # For each element i in A:
        # It contributes to sums s = i + j for j in A, j >= i.
        # If we remove i, we subtract its contribution from these sums.
        # If we add i, we add its contribution.
        
        # Let's try a simple iterative improvement:
        # 1. Compute counts.
        # 2. Identify the sum s_max with maximum count.
        # 3. Identify elements i that contribute to s_max.
        # 4. Try to remove one of them if it doesn't break coverage (counts > 0).
        # 5. If no removal possible, try to add an element that contributes to a "cold" sum.
        
        # This is essentially the local search we already have, but let's make it smarter
        # by looking at the *distribution* of counts, not just the max.
        
        # Let's implement a "load balancing" local search.
        # Score function: sum of squares of counts? Or sum of max(0, count - target)?
        # Let's use sum of squares to smooth the landscape.
        
        def get_smooth_cost(mask):
            counts = get_counts(mask).astype(np.float64)
            if np.any(counts == 0):
                return 1e6 + np.sum(counts == 0)
            return np.sum(counts * counts)

        current_mask = best_mask.copy()
        current_cost = get_smooth_cost(current_mask)
        best_smooth_cost = current_cost
        best_smooth_mask = current_mask.copy()
        
        no_improve = 0
        max_no_improve = 20
        t_limit = deadline - 5
        
        while time.time() < t_limit:
            improved = False
            
            # Compute counts for current mask
            counts = get_counts(current_mask)
            if np.any(counts == 0):
                # Must fix zeros first
                zero_sums = np.where(counts == 0)[0]
                # Find elements that cover these zero sums
                # For a zero sum s, we need i, j in A such that i+j=s.
                # If we add i, it covers s if s-i is in A.
                
                # Try adding elements that cover the most zero sums
                zeros = [i for i in range(n + 1) if current_mask[i] == 0]
                random.shuffle(zeros)
                
                best_add = None
                best_gain = 0
                for i in zeros:
                    # Check how many zero sums are covered by adding i
                    # i covers sum s if s-i is in A.
                    # s is in [128, 256].
                    # j = s - i must be in A.
                    # So for each s in zero_sums, check if (s-i) is in A.
                    # We can vectorize this.
                    s_vals = zero_sums + min_sum # actual sum values
                    j_vals = s_vals - i
                    valid_j = (j_vals >= 0) & (j_vals <= n) & (current_mask[j_vals] == 1)
                    gain = np.sum(valid_j)
                    if gain > best_gain:
                        best_gain = gain
                        best_add = i
                        if gain == len(zero_sums):
                            break
                
                if best_add is not None and best_gain > 0:
                    trial_mask = current_mask.copy()
                    trial_mask[best_add] = 1
                    tc = get_smooth_cost(trial_mask)
                    if tc < current_cost:
                        current_mask = trial_mask
                        current_cost = tc
                        if tc < best_smooth_cost:
                            best_smooth_cost = tc
                            best_smooth_mask = current_mask.copy()
                        improved = True
                        no_improve = 0
                        continue
            
            # If no zeros, try to reduce max count / smooth distribution
            # Try removing elements
            ones = [i for i in range(n + 1) if current_mask[i] == 1]
            random.shuffle(ones)
            
            best_remove = None
            best_cost_remove = current_cost
            for i in ones:
                trial_mask = current_mask.copy()
                trial_mask[i] = 0
                tc = get_smooth_cost(trial_mask)
                if tc < best_cost_remove:
                    best_cost_remove = tc
                    best_remove = i
            
            if best_remove is not None:
                current_mask[best_remove] = 0
                current_cost = best_cost_remove
                if current_cost < best_smooth_cost:
                    best_smooth_cost = current_cost
                    best_smooth_mask = current_mask.copy()
                improved = True
                no_improve = 0
                continue
            
            # Try adding elements to balance
            zeros = [i for i in range(n + 1) if current_mask[i] == 0]
            random.shuffle(zeros)
            
            best_add = None
            best_cost_add = current_cost
            for i in zeros:
                trial_mask = current_mask.copy()
                trial_mask[i] = 1
                tc = get_smooth_cost(trial_mask)
                if tc < best_cost_add:
                    best_cost_add = tc
                    best_add = i
            
            if best_add is not None:
                current_mask[best_add] = 1
                current_cost = best_cost_add
                if current_cost < best_smooth_cost:
                    best_smooth_cost = current_cost
                    best_smooth_mask = current_mask.copy()
                improved = True
                no_improve = 0
                continue
            
            # Try swaps
            ones = [i for i in range(n + 1) if current_mask[i] == 1]
            zeros = [i for i in range(n + 1) if current_mask[i] == 0]
            random.shuffle(ones)
            random.shuffle(zeros)
            
            # Limit search space for swaps to speed up
            if len(ones) > 20: ones = ones[:20]
            if len(zeros) > 20: zeros = zeros[:20]
            
            best_swap = None
            best_cost_swap = current_cost
            for i in ones:
                for j in zeros:
                    trial_mask = current_mask.copy()
                    trial_mask[i] = 0
                    trial_mask[j] = 1
                    tc = get_smooth_cost(trial_mask)
                    if tc < best_cost_swap:
                        best_cost_swap = tc
                        best_swap = (i, j)
            
            if best_swap is not None:
                current_mask[best_swap[0]] = 0
                current_mask[best_swap[1]] = 1
                current_cost = best_cost_swap
                if current_cost < best_smooth_cost:
                    best_smooth_cost = current_cost
                    best_smooth_mask = current_mask.copy()
                improved = True
                no_improve = 0
                continue
            
            if not improved:
                no_improve += 1
                if no_improve >= max_no_improve:
                    # Perturbation
                    flip_count = random.randint(2, 5)
                    indices = random.sample(range(n + 1), flip_count)
                    trial_mask = current_mask.copy()
                    for idx in indices:
                        trial_mask[idx] = 1 - trial_mask[idx]
                    tc = get_smooth_cost(trial_mask)
                    if tc < current_cost:
                        current_mask = trial_mask
                        current_cost = tc
                        if tc < best_smooth_cost:
                            best_smooth_cost = tc
                            best_smooth_mask = current_mask.copy()
                        no_improve = 0
                    elif tc == current_cost and random.random() < 0.2:
                        current_mask = trial_mask
                        no_improve = 0
                    else:
                        break

        # Update global best if smooth search found something better in terms of max count
        if best_smooth_mask is not None:
            update_best(best_smooth_mask)

    # 3. Final Local Search Polish (Minimize Max Count)
    if best_mask is None:
        best_mask = np.zeros(n + 1, dtype=np.int64)
        best_mask[64:129] = 1
        best_score = get_cost(best_mask)

    current_mask = best_mask.copy()
    current_cost = best_score

    def try_local_search(mask, cost, time_limit):
        nonlocal best_score, best_mask
        current_mask = mask.copy()
        current_cost = cost
        no_improve_count = 0
        max_no_improve = 5

        while time.time() < time_limit:
            improved = False

            # 1. Try removing (order by least impact)
            ones = [i for i in range(n + 1) if current_mask[i] == 1]
            random.shuffle(ones)
            for i in ones:
                if time.time() > time_limit:
                    break
                trial_mask = current_mask.copy()
                trial_mask[i] = 0
                if not np.any(trial_mask):
                    continue
                tc = get_cost(trial_mask)
                if tc <= current_cost:
                    current_mask = trial_mask
                    current_cost = tc
                    if tc < best_score:
                        best_score = tc
                        best_mask = current_mask.copy()
                    improved = True
                    no_improve_count = 0
                    break

            # 2. Try adding
            if not improved:
                zeros = [i for i in range(n + 1) if current_mask[i] == 0]
                random.shuffle(zeros)
                for i in zeros:
                    if time.time() > time_limit:
                        break
                    trial_mask = current_mask.copy()
                    trial_mask[i] = 1
                    tc = get_cost(trial_mask)
                    if tc < current_cost:
                        current_mask = trial_mask
                        current_cost = tc
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        improved = True
                        no_improve_count = 0
                        break

            # 3. Try swapping
            if not improved:
                ones = [i for i in range(n + 1) if current_mask[i] == 1]
                zeros = [i for i in range(n + 1) if current_mask[i] == 0]
                random.shuffle(ones)
                random.shuffle(zeros)
                if len(ones) > 30:
                    ones = ones[:30]
                if len(zeros) > 30:
                    zeros = zeros[:30]

                for i in ones:
                    if time.time() > time_limit:
                        break
                    for j in zeros:
                        if time.time() > time_limit:
                            break
                        trial_mask = current_mask.copy()
                        trial_mask[i] = 0
                        trial_mask[j] = 1
                        tc = get_cost(trial_mask)
                        if tc < current_cost:
                            current_mask = trial_mask
                            current_cost = tc
                            if tc < best_score:
                                best_score = tc
                                best_mask = current_mask.copy()
                            improved = True
                            no_improve_count = 0
                            break
                    if improved:
                        break

            # 4. Random perturbation
            if not improved and time.time() < time_limit - 2:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    flip_count = random.randint(2, 4)
                else:
                    flip_count = random.randint(1, 3)
                indices = random.sample(range(n + 1), flip_count)
                trial_mask = current_mask.copy()
                for idx in indices:
                    trial_mask[idx] = 1 - trial_mask[idx]
                if np.any(trial_mask):
                    tc = get_cost(trial_mask)
                    if tc < current_cost:
                        current_mask = trial_mask
                        current_cost = tc
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        improved = True
                        no_improve_count = 0
                    elif tc == current_cost and random.random() < 0.3:
                        current_mask = trial_mask
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        improved = True
                        no_improve_count = 0

            if not improved and no_improve_count >= max_no_improve:
                break

    try_local_search(best_mask, best_score, deadline - 2)

    # Final check
    if best_mask is None or not np.any(best_mask):
        return list(range(64, 129))

    # Final polish: try to remove any redundant elements
    final_mask = best_mask.copy()
    ones = [i for i in range(n + 1) if final_mask[i] == 1]
    random.shuffle(ones)
    for i in ones:
        if time.time() > deadline - 0.5:
            break
        trial_mask = final_mask.copy()
        trial_mask[i] = 0
        if not np.any(trial_mask):
            continue
        tc = get_cost(trial_mask)
        if tc <= best_score:
            final_mask = trial_mask
            if tc < best_score:
                best_score = tc
                best_mask = final_mask.copy()

    return mask_to_list(final_mask)
