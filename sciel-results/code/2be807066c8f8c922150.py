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

    # Precompute pair indices for unordered pairs (i <= j) for validation/repair if needed
    # But for the main loop, we use FFT which is much faster.
    # pairs_for_sum is kept for potential greedy repair logic if FFT logic fails (unlikely)
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

    # FFT-based representation calculation
    # A is a vector of length 129 (indices 0..128).
    # The convolution C = A * A has length 257 (indices 0..256).
    # C[s] = sum_{i+j=s} A[i]A[j].
    # For unordered pairs with equal summands allowed:
    # If i < j, the pair (i,j) contributes 2 to C[s] in convolution, but 1 to representation count.
    # If i = j, the pair (i,i) contributes 1 to C[s] in convolution, and 1 to representation count.
    # So, Rep[s] = (C[s] + Diag[s]) / 2, where Diag[s] = 1 if s is even and s/2 in A, else 0.
    
    # We need to compute Rep[s] for s in [128, 256].
    # C can be computed via np.convolve or FFT. np.convolve is O(N^2) but N=129 is small.
    # However, FFT is O(N log N). Let's use np.convolve first as it's simpler and N is small.
    # Actually, np.convolve for 129x129 is ~16k ops, which is very fast.
    # Let's benchmark mentally: 16k ops * 10000 iterations = 160M ops. In Python/NumPy, this is fine.
    # But let's use FFT to be safe for speed, or just np.convolve which is highly optimized in C.
    
    def calc_reps_fft(mask):
        # mask is int array of length 129
        # Compute full convolution
        # np.convolve returns length 257
        conv = np.convolve(mask, mask, mode='full')
        # conv[s] is the sum of mask[i]*mask[j] for i+j=s
        # We need s in [128, 256]
        conv_target = conv[128:257] # Length 129
        
        # Compute diagonal contribution
        # s is even, s/2 in [0, 128]
        # For s in [128, 256], s/2 is in [64, 128]
        # Let s_idx = 0..128 correspond to s = 128..256
        # s = 128 + s_idx
        # s/2 = 64 + s_idx/2
        # If s_idx is even, s/2 is integer.
        # Let k = s/2. k ranges from 64 to 128.
        # mask[k] is 1 if k in A.
        
        # We need an array D of length 129 such that D[s_idx] = 1 if (128+s_idx) is even and mask[(128+s_idx)//2] == 1, else 0.
        # (128+s_idx) is even iff s_idx is even.
        # Let even_mask = np.array([1 if i%2==0 else 0 for i in range(129)])
        # Then D = even_mask * mask[64:129] ? 
        # Wait.
        # s_idx = 0 -> s=128 -> k=64. mask[64].
        # s_idx = 1 -> s=129 -> odd -> 0.
        # s_idx = 2 -> s=130 -> k=65. mask[65].
        # ...
        # s_idx = 128 -> s=256 -> k=128. mask[128].
        
        # So D[i] = 1 if i is even and mask[64 + i//2] == 1.
        # Let's construct D.
        # Indices i where i is even: 0, 2, 4, ..., 128.
        # Corresponding mask indices: 64, 65, 66, ..., 128.
        # So D[0] = mask[64], D[2] = mask[65], ..., D[128] = mask[128].
        
        D = np.zeros(129, dtype=np.int32)
        # i is even: i = 2*j for j in 0..64
        # D[2*j] = mask[64+j]
        # j ranges 0..64. 64+j ranges 64..128.
        # D[0::2] = mask[64:129]
        D[0::2] = mask[64:129]
        
        reps = (conv_target + D) // 2
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

    # Try multiple initializations
    for init_round in range(10):
        mask = np.zeros(129, dtype=np.int32)

        if init_round < 5:
            # Random with forced 0 and 128
            mask[0] = 1
            mask[128] = 1
            k = random.randint(35, 55)
            candidates = list(range(1, 128))
            random.shuffle(candidates)
            for i in range(k):
                mask[candidates[i]] = 1
        else:
            # Greedy Coverage Initialization
            mask[0] = 1
            mask[128] = 1
            reps = calc_reps_fft(mask)
            uncovered = set(range(N_LOW, N_HIGH + 1))
            for s in range(N_LOW, N_HIGH + 1):
                if reps[s - N_LOW] > 0:
                    uncovered.discard(s)

            for _ in range(50):
                if len(uncovered) == 0:
                    break
                best_elem = -1
                best_coverage = 0
                for c in range(1, 128):
                    if mask[c] == 1:
                        continue
                    # Check how many uncovered sums this element would cover
                    coverage = 0
                    for s in uncovered:
                        a = s - c
                        if 0 <= a <= 128 and mask[a] == 1:
                            coverage += 1
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_elem = c
                if best_elem == -1 or best_coverage == 0:
                    best_elem = random.randint(1, 127)
                mask[best_elem] = 1
                reps = calc_reps_fft(mask)
                uncovered = set()
                for s in range(N_LOW, N_HIGH + 1):
                    if reps[s - N_LOW] == 0:
                        uncovered.add(s)

        reps = calc_reps_fft(mask)
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
    current_reps = calc_reps_fft(current_mask)
    current_obj = get_objective(current_reps)

    # Phase 2: Local Search with Vectorized Moves
    # We will perform hill climbing with random restarts.
    # Moves: Flip, Swap.
    # To speed up, we can batch evaluate flips.
    # However, 129 flips is small. We can just loop.
    
    no_improve_count = 0
    stagnation_limit = 2000
    
    while time.time() < deadline:
        if no_improve_count > stagnation_limit:
            # Restart
            mask = np.zeros(129, dtype=np.int32)
            mask[0] = 1
            mask[128] = 1
            k = random.randint(35, 55)
            candidates = list(range(1, 128))
            random.shuffle(candidates)
            for i in range(k):
                mask[candidates[i]] = 1
            current_mask = mask.copy()
            current_reps = calc_reps_fft(current_mask)
            current_obj = get_objective(current_reps)
            no_improve_count = 0
            stagnation_limit = random.randint(1000, 3000)
            continue

        improved = False
        
        # Try all single-bit flips
        for i in range(129):
            new_mask = current_mask.copy()
            new_mask[i] = 1 - new_mask[i]
            new_reps = calc_reps_fft(new_mask)
            new_obj = get_objective(new_reps)
            
            if new_obj < current_obj:
                current_mask = new_mask
                current_reps = new_reps
                current_obj = new_obj
                improved = True
                no_improve_count = 0
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_mask = current_mask.copy()
                    best_A = mask_to_list(best_mask)
                break # Restart search from new state
        
        if not improved:
            # Try swaps
            present = np.where(current_mask == 1)[0]
            absent = np.where(current_mask == 0)[0]
            
            if len(present) > 0 and len(absent) > 0:
                # To avoid O(N^2) swaps, we can sample or iterate partially
                # But N=129, so 129*129 = 16k swaps. Each swap is O(N) for calc_reps? No, calc_reps is O(N log N) or O(N^2) small.
                # np.convolve is fast. 16k * 1us = 16ms. This is fine.
                # Let's try a subset of swaps to keep it fast, or all if time permits.
                # Given 85s budget, we can afford more.
                
                # Shuffle to randomize order
                perm = np.random.permutation(len(absent))
                
                for idx in perm:
                    j = absent[idx]
                    for i in present:
                        new_mask = current_mask.copy()
                        new_mask[i] = 0
                        new_mask[j] = 1
                        new_reps = calc_reps_fft(new_mask)
                        new_obj = get_objective(new_reps)
                        
                        if new_obj < current_obj:
                            current_mask = new_mask
                            current_reps = new_reps
                            current_obj = new_obj
                            improved = True
                            no_improve_count = 0
                            if current_obj < best_obj:
                                best_obj = current_obj
                                best_mask = current_mask.copy()
                                best_A = mask_to_list(best_mask)
                            break
                    if improved:
                        break
                    if time.time() > deadline - 5:
                        break
            if not improved:
                no_improve_count += 1

    # Final validation and repair if necessary
    final_reps = calc_reps_fft(best_mask)
    if np.any(final_reps == 0):
        # Greedy repair
        missing_sums = [s for s in range(N_LOW, N_HIGH + 1) if final_reps[s - N_LOW] == 0]

        repair_mask = best_mask.copy()
        repair_reps = final_reps.copy()

        for _ in range(100):
            if not np.any(repair_reps == 0):
                break

            uncovered = [s for s in range(N_LOW, N_HIGH + 1) if repair_reps[s - N_LOW] == 0]
            if not uncovered:
                break

            s = random.choice(uncovered)
            pairs = pairs_for_sum[s]
            if len(pairs) == 0:
                continue

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
                trial_mask = repair_mask.copy()
                trial_mask[c] = 1
                trial_reps = calc_reps_fft(trial_mask)
                new_uncovered = int(np.sum(trial_reps == 0))
                if new_uncovered < best_new_uncovered:
                    best_new_uncovered = new_uncovered
                    best_add_elem = c
                    if new_uncovered == 0:
                        break

            if best_add_elem != -1:
                repair_mask[best_add_elem] = 1
                repair_reps = calc_reps_fft(repair_mask)
            else:
                if candidates:
                    best_add_elem = random.choice(list(candidates))
                    repair_mask[best_add_elem] = 1
                    repair_reps = calc_reps_fft(repair_mask)

        if not np.any(repair_reps == 0):
            best_mask = repair_mask
            best_A = mask_to_list(best_mask)
        else:
            mask = np.zeros(N_ELEMENTS, dtype=np.int32)
            for i in range(64, 129):
                mask[i] = 1
            best_A = mask_to_list(mask)

    return best_A
