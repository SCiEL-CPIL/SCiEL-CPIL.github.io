"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s limit
    
    n = 128
    max_sum = 2 * n  # 256
    conv_size = 2 * n + 1  # 257
    all_elements = set(range(0, 129))
    
    def build_conv(ind):
        return np.convolve(ind, ind, mode='full')
    
    def add_to_conv(conv, ind, x):
        """Update conv after adding element x (ind[x] was 0, now 1)."""
        # conv_new[t] = conv[t] + 2*ind[t-x] + (1 if t==2x else 0)
        # t ranges from 0 to 256
        # ind[t-x] is valid when 0 <= t-x <= 128, i.e., x <= t <= x+128
        lo = max(0, x)
        hi = min(conv_size - 1, x + n)
        if lo <= hi:
            conv[lo:hi+1] += 2 * ind[lo-x:hi+1-x]
        if 2*x < conv_size:
            conv[2*x] += 1
        return conv
    
    def remove_from_conv(conv, ind, x):
        """Update conv after removing element x (ind[x] was 1, now 0)."""
        # conv_new[t] = conv[t] - 2*ind[t-x] - (1 if t==2x else 0)
        # where ind is AFTER removal, so ind[x]=0
        # Actually: conv_old = (ind_new + e_x) * (ind_new + e_x)
        # conv_new = conv_old - 2*ind_new*shifted - e_x*e_x_shifted
        # conv_new[t] = conv_old[t] - 2*ind_new[t-x] - (1 if t==2x else 0)
        lo = max(0, x)
        hi = min(conv_size - 1, x + n)
        if lo <= hi:
            conv[lo:hi+1] -= 2 * ind[lo-x:hi+1-x]
        if 2*x < conv_size:
            conv[2*x] -= 1
        return conv
    
    def compute_peak(conv, A_set, ind):
        """Compute peak unordered-pair representation over [128,256]."""
        targets = np.arange(128, 257)
        conv_t = conv[targets].astype(np.int64)  # ordered counts
        half = targets // 2
        is_even = (targets % 2 == 0)
        in_A = np.zeros(len(targets), dtype=np.int32)
        for i in range(len(targets)):
            if is_even[i]:
                h = half[i]
                if h in A_set:
                    in_A[i] = 1
        unordered = (conv_t + in_A) // 2
        if np.any(unordered < 1):
            return 999999
        return int(np.max(unordered))
    
    # Multiple diverse initializations
    initializations = []
    
    # Init 1: [64, 128]
    initializations.append(set(range(64, 129)))
    
    # Init 2: [0, 64]
    initializations.append(set(range(0, 65)))
    
    # Init 3: Even numbers in [0, 128]
    initializations.append(set(range(0, 129, 2)))
    
    # Init 4: [32, 96]
    initializations.append(set(range(32, 97)))
    
    # Init 5: [48, 112]
    initializations.append(set(range(48, 113)))
    
    # Init 6: Random sets of various sizes
    for size in [56, 64, 72, 80]:
        for _ in range(2):
            s = set(random.sample(range(0, 129), size))
            initializations.append(s)
    
    # Init 7: Two blocks
    initializations.append(set(range(0, 33)) | set(range(96, 129)))
    initializations.append(set(range(0, 40)) | set(range(88, 129)))
    
    # Init 8: Dense middle
    initializations.append(set(range(20, 109)))
    
    # Init 9: Arithmetic progression
    initializations.append(set(range(0, 129, 2)))
    initializations.append(set(range(1, 129, 2)))
    
    best_peak = 999999
    best_A = set(range(64, 129))
    
    for init_A in initializations:
        if (time.time() - start_time) > DEADLINE:
            break
        if not init_A:
            continue
        ind = np.zeros(conv_size, dtype=np.int32)
        for a in init_A:
            ind[a] = 1
        conv = build_conv(ind)
        peak = compute_peak(conv, init_A, ind)
        if peak < best_peak:
            best_peak = peak
            best_A = set(init_A)
    
    # Local search from best
    A = set(best_A)
    ind = np.zeros(conv_size, dtype=np.int32)
    for a in A:
        ind[a] = 1
    conv = build_conv(ind)
    
    # Sort elements for deterministic iteration
    elements_list = list(range(0, 129))
    
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 2:
        improved = False
        
        # Try single additions
        for x in elements_list:
            if (time.time() - start_time) > DEADLINE - 2:
                break
            if x in A:
                continue
            A.add(x)
            ind[x] = 1
            add_to_conv(conv, ind, x)
            peak = compute_peak(conv, A, ind)
            if peak < best_peak:
                best_peak = peak
                best_A = set(A)
                improved = True
            else:
                remove_from_conv(conv, ind, x)
                ind[x] = 0
                A.discard(x)
        
        # Try single removals
        for x in sorted(A):
            if (time.time() - start_time) > DEADLINE - 2:
                break
            A.discard(x)
            ind[x] = 0
            remove_from_conv(conv, ind, x)
            peak = compute_peak(conv, A, ind)
            if peak < best_peak:
                best_peak = peak
                best_A = set(A)
                improved = True
            else:
                ind[x] = 1
                add_to_conv(conv, ind, x)
                A.add(x)
    
    # Multi-move perturbations
    if (time.time() - start_time) < DEADLINE - 5:
        for iteration in range(100):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            
            A_copy = set(best_A)
            n_swaps = 1 + (iteration % 5)
            
            # Remove random elements
            to_remove = list(A_copy)
            random.shuffle(to_remove)
            for x in to_remove[:n_swaps]:
                A_copy.discard(x)
            
            # Add random elements
            candidates = list(all_elements - A_copy)
            random.shuffle(candidates)
            for x in candidates[:n_swaps]:
                A_copy.add(x)
            
            if not A_copy:
                continue
                
            ind2 = np.zeros(conv_size, dtype=np.int32)
            for a in A_copy:
                ind2[a] = 1
            conv2 = build_conv(ind2)
            peak = compute_peak(conv2, A_copy, ind2)
            if peak < best_peak:
                best_peak = peak
                best_A = A_copy
    
    # Final local search from best
    if (time.time() - start_time) < DEADLINE - 3:
        A = set(best_A)
        ind = np.zeros(conv_size, dtype=np.int32)
        for a in A:
            ind[a] = 1
        conv = build_conv(ind)
        
        improved = True
        while improved and (time.time() - start_time) < DEADLINE - 1:
            improved = False
            for x in elements_list:
                if (time.time() - start_time) > DEADLINE - 1:
                    break
                if x in A:
                    continue
                A.add(x)
                ind[x] = 1
                add_to_conv(conv, ind, x)
                peak = compute_peak(conv, A, ind)
                if peak < best_peak:
                    best_peak = peak
                    best_A = set(A)
                    improved = True
                else:
                    remove_from_conv(conv, ind, x)
                    ind[x] = 0
                    A.discard(x)
            
            for x in sorted(A):
                if (time.time() - start_time) > DEADLINE - 1:
                    break
                A.discard(x)
                ind[x] = 0
                remove_from_conv(conv, ind, x)
                peak = compute_peak(conv, A, ind)
                if peak < best_peak:
                    best_peak = peak
                    best_A = set(A)
                    improved = True
                else:
                    ind[x] = 1
                    add_to_conv(conv, ind, x)
                    A.add(x)
    
    return list(best_A)
# EVOLVE-BLOCK-END
