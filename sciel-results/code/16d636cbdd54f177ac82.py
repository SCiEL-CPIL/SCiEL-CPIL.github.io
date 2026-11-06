import numpy as np
import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    n = 24
    half_n = n // 2
    size_half = 1 << half_n  # 4096
    
    # --- Precompute Z_fine once for reuse in refinement ---
    num_fine = 4096
    thetas_fine = np.linspace(0, 2 * np.pi, num_fine, endpoint=False)
    z_fine = np.exp(1j * thetas_fine)
    Z_fine = np.ones((num_fine, n), dtype=np.complex128)
    for k in range(1, n):
        Z_fine[:, k] = Z_fine[:, k-1] * z_fine
    
    # --- Stage 1: Coarse exhaustive search ---
    num_coarse = 192
    thetas_coarse = np.linspace(0, 2 * np.pi, num_coarse, endpoint=False)
    z_coarse = np.exp(1j * thetas_coarse)
    Z_coarse = np.ones((num_coarse, n), dtype=np.complex128)
    for k in range(1, n):
        Z_coarse[:, k] = Z_coarse[:, k-1] * z_coarse
    
    S = np.sum(Z_coarse, axis=1)
    
    # Precompute lookup tables
    Sum_first = np.zeros((size_half, num_coarse), dtype=np.complex128)
    Sum_last = np.zeros((size_half, num_coarse), dtype=np.complex128)
    
    for m in range(1, size_half):
        lsb = m & -m
        idx = lsb.bit_length() - 1
        prev = m ^ lsb
        Sum_first[m] = Sum_first[prev] + Z_coarse[:, idx]
    
    for m in range(1, size_half):
        lsb = m & -m
        idx = lsb.bit_length() - 1
        prev = m ^ lsb
        Sum_last[m] = Sum_last[prev] + Z_coarse[:, half_n + idx]
    
    # Exhaustive search with coarse grid - vectorized inner loop
    K1 = 20000
    best_heap = []  # (max_val, m1, m2)
    
    t0 = time.time()
    # Process in batches to manage memory and allow time checks
    batch_size = 256
    m2_batch_size = 128
    for m1_start in range(0, size_half, batch_size):
        m1_end = min(m1_start + batch_size, size_half)
        actual_batch = m1_end - m1_start
        
        # Get Sum_first for this batch of m1 values
        S1_batch = Sum_first[m1_start:m1_end]  # shape (batch, num_coarse)
        
        for m2_start in range(0, size_half, m2_batch_size):
            m2_end = min(m2_start + m2_batch_size, size_half)
            S2_batch = Sum_last[m2_start:m2_end]  # shape (m2_batch, num_coarse)
            
            # P[batch_m1, batch_m2, coarse] = S[coarse] - 2*(S1[batch_m1, coarse] + S2[batch_m2, coarse])
            # Shape: (actual_batch, m2_batch, num_coarse)
            P_full = S[None, None, :] - 2.0 * (S1_batch[:, None, :] + S2_batch[None, :, :])
            abs_P_full = np.abs(P_full)
            max_vals_2d = abs_P_full.max(axis=2)  # shape (actual_batch, m2_batch)
            
            # Flatten and find top candidates
            flat_max = max_vals_2d.ravel()
            
            k_to_consider = min(K1, flat_max.size)
            if len(best_heap) == 0:
                # First batch: push all
                for flat_i in range(flat_max.size):
                    m1_local = flat_i // m2_batch_size
                    m2_local = flat_i % m2_batch_size
                    m1 = m1_start + m1_local
                    m2 = m2_start + m2_local
                    heapq.heappush(best_heap, (flat_max[flat_i], m1, m2))
            else:
                # Find threshold: the K1-th smallest
                if k_to_consider < flat_max.size:
                    kth_val = np.partition(flat_max, k_to_consider - 1)[k_to_consider - 1]
                    mask = flat_max <= kth_val
                    candidate_indices = np.where(mask)[0]
                else:
                    candidate_indices = np.arange(flat_max.size)
                
                for flat_i in candidate_indices:
                    m1_local = flat_i // m2_batch_size
                    m2_local = flat_i % m2_batch_size
                    m1 = m1_start + m1_local
                    m2 = m2_start + m2_local
                    if flat_max[flat_i] < best_heap[0][0]:
                        heapq.heapreplace(best_heap, (flat_max[flat_i], m1, m2))
            
            # Check time periodically
            if time.time() > deadline - 40.0:
                break
        
        # Check time after each m1 batch
        if time.time() > deadline - 40.0:
            break
    
    elapsed_stage1 = time.time() - t0
    
    # --- Stage 2: Medium-resolution re-ranking ---
    num_medium = 4096
    thetas_medium = np.linspace(0, 2 * np.pi, num_medium, endpoint=False)
    z_medium = np.exp(1j * thetas_medium)
    Z_medium = np.ones((num_medium, n), dtype=np.complex128)
    for k in range(1, n):
        Z_medium[:, k] = Z_medium[:, k-1] * z_medium
    
    S_med = np.sum(Z_medium, axis=1)
    
    Sum_first_med = np.zeros((size_half, num_medium), dtype=np.complex128)
    Sum_last_med = np.zeros((size_half, num_medium), dtype=np.complex128)
    
    for m in range(1, size_half):
        lsb = m & -m
        idx = lsb.bit_length() - 1
        prev = m ^ lsb
        Sum_first_med[m] = Sum_first_med[prev] + Z_medium[:, idx]
    
    for m in range(1, size_half):
        lsb = m & -m
        idx = lsb.bit_length() - 1
        prev = m ^ lsb
        Sum_last_med[m] = Sum_last_med[prev] + Z_medium[:, half_n + idx]
    
    # Re-rank the top K1 candidates with medium grid
    candidates = sorted(best_heap, key=lambda x: x[0])
    K2 = 200
    medium_heap = []  # (max_val, m1, m2)
    
    # Vectorize medium re-ranking in batches
    med_batch_size = 128
    for med_start in range(0, len(candidates), med_batch_size):
        med_end = min(med_start + med_batch_size, len(candidates))
        batch_candidates = candidates[med_start:med_end]
        
        if time.time() > deadline - 15.0:
            break
        
        # Get S1 and S2 for this batch
        m1s = np.array([c[1] for c in batch_candidates])
        m2s = np.array([c[2] for c in batch_candidates])
        
        S1_batch_med = Sum_first_med[m1s]  # shape (batch, num_medium)
        S2_batch_med = Sum_last_med[m2s]  # shape (batch, num_medium)
        
        P_batch_med = S_med[None, :] - 2.0 * (S1_batch_med + S2_batch_med)
        max_vals_med = np.abs(P_batch_med).max(axis=1)
        
        for i, (max_val, m1, m2) in enumerate(batch_candidates):
            max_val_med = max_vals_med[i]
            if len(medium_heap) < K2:
                heapq.heappush(medium_heap, (max_val_med, m1, m2))
            elif max_val_med < medium_heap[0][0]:
                heapq.heapreplace(medium_heap, (max_val_med, m1, m2))
    
    # --- Stage 3: Golden-section refinement of top candidates ---
    def refine_peak(coeffs):
        def f(theta):
            z_point = np.exp(1j * theta)
            p_val = 0.0
            zp = 1.0 + 0.0j
            for c in coeffs:
                p_val += c * zp
                zp *= z_point
            return np.abs(p_val)
        
        # Evaluate P at all fine points using precomputed Z_fine
        vals = Z_fine @ coeffs
        abs_vals = np.abs(vals)
        idx = np.argmax(abs_vals)
        theta_max = thetas_fine[idx]
        
        # Golden section with more iterations for better accuracy
        delta = 2 * np.pi / num_fine
        a = theta_max - delta
        b = theta_max + delta
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = f(c)
        fd = f(d)
        
        for _ in range(50):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
        
        return f((a + b) / 2)
    
    # Refine top candidates
    best_val = float('inf')
    best_mask = 0
    
    medium_candidates = sorted(medium_heap, key=lambda x: x[0])
    
    for max_val_med, m1, m2 in medium_candidates:
        if time.time() > deadline - 3.0:
            break
        if max_val_med >= best_val and best_val < float('inf'):
            break
        
        mask = m1 | (m2 << half_n)
        coeffs = np.ones(n, dtype=np.float64)
        for k in range(n):
            if (mask >> k) & 1:
                coeffs[k] = -1.0
        
        refined_val = refine_peak(coeffs)
        
        if refined_val < best_val:
            best_val = refined_val
            best_mask = mask
    
    # If we have time, do a final polish with more golden-section iterations
    if time.time() < deadline - 1.0 and best_mask != 0:
        final_coeffs = np.ones(n, dtype=np.float64)
        for k in range(n):
            if (best_mask >> k) & 1:
                final_coeffs[k] = -1.0
        
        # Re-evaluate with higher precision
        def f_high(theta):
            z_point = np.exp(1j * theta)
            p_val = 0.0
            zp = 1.0 + 0.0j
            for c in final_coeffs:
                p_val += c * zp
                zp *= z_point
            return np.abs(p_val)
        
        # Find peak more precisely using dense grid
        num_dense = 16384
        thetas_dense = np.linspace(0, 2 * np.pi, num_dense, endpoint=False)
        z_dense = np.exp(1j * thetas_dense)
        Z_dense = np.ones((num_dense, n), dtype=np.complex128)
        for k in range(1, n):
            Z_dense[:, k] = Z_dense[:, k-1] * z_dense
        
        vals_dense = Z_dense @ final_coeffs
        abs_vals_dense = np.abs(vals_dense)
        idx_peak = np.argmax(abs_vals_dense)
        theta_peak = thetas_dense[idx_peak]
        
        # Golden section with many iterations
        delta = 2 * np.pi / num_dense
        a = theta_peak - delta
        b = theta_peak + delta
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = f_high(c)
        fd = f_high(d)
        
        for _ in range(80):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f_high(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f_high(d)
    
    # Return final coefficients
    final_coeffs = np.ones(n, dtype=int)
    for k in range(n):
        if (best_mask >> k) & 1:
            final_coeffs[k] = -1
            
    return final_coeffs.tolist()
