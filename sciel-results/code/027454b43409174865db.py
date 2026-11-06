import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    N = 24
    half = 12
    
    # Fix c[0] = 1 (symmetry: flipping all signs doesn't change |P(z)|)
    # Enumerate 2^23 candidates for c[1:24]
    total_candidates = 1 << 23
    
    # Stage 1: Coarse screening with M1 points
    M1 = 256
    angles1 = np.linspace(0, 2 * np.pi, M1, endpoint=False)
    z1 = np.exp(1j * angles1)
    
    # Precompute Z1[k, m] = z1[m]^k for k=0..23
    Z1 = np.empty((N, M1), dtype=np.complex128)
    Z1[0, :] = 1.0
    for k in range(1, N):
        Z1[k, :] = Z1[k-1, :] * z1
    
    # base = sum of all z^k (all +1 coefficients)
    base1 = np.sum(Z1, axis=0)  # shape (M1,)
    # Z_flip[k, m] = z1[m]^(k+1) for k=0..22 (corresponding to c[1]..c[23])
    Z_flip = Z1[1:N]  # shape (23, M1)
    
    # Batch size
    B1 = 10000
    num_batches1 = (total_candidates + B1 - 1) // B1
    
    # Track top candidates from stage 1
    top_k = 200
    top_scores = np.full(top_k, np.inf)
    top_indices = np.zeros(top_k, dtype=np.int64)
    
    for b in range(num_batches1):
        if time.time() > deadline - 40.0:
            break
        
        idx_start = b * B1
        idx_end = min(idx_start + B1, total_candidates)
        B_actual = idx_end - idx_start
        
        indices = np.arange(idx_start, idx_end, dtype=np.int64)
        
        # Build flip matrix: (B_actual, 23) where entry (j,i) = bit i of indices[j]
        # bit i of index corresponds to c[i+1] being -1
        flips = np.zeros((B_actual, 23), dtype=np.float64)
        for i in range(23):
            flips[:, i] = (indices >> i) & 1
        
        # P(z) = base - 2 * sum_{i: flip_i=1} z^(i+1)
        # batch_P[j, m] = base1[m] - 2 * sum_i flips[j,i] * Z_flip[i,m]
        batch_P = base1[None, :] - 2.0 * (flips @ Z_flip)
        
        # Score = max_m |batch_P[j, m]|
        scores = np.max(np.abs(batch_P), axis=1)
        
        # Find top few in this batch
        num_to_check = min(20, B_actual)
        top_in_batch = np.argpartition(scores, num_to_check)[:num_to_check]
        
        for j in top_in_batch:
            s = scores[j]
            idx = idx_start + j
            if s < top_scores[-1]:
                # Insert into sorted position
                pos = np.searchsorted(top_scores, s)
                top_scores[pos+1:] = top_scores[pos:-1]
                top_scores[pos] = s
                top_indices[pos+1:] = top_indices[pos:-1]
                top_indices[pos] = idx
    
    if not np.isfinite(top_scores[0]):
        # Fallback: all +1
        return [1] * 24
    
    # Stage 2: Re-score top candidates on finer grid
    M2 = 4096
    angles2 = np.linspace(0, 2 * np.pi, M2, endpoint=False)
    z2 = np.exp(1j * angles2)
    
    Z2 = np.empty((N, M2), dtype=np.complex128)
    Z2[0, :] = 1.0
    for k in range(1, N):
        Z2[k, :] = Z2[k-1, :] * z2
    
    base2 = np.sum(Z2, axis=0)
    Z_flip2 = Z2[1:N]
    
    # Re-score top_k candidates
    num_to_refine = min(100, int(np.sum(np.isfinite(top_scores))))
    stage2_scores = np.zeros(num_to_refine)
    stage2_indices = np.zeros(num_to_refine, dtype=np.int64)
    
    for j in range(num_to_refine):
        if time.time() > deadline - 20.0:
            break
        idx = top_indices[j]
        # Reconstruct flip vector
        flip_vec = np.zeros(23, dtype=np.float64)
        for i in range(23):
            flip_vec[i] = (idx >> i) & 1
        
        P_vals = base2 - 2.0 * (flip_vec @ Z_flip2)
        stage2_scores[j] = np.max(np.abs(P_vals))
        stage2_indices[j] = idx
    
    # Get top 20 from stage 2
    num_top2 = min(20, num_to_refine)
    top2_order = np.argsort(stage2_scores)[:num_top2]
    top2_indices = stage2_indices[top2_order]
    top2_scores = stage2_scores[top2_order]
    
    # Stage 3: Fine scoring + golden-section refinement
    M3 = 65536
    angles3 = np.linspace(0, 2 * np.pi, M3, endpoint=False)
    z3 = np.exp(1j * angles3)
    
    Z3 = np.empty((N, M3), dtype=np.complex128)
    Z3[0, :] = 1.0
    for k in range(1, N):
        Z3[k, :] = Z3[k-1, :] * z3
    
    base3 = np.sum(Z3, axis=0)
    Z_flip3 = Z3[1:N]
    
    best_score = np.inf
    best_coeffs = np.ones(N, dtype=np.int8)
    
    num_to_fine = min(10, num_top2)
    
    for j in range(num_to_fine):
        if time.time() > deadline - 5.0:
            break
        
        idx = top2_indices[j]
        
        # Reconstruct coefficients
        coeffs = np.ones(N, dtype=np.float64)
        for i in range(23):
            if (idx >> i) & 1:
                coeffs[i + 1] = -1.0
        
        # Evaluate on fine grid
        flip_vec = np.zeros(23, dtype=np.float64)
        for i in range(23):
            flip_vec[i] = (idx >> i) & 1
        
        P_vals = base3 - 2.0 * (flip_vec @ Z_flip3)
        abs_vals = np.abs(P_vals)
        fine_score = np.max(abs_vals)
        
        if fine_score < best_score:
            best_score = fine_score
            best_coeffs = coeffs.astype(np.int8)
            
            # Golden-section refinement of the peak
            peak_idx = np.argmax(abs_vals)
            angle_center = angles3[peak_idx]
            delta_angle = 2 * np.pi / M3
            a = angle_center - delta_angle
            c = angle_center + delta_angle
            
            def P_at_angle(angle, coeffs=coeffs):
                z_val = np.exp(1j * angle)
                powers = np.ones(N, dtype=np.complex128)
                for k in range(1, N):
                    powers[k] = powers[k-1] * z_val
                return np.abs(np.sum(coeffs * powers))
            
            # Golden section search
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            b_pt = a + gr * (c - a)
            d_pt = a + (1.0 - gr) * (c - a)
            f_b = P_at_angle(b_pt)
            f_d = P_at_angle(d_pt)
            
            for _ in range(60):
                if f_b < f_d:
                    c = d_pt
                    d_pt = b_pt
                    f_d = f_b
                    b_pt = a + gr * (c - a)
                    f_b = P_at_angle(b_pt)
                else:
                    a = b_pt
                    b_pt = d_pt
                    f_b = f_d
                    d_pt = a + (1.0 - gr) * (c - a)
                    f_d = P_at_angle(d_pt)
            
            refined_score = max(f_b, f_d)
            if refined_score < best_score:
                best_score = refined_score
    
    return [int(c) for c in best_coeffs]
