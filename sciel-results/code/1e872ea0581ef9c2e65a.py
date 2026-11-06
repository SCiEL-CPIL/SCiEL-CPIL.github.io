import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    N = 24  # number of coefficients
    
    # Stage 1: Very fast coarse screening
    M1 = 128
    angles1 = np.linspace(0, 2 * np.pi, M1, endpoint=False)
    z1 = np.exp(1j * angles1)
    
    # Precompute Z1[k, m] = z1[m]^k
    Z1 = np.zeros((N, M1), dtype=complex)
    Z1[0, :] = 1.0
    for k in range(1, N):
        Z1[k, :] = Z1[k-1, :] * z1
    
    base1 = np.sum(Z1, axis=0)  # shape (M1,)
    Z1_flip = Z1[1:24]  # shape (23, M1)
    
    # Large batch size for speed
    B1 = 50000
    total_candidates = 1 << 23
    num_batches1 = (total_candidates + B1 - 1) // B1
    
    # Track top candidates
    top_k = 200
    top_scores = np.full(top_k, np.inf)
    top_indices = np.zeros(top_k, dtype=np.int64)
    
    for b in range(num_batches1):
        if time.time() > deadline - 50.0:
            break
        
        idx_start = b * B1
        idx_end = min(idx_start + B1, total_candidates)
        B_actual = idx_end - idx_start
        
        indices = np.arange(idx_start, idx_end, dtype=np.int64)
        
        # Build flip matrix efficiently
        flips = np.zeros((B_actual, 23), dtype=np.float64)
        for i in range(23):
            flips[:, i] = (indices >> i) & 1
        
        # Compute batch_P = base1[None,:] - 2 * (flips @ Z1_flip)
        batch_P = base1[None, :] - 2.0 * (flips @ Z1_flip)
        
        # Get max |P(z)| for each candidate
        scores = np.max(np.abs(batch_P), axis=1)
        
        # Find top candidates in this batch
        num_to_check = min(10, B_actual)
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
    
    # Stage 2: Fine screening for top candidates
    M2 = 4096
    angles2 = np.linspace(0, 2 * np.pi, M2, endpoint=False)
    z2 = np.exp(1j * angles2)
    
    Z2 = np.zeros((N, M2), dtype=complex)
    Z2[0, :] = 1.0
    for k in range(1, N):
        Z2[k, :] = Z2[k-1, :] * z2
    
    base2 = np.sum(Z2, axis=0)
    Z2_flip = Z2[1:24]
    
    # Evaluate top candidates on fine grid
    best_score2 = np.inf
    best_idx2 = top_indices[0]
    
    # Only check the top 50
    num_to_check2 = min(50, top_k)
    for i in range(num_to_check2):
        if time.time() > deadline - 40.0:
            break
        
        idx = top_indices[i]
        
        # Reconstruct flip mask
        flip_vec = np.zeros(23, dtype=np.float64)
        for bit in range(23):
            flip_vec[bit] = (idx >> bit) & 1
        
        # Compute P(z) on fine grid
        P_fine = base2 - 2.0 * (flip_vec @ Z2_flip)
        score = np.max(np.abs(P_fine))
        
        if score < best_score2:
            best_score2 = score
            best_idx2 = idx
    
    # Stage 3: Local refinement
    # Reconstruct best coefficients
    best_coeffs = np.ones(N)
    for bit in range(23):
        if (best_idx2 >> bit) & 1:
            best_coeffs[bit + 1] = -1.0
    
    # Use a very fine grid to locate the peak
    M3 = 32768
    angles3 = np.linspace(0, 2 * np.pi, M3, endpoint=False)
    z3 = np.exp(1j * angles3)
    
    Z3 = np.zeros((N, M3), dtype=complex)
    Z3[0, :] = 1.0
    for k in range(1, N):
        Z3[k, :] = Z3[k-1, :] * z3
    
    vals3 = best_coeffs @ Z3
    abs_vals3 = np.abs(vals3)
    peak_idx = np.argmax(abs_vals3)
    
    angle_center = angles3[peak_idx]
    delta_angle = 2 * np.pi / M3
    a = angle_center - delta_angle
    c = angle_center + delta_angle
    
    def P_at_angle(angle):
        z_val = np.exp(1j * angle)
        powers = np.ones(N, dtype=complex)
        for k in range(1, N):
            powers[k] = powers[k-1] * z_val
        return np.abs(np.sum(best_coeffs * powers))
    
    # Golden section search
    gr = (np.sqrt(5) - 1) / 2
    b_pt = a + gr * (c - a)
    d_pt = a + (1 - gr) * (c - a)
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
            d_pt = a + (1 - gr) * (c - a)
            f_d = P_at_angle(d_pt)
    
    # Return the coefficients
    return [int(c) for c in best_coeffs]
