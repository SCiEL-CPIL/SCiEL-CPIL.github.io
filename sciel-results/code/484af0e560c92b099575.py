import numpy as np
import time

def construct():
    """
    Finds a polynomial P(z) = sum c_k z^k with c_k in {+1, -1}
    that minimizes the normalized sup-norm.
    """
    N = 24
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Precompute powers of z for the coarse grid
    M = 4096  # Increased resolution for better initial peak identification
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)
    # Z[k] is a vector of length M containing z^k for all z in the grid
    # Z[k][m] = exp(1j * thetas[m] * k)
    Z = np.exp(1j * np.outer(thetas, np.arange(N)))
    
    def calc_max_fast(c):
        """
        Returns the max |P(z)| on the coarse grid.
        c is a numpy array of shape (N,) with values +/- 1.
        """
        # P_vals = Z.T @ c ? No.
        # Z is (M, N). Z[m, k] = z_m^k.
        # P_m = sum_k c_k Z[m, k].
        # So P = Z @ c.
        P_vals = Z @ c
        return np.max(np.abs(P_vals))

    def calc_max_precise(c):
        """
        High precision evaluation.
        1. Find top K peaks on coarse grid.
        2. Refine each peak with a fine local grid.
        """
        P_vals = Z @ c
        mags = np.abs(P_vals)
        
        # Find indices of top K peaks
        # We want to avoid adjacent indices in the coarse grid being picked as separate peaks
        # if they are part of the same broad peak.
        # Simple approach: take the global max, then zero out a window around it, repeat.
        K = 5
        peaks = []
        mags_copy = mags.copy()
        for _ in range(K):
            max_idx = np.argmax(mags_copy)
            if mags_copy[max_idx] < 1e-6:
                break
            peaks.append(max_idx)
            # Zero out a window around max_idx to prevent picking adjacent points
            # Window size: maybe 10 points?
            start = (max_idx - 10) % M
            end = (max_idx + 10) % M
            # Need to handle wrap-around for zeroing
            # Simpler: just set a small range
            for i in range(-10, 11):
                mags_copy[(max_idx + i) % M] = 0
                
        max_val = 0.0
        step = 2 * np.pi / M
        
        for idx in peaks:
            center = thetas[idx]
            # Refine in [center - step, center + step]
            # Use 200 points for high precision
            ref_thetas = np.linspace(center - step, center + step, 200)
            ref_z = np.exp(1j * ref_thetas)
            # Compute P on ref grid
            # ref_Z[k] = ref_z^k
            # We can compute this efficiently
            # P_ref = sum c_k ref_z^k
            # Let's just use a small Vandermonde or direct sum
            # Direct sum is fast for N=24, M_ref=200
            P_ref = np.zeros(len(ref_z), dtype=complex)
            for k in range(N):
                P_ref += c[k] * (ref_z ** k)
            
            local_max = np.max(np.abs(P_ref))
            if local_max > max_val:
                max_val = local_max
                
        return max_val

    best_coeffs = None
    best_score = float('inf')
    
    # Initial candidates
    # 1. GRS (Golomb-Rudin Sequence)
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    grs_coeffs = np.array(p[:N], dtype=np.int8)
    
    candidates = [grs_coeffs]
    
    # 2. Random starts
    for _ in range(10):
        candidates.append(np.random.choice([-1, 1], N).astype(np.int8))
        
    # 3. Alternating
    candidates.append(np.array([1 if i%2==0 else -1 for i in range(N)], dtype=np.int8))
    
    for start_c in candidates:
        if time.time() > deadline:
            break
            
        c = start_c.copy()
        # Hill Climbing
        improved = True
        current_max = calc_max_fast(c)
        
        while improved:
            improved = False
            # Shuffle indices to avoid bias? Or just iterate.
            # Iterating is fine.
            for i in range(N):
                if time.time() > deadline:
                    break
                # Flip coefficient i
                old_val = c[i]
                c[i] = -old_val
                new_max = calc_max_fast(c)
                if new_max < current_max - 1e-12:
                    current_max = new_max
                    improved = True
                else:
                    c[i] = old_val
            
        # Evaluate the local optimum precisely
        final_max = calc_max_precise(c)
        if final_max < best_score:
            best_score = final_max
            best_coeffs = c.copy()
            
        # Generate new random candidates if time permits
        if time.time() < deadline - 5:
             # Add some random perturbations of the best found so far
             if best_coeffs is not None:
                 perturbed = best_coeffs.copy()
                 for _ in range(3):
                     idx = np.random.randint(N)
                     perturbed[idx] = -perturbed[idx]
                 candidates.append(perturbed)
             else:
                 candidates.append(np.random.choice([-1, 1], N).astype(np.int8))
            
    if best_coeffs is None:
        return grs_coeffs.tolist()
        
    return best_coeffs.tolist()
