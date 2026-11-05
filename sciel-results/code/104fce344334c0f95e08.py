import numpy as np
import time

def construct():
    """
    Finds the optimal 24 coefficients in {+1, -1} minimizing the score.
    Uses vectorized evaluation for fast local search and spectral synthesis for initialization.
    """
    N = 24
    start_time = time.time()
    deadline = start_time + 145.0  # 145s limit, leaving 5s buffer for safety
    
    # Grid for evaluation
    # 32768 points gives ~1e-4 angular resolution.
    # We will refine peaks to 1e-9 accuracy.
    M = 1 << 15  # 32768
    
    # Precompute the evaluation matrix E[k, j] = exp(2j * pi * k * j / M)
    # Shape: (N, M)
    # This allows P = E @ coeffs
    theta = 2 * np.pi * np.arange(M) / M
    # E[k, j] = exp(1j * k * theta[j])
    # We can use outer product of k and theta
    k_idx = np.arange(N)
    E = np.exp(1j * np.outer(k_idx, theta))
    
    # Basis for flips: basis[i] is the i-th row of E, i.e., z^i
    # Actually, E is already (N, M), so E[i] is the vector for z^i
    
    def _eval_supnorm_refined(coeffs):
        """
        Computes the sup norm of P(z) = sum c_k z^k for |z|=1.
        1. Dense sampling via matrix-vector product.
        2. Identify top peaks.
        3. Golden section refinement for high accuracy.
        """
        # 1. Dense sampling
        P_vals = E @ coeffs
        mags = np.abs(P_vals)
        
        # 2. Identify top K peaks
        K = 20
        # Get indices of top K magnitudes
        idx = np.argsort(mags)[-K:][::-1]
        
        max_val = np.max(mags)
        
        # 3. Refine each peak
        # Golden section search for max of |P(e^{i theta})|
        gr = (5 ** 0.5 - 1) / 2  # golden ratio
        
        def f(theta_val):
            z_val = np.exp(1j * theta_val)
            # Horner's method for single point
            p_val = 0.0
            for c in reversed(coeffs):
                p_val = p_val * z_val + c
            return np.abs(p_val)
        
        for i in idx:
            # Initial interval from grid spacing
            w = 2 * np.pi / M
            a = theta[i] - w
            b = theta[i] + w
            
            c_pt = b - gr * (b - a)
            d_pt = a + gr * (b - a)
            
            fc = f(c_pt)
            fd = f(d_pt)
            
            # Iterate ~30-40 times for high precision
            for _ in range(35):
                if fc < fd:
                    a = c_pt
                    c_pt = d_pt
                    fc = fd
                    d_pt = a + gr * (b - a)
                    fd = f(d_pt)
                else:
                    b = d_pt
                    d_pt = c_pt
                    fd = fc
                    c_pt = b - gr * (b - a)
                    fc = f(c_pt)
            
            peak_val = max(fc, fd)
            if peak_val > max_val:
                max_val = peak_val
                
        return max_val

    def _score_fast(coeffs):
        """
        Fast score calculation using only dense sampling (approximate).
        Used for quick comparisons in local search.
        """
        P_vals = E @ coeffs
        return np.max(np.abs(P_vals))

    def _score_accurate(coeffs):
        """
        Accurate score calculation with refinement.
        Used for final candidates.
        """
        return _eval_supnorm_refined(coeffs) / np.sqrt(N)

    # --- Initialization ---
    candidates = []
    
    # 1. Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    candidates.append(np.array(p[:N], dtype=float))
    
    # 2. Spectral Synthesis
    # Generate random phase spectra with flat magnitude, IFFT, threshold
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        # Random phases for N frequency bins
        phases = np.random.uniform(0, 2 * np.pi, N)
        spectrum = np.exp(1j * phases)
        # IFFT
        c_complex = np.fft.ifft(spectrum) * N # Scale to get order 1 magnitude
        c_real = np.real(c_complex)
        c_bin = np.sign(c_real)
        c_bin[c_bin == 0] = 1
        candidates.append(c_bin)
        
        # Also try with some magnitude variation
        mags = np.ones(N)
        # Add a small random perturbation to magnitudes
        mags *= (1 + 0.1 * np.random.randn(N))
        mags = np.abs(mags)
        spectrum2 = mags * np.exp(1j * phases)
        c_complex2 = np.fft.ifft(spectrum2) * N
        c_real2 = np.real(c_complex2)
        c_bin2 = np.sign(c_real2)
        c_bin2[c_bin2 == 0] = 1
        candidates.append(c_bin2)

    # 3. Random starts
    for _ in range(20):
        if time.time() > deadline - 30:
            break
        c_rand = np.random.choice([-1, 1], size=N).astype(float)
        candidates.append(c_rand)

    # --- Local Search ---
    # We will run local search on each candidate.
    # The local search uses vectorized flip evaluation.
    
    def local_search(initial_coeffs, time_limit):
        """
        Performs local search using single and pair flips.
        Uses vectorized evaluation for speed.
        """
        current_c = initial_coeffs.copy()
        current_P = E @ current_c
        current_val = np.max(np.abs(current_P))
        best_c = current_c.copy()
        best_val = current_val
        
        # Precompute deltas for single flips? 
        # delta_i = 2 * c_i * E[i]
        # We can compute these on the fly or update them.
        # Since c_i changes, deltas change.
        
        max_iters = 10000
        for _ in range(max_iters):
            if time.time() > time_limit:
                break
            
            # 1. Try all single flips
            best_flip_val = current_val
            best_flip_idx = -1
            
            # Vectorized check for single flips
            # For each i, P_new = current_P - 2 * current_c[i] * E[i]
            # We can loop over i (24 iterations) and compute max abs
            for i in range(N):
                delta = 2 * current_c[i] * E[i]
                P_new = current_P - delta
                val = np.max(np.abs(P_new))
                if val < best_flip_val - 1e-12:
                    best_flip_val = val
                    best_flip_idx = i
            
            if best_flip_idx >= 0:
                # Apply flip
                current_c[best_flip_idx] *= -1
                # Update P
                delta = -2 * current_c[best_flip_idx] * E[best_flip_idx] # Note: c changed, so delta sign flips relative to old
                # Easier: recompute P or update
                # P_new = P_old - 2 * c_old * E[i]
                # We already computed P_new in the loop? No, we only computed max.
                # Let's update P efficiently.
                # old_c = current_c[best_flip_idx] (before flip)
                # We flipped it, so new_c = -old_c
                # P_new = P_old - 2 * old_c * E[i]
                # But we don't have old_c stored easily if we just flipped.
                # Let's just recompute P for the flipped index? No, P is a vector.
                # It's faster to just update P using the delta we would have used.
                # Let's store the best delta vector.
                
                # Actually, let's just recompute P = E @ current_c. It's N*M ops, which is fast (24*32768 ~ 1M).
                # Or better, keep the delta.
                # Let's modify the loop to store the best delta vector.
                
                pass # Logic handled below
                
            # Let's restructure the loop to be cleaner with vectorized updates
            
            # --- Revised Local Search Step ---
            
            # Check single flips
            best_single_val = current_val
            best_single_i = -1
            best_single_P = None
            
            for i in range(N):
                # delta = 2 * c_i * E[i]
                # P_new = P - delta
                # We can compute max|P_new|
                delta_i = 2 * current_c[i] * E[i]
                P_candidate = current_P - delta_i
                val = np.max(np.abs(P_candidate))
                if val < best_single_val - 1e-12:
                    best_single_val = val
                    best_single_i = i
                    best_single_P = P_candidate
            
            if best_single_i != -1:
                current_c[best_single_i] *= -1
                current_P = best_single_P
                current_val = best_single_val
                if current_val < best_val - 1e-12:
                    best_val = current_val
                    best_c = current_c.copy()
                continue
            
            # Check pair flips
            best_pair_val = current_val
            best_pair_ij = None
            best_pair_P = None
            
            # Precompute single deltas for current state
            # deltas[i] = 2 * c_i * E[i]
            # P_new(i,j) = P - deltas[i] - deltas[j]
            
            # This is O(N^2 * M) which is 276 * 32768 ~ 9M ops. Might be slow in Python loop.
            # Let's optimize.
            
            # We can iterate i, and for each i, vectorize over j?
            # Or just loop. 276 iterations is small.
            
            for i in range(N - 1):
                delta_i = 2 * current_c[i] * E[i]
                for j in range(i + 1, N):
                    delta_j = 2 * current_c[j] * E[j]
                    P_candidate = current_P - delta_i - delta_j
                    val = np.max(np.abs(P_candidate))
                    if val < best_pair_val - 1e-12:
                        best_pair_val = val
                        best_pair_ij = (i, j)
                        best_pair_P = P_candidate
            
            if best_pair_ij is not None:
                i, j = best_pair_ij
                current_c[i] *= -1
                current_c[j] *= -1
                current_P = best_pair_P
                current_val = best_pair_val
                if current_val < best_val - 1e-12:
                    best_val = current_val
                    best_c = current_c.copy()
                continue
            
            # Local minimum reached
            # Perform a "kick" to escape
            # Flip 3 random bits
            kick_indices = np.random.choice(N, 3, replace=False)
            current_c[kick_indices] *= -1
            current_P = E @ current_c
            current_val = np.max(np.abs(current_P))
            
            # Don't update best_c here, as kick usually worsens it.
            # But if by chance it improves, next loop will catch it.
            
        return best_c, best_val

    best_overall_c = None
    best_overall_val = float('inf')
    
    # Evaluate initial candidates with fast score to pick a good starting point
    # Then run local search on the best few
    
    scored_candidates = []
    for c in candidates:
        # Fast score
        P_vals = E @ c
        val = np.max(np.abs(P_vals))
        scored_candidates.append((val, c))
    
    # Sort by fast score
    scored_candidates.sort(key=lambda x: x[0])
    
    # Run local search on the top 10 candidates
    # And some random ones to ensure diversity
    
    # Time budget per local search run
    # We have ~100s. 
    # Let's run local search for up to 5 seconds each, on 10 candidates = 50s.
    # Then use remaining time for more random restarts + local search.
    
    for val, c in scored_candidates[:10]:
        if time.time() > deadline - 20:
            break
        # Run local search
        c_best, v_best = local_search(c, time.time() + 5.0)
        
        # Verify with accurate score
        v_accurate = _score_accurate(c_best)
        
        if v_accurate < best_overall_val:
            best_overall_val = v_accurate
            best_overall_c = c_best.copy()
            
    # If we have time, do more random restarts
    while time.time() < deadline - 10:
        c_rand = np.random.choice([-1, 1], size=N).astype(float)
        # Quick local search with shorter time
        c_best, v_best = local_search(c_rand, time.time() + 2.0)
        
        # Only check accurate score if fast score is promising
        if v_best < best_overall_val * np.sqrt(N) + 1.0: # Heuristic
            v_accurate = _score_accurate(c_best)
            if v_accurate < best_overall_val:
                best_overall_val = v_accurate
                best_overall_c = c_best.copy()

    if best_overall_c is None:
        # Fallback
        return [1] * N
        
    return [int(x) for x in best_overall_c]
