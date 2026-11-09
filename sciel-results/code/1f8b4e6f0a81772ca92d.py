import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer
    
    N = 24
    M = 65536  # 2^16
    
    # Precompute FFT related things?
    # We can just call np.fft.ifft each time. It's fast enough for M=65536.
    
    def evaluate_sup_norm(c):
        # c is array of length N with +/-1
        # Pad to M
        c_pad = np.zeros(M, dtype=complex)
        c_pad[:N] = c
        # Compute values on unit circle
        # P(e^{i 2pi j/M}) = M * ifft(c_pad)_j
        vals = M * np.fft.ifft(c_pad)
        abs_vals = np.abs(vals)
        
        # Find top peaks
        # We want the max, but to be safe against grid aliasing, check top few
        # Top 5 peaks
        top_indices = np.argsort(abs_vals)[-5:]
        
        max_val = 0.0
        grid_spacing = 2 * np.pi / M
        
        for idx in top_indices:
            theta_center = 2 * np.pi * idx / M
            # Search in [theta_center - 2*grid_spacing, theta_center + 2*grid_spacing]
            a = theta_center - 2 * grid_spacing
            b = theta_center + 2 * grid_spacing
            
            # Golden section search
            # Maximize f(theta) = |P(e^{i theta})|
            def f(theta):
                # Evaluate P(e^{i theta})
                # P(z) = sum c_k z^k
                # z = exp(i theta)
                # We can compute this directly for precision
                # theta is scalar
                # c is array
                # Use np.exp(1j * theta * np.arange(N))
                # This is O(N)
                angles = np.arange(N) * theta
                z_powers = np.exp(1j * angles)
                P_val = np.dot(c, z_powers)
                return np.abs(P_val)
                
            # Golden section parameters
            gr = (np.sqrt(5) - 1) / 2
            c_pt = a + (1 - gr) * (b - a)
            d_pt = a + gr * (b - a)
            
            fc = f(c_pt)
            fd = f(d_pt)
            
            # Iterate
            for _ in range(20):  # 20 iterations is plenty for high precision
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
                    c_pt = a + (1 - gr) * (b - a)
                    fc = f(c_pt)
                    
                if b - a < 1e-10:
                    break
                    
            # Check mid
            mid = (a + b) / 2
            fm = f(mid)
            if fm > fc and fm > fd:
                fc = fm
            
            if fc > max_val:
                max_val = fc
                if fd > max_val:
                    max_val = fd
                    
        return max_val

    # Initialize with GRS
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:N], dtype=float)
    
    best_score = evaluate_sup_norm(best_coeffs)
    
    # Local Search
    def local_search(coeffs):
        current = coeffs.copy()
        current_score = evaluate_sup_norm(current)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try 1-flips
            best_single_flip = None
            best_single_score = current_score
            for k in range(N):
                new_c = current.copy()
                new_c[k] *= -1
                score = evaluate_sup_norm(new_c)
                if score < best_single_score:
                    best_single_score = score
                    best_single_flip = new_c
            
            if best_single_flip is not None:
                current = best_single_flip
                current_score = best_single_score
                improved = True
                continue
                
            # Try 2-flips if 1-flip didn't improve
            best_pair_flip = None
            best_pair_score = current_score
            for i in range(N):
                for j in range(i+1, N):
                    new_c = current.copy()
                    new_c[i] *= -1
                    new_c[j] *= -1
                    score = evaluate_sup_norm(new_c)
                    if score < best_pair_score:
                        best_pair_score = score
                        best_pair_flip = new_c
                        
            if best_pair_flip is not None:
                current = best_pair_flip
                current_score = best_pair_score
                improved = True
                continue
                
        return current, current_score

    # Iterated Local Search
    while time.time() < deadline:
        # Perturb
        num_flips = 3 + np.random.randint(0, 3) # 3-5 flips
        indices = np.random.choice(N, num_flips, replace=False)
        perturbed = best_coeffs.copy()
        perturbed[indices] *= -1
        
        # Local search from perturbed
        local_best, local_score = local_search(perturbed)
        
        if local_score < best_score:
            best_score = local_score
            best_coeffs = local_best
            
        # Check time
        if time.time() > deadline - 5:
            break
            
    return [int(x) for x in best_coeffs]
