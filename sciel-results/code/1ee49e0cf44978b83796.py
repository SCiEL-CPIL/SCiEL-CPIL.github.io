import numpy as np
import time

def construct():
    """
    Finds a polynomial P(z) = sum c_k z^k with c_k in {+1, -1}
    that minimizes the normalized sup-norm.
    """
    N = 24
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Evaluation function: returns max |P(e^{i theta})|
    def calc_max(c):
        # Dense sampling
        M = 2048
        thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)
        z = np.exp(1j * thetas)
        # Compute P(z) for all z
        # c is a list of 24 coefficients
        # P(z) = c_0 + c_1 z + ... + c_23 z^23
        # We can use np.polynomial or just matrix multiplication
        # z^k for k=0..23
        # z_matrix shape (M, 24)
        # z_matrix[:, k] = z^k
        z_powers = np.vander(z, N, increasing=True)
        P_vals = z_powers @ c
        max_idx = np.argmax(np.abs(P_vals))
        
        # Refine the peak around max_idx
        # The peak is between thetas[max_idx-1] and thetas[max_idx+1]
        # Actually, the grid is uniform. Let's refine in the interval [thetas[max_idx-1], thetas[max_idx+1]]
        # But the max could be exactly at a grid point or between.
        # Golden section search on |P(e^{i theta})| is not straightforward because it's not unimodal in a simple interval?
        # Actually, |P(e^{i theta})| is a sum of cosines/sines, so it is smooth.
        # We can do a local minimization/maximization.
        
        # Let's just do a fine grid around the peak for simplicity and speed, or use scipy.optimize if available.
        # SciPy is available.
        from scipy.optimize import minimize_scalar
        
        def f_neg(theta):
            z = np.exp(1j * theta)
            # Compute P(z) for single theta
            # This is slower than vectorized, but we only do it a few times.
            # Actually, let's vectorize the refinement too if we do a small grid.
            # Or just use the vectorized eval for a small range.
            return -np.abs(np.sum(c * np.power(z, np.arange(N))))
            
        # Interval for refinement
        # The step size is 2*pi/M
        step = 2 * np.pi / M
        # The peak is likely in [thetas[max_idx] - step, thetas[max_idx] + step]
        # But we need to be careful with boundaries.
        # Let's just refine in a small window around thetas[max_idx]
        # A better way: use the fact that the max is near thetas[max_idx].
        # Let's use minimize_scalar on the negative magnitude.
        # Bounds: [thetas[max_idx] - step, thetas[max_idx] + step]
        # But if max_idx is 0, we wrap around.
        
        # Simpler approach for refinement:
        # Evaluate on a finer grid around the peak.
        # Let's use 100 points in the interval of width 2*step.
        center = thetas[max_idx]
        # Handle wrap-around
        # If max_idx == 0, the interval is [0, step] and [2*pi-step, 2*pi]
        # Let's just evaluate on a global fine grid if the coarse max is high?
        # No, that defeats the purpose.
        
        # Let's use scipy's minimize_scalar with method='bounded'
        # We need to handle the periodicity.
        # Let's just try a few points around the peak with high precision.
        
        # Actually, for N=24, the function is very smooth.
        # A simple golden section search on the interval [center - step, center + step]
        # might fail if the peak is exactly at the boundary or if the function is flat.
        # Let's use a robust method: evaluate on 1000 points in [center - step, center + step]
        # and take the max.
        
        # To avoid wrap-around complexity in the code, let's just use a global fine grid for the final evaluation
        # of the best candidates, but for the search, use the coarse grid max.
        # The coarse grid max is a good proxy.
        
        # Let's return the coarse max for speed during search.
        # We will do a final high-precision check on the best candidate.
        
        return np.max(np.abs(P_vals))

    def calc_max_precise(c):
        # High precision evaluation for the final candidate
        M = 10000
        thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)
        z = np.exp(1j * thetas)
        z_powers = np.vander(z, N, increasing=True)
        P_vals = z_powers @ c
        max_idx = np.argmax(np.abs(P_vals))
        
        # Refine around max_idx
        step = 2 * np.pi / M
        center = thetas[max_idx]
        # Refine in [center - step, center + step]
        # Use 1000 points
        ref_thetas = np.linspace(center - step, center + step, 1000)
        ref_z = np.exp(1j * ref_thetas)
        ref_z_powers = np.vander(ref_z, N, increasing=True)
        ref_P_vals = ref_z_powers @ c
        return np.max(np.abs(ref_P_vals))

    best_coeffs = None
    best_score = float('inf')
    
    # Initial candidates
    # 1. GRS
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    grs_coeffs = p[:N]
    
    candidates = [grs_coeffs]
    
    # 2. Random
    for _ in range(5):
        candidates.append(np.random.choice([-1, 1], N).tolist())
        
    # 3. All 1s
    candidates.append([1]*N)
    # 4. Alternating
    candidates.append([1 if i%2==0 else -1 for i in range(N)])
    
    for start_c in candidates:
        if time.time() > deadline:
            break
            
        c = start_c.copy()
        # Hill Climbing
        improved = True
        while improved:
            improved = False
            current_max = calc_max(c)
            for i in range(N):
                if time.time() > deadline:
                    break
                # Flip coefficient i
                old_val = c[i]
                c[i] = -old_val
                new_max = calc_max(c)
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
             if best_coeffs:
                 perturbed = best_coeffs.copy()
                 for _ in range(3):
                     idx = np.random.randint(N)
                     perturbed[idx] = -perturbed[idx]
                 candidates.append(perturbed)
             else:
                 candidates.append(np.random.choice([-1, 1], N).tolist())
            
    if best_coeffs is None:
        return grs_coeffs
        
    return best_coeffs
