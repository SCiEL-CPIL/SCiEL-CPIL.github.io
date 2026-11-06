import numpy as np
import time

def _supnorm(c):
    """
    Compute the sup norm of the polynomial with coefficients c on the unit circle.
    """
    # Dense sampling
    n_samples = 20000
    # Use complex exponential
    # theta from 0 to 2*pi
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    z = np.exp(1j * theta)
    
    # Compute powers z^k for k=0..N-1
    # z is (n_samples,), c is (N,)
    # We want P(z_i) = sum_k c_k z_i^k
    # Vectorized: 
    # Create matrix Z where Z[i, k] = z_i^k
    # This can be done by multiplying cumulative powers or using broadcasting
    # z**k is (n_samples, N) if z is (n_samples, 1) and k is (1, N)
    
    k = np.arange(len(c))
    Z = z[:, None] ** k[None, :]
    vals = Z @ c
    max_val = np.max(np.abs(vals))
    
    # Refine peaks
    # Get indices of top peaks
    abs_vals = np.abs(vals)
    # Sort indices by value
    idx = np.argsort(abs_vals)[::-1]
    
    # Refine top 10 peaks
    best_refined = max_val
    for i in idx[:10]:
        # Golden section search around theta[i]
        # Range +/- 2*pi/n_samples
        delta = 2 * np.pi / n_samples
        a = theta[i] - delta
        b = theta[i] + delta
        
        # Golden ratio
        gr = (np.sqrt(5) - 1) / 2
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        
        # Evaluate function at c_pt and d_pt
        # We need a helper to evaluate P at a single angle
        def eval_at_angle(ang):
            zz = np.exp(1j * ang)
            # z^k
            powers = zz ** k
            val = np.sum(c * powers)
            return np.abs(val)
            
        fc = eval_at_angle(c_pt)
        fd = eval_at_angle(d_pt)
        
        for _ in range(50):
            if fc < fd:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = b - gr * (b - a)
                fc = eval_at_angle(c_pt)
            else:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + gr * (b - a)
                fd = eval_at_angle(d_pt)
                
        refined_val = max(fc, fd)
        if refined_val > best_refined:
            best_refined = refined_val
            
    return best_refined

def construct():
    # Initial guess: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=float)
    
    # Convert to float for numpy ops, but keep track of +/- 1
    # Actually, we can just work with integer arrays and cast to float in _supnorm
    
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = _supnorm(best_coeffs)
    
    # Local search
    # Try flipping bits
    # To speed up, we can precompute the impact of flipping a bit?
    # Flipping bit k changes P(z) by -2 * c_k * z^k.
    # This is not additive in sup norm.
    
    # Simple hill climbing with random restarts
    # Since N=24, 2^24 is 16M.
    # We can try to do a smarter search.
    
    # Let's try a simulated annealing or just aggressive hill climbing.
    # Given the time limit, simple hill climbing with multiple restarts is safer.
    
    # We will perform hill climbing for a fixed number of steps or until time runs out.
    # Each step: pick a random bit, flip it, check score.
    # To be faster, we can check a batch of candidates? No, sup norm is max, hard to parallelize easily without overhead.
    
    # Optimization: Use a smaller sample size for the search phase, and refine only the best candidate at the end.
    
    def _fast_supnorm(c, n_samples=2000):
        theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        z = np.exp(1j * theta)
        k = np.arange(len(c))
        Z = z[:, None] ** k[None, :]
        vals = Z @ c
        return np.max(np.abs(vals))

    # Restart loop
    num_restarts = 50
    steps_per_restart = 1000
    
    # Precompute powers for fast evaluation?
    # No, z changes.
    
    # Let's try to use the fact that we only need to beat the current best.
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Start from random or best
        if r == 0:
            current = best_coeffs.copy()
        else:
            current = np.random.choice([-1, 1], size=24)
            
        current_score = _fast_supnorm(current)
        
        # If this random start is already better, update
        if current_score < best_score:
            best_score = current_score
            best_coeffs = current.copy()
        
        # Hill climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try flipping each bit
            # To avoid order bias, shuffle indices
            indices = np.arange(24)
            np.random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                # Flip bit i
                old_val = current[i]
                current[i] = -current[i]
                
                new_score = _fast_supnorm(current)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    # Update global best
                    if current_score < best_score:
                        best_score = current_score
                        best_coeffs = current.copy()
                else:
                    # Revert
                    current[i] = old_val
                    
    # Final refinement of the best found
    # Use the accurate _supnorm to get the precise score, but we return coefficients.
    # The evaluator will compute the score.
    
    # Ensure we return a list of ints
    return best_coeffs.tolist()
