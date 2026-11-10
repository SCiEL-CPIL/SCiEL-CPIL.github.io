import numpy as np
import time
import random
from scipy.optimize import minimize_scalar

def _precise_sup_norm(coeffs):
    """
    Calculate the precise sup norm of the polynomial on the unit circle.
    Uses dense sampling to find candidate peaks, then golden-section refinement.
    """
    c = np.array(coeffs, dtype=float)
    n = len(c)
    
    # 1. Dense sampling to find approximate peaks
    # 20000 points is a good balance between speed and accuracy for initial peak location
    M = 20000
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    z = np.exp(1j * theta)
    
    # Evaluate polynomial at all points
    # P(z) = sum c_k z^k
    # We can use np.vander or just matrix multiplication
    # V[j, k] = z[j]^k
    # To save memory/time, we can compute powers iteratively or use vander
    # For N=24, V is 20000x24, which is small.
    V = np.vander(z, N=n, increasing=True)
    p_vals = V @ c
    mags = np.abs(p_vals)
    
    # Find the global max index
    idx_max = np.argmax(mags)
    max_val = mags[idx_max]
    
    # Refine the peak around idx_max
    h = 2 * np.pi / M
    
    def objective(t):
        z_loc = np.exp(1j * (theta[idx_max] + t))
        # Compute P(z_loc)
        # Horner's method or direct sum
        # Direct sum is fine for N=24
        powers = 1.0 + 0j
        sum_val = 0.0 + 0.0j
        for k in range(n):
            sum_val += c[k] * powers
            powers *= z_loc
        return -np.abs(sum_val)
    
    # Minimize -|P| in [-h, h]
    res = minimize_scalar(objective, bounds=(-h, h), method='bounded', options={'xatol': 1e-10})
    
    if res.success:
        refined_max = -res.fun
        if refined_max > max_val:
            max_val = refined_max
            
    # Check if there are other significant peaks
    # For robustness, check the top 3 distinct peaks from the grid
    # Get top 3 indices
    # To avoid checking the same peak multiple times, we can sort and pick distinct ones
    # But for N=24, the main peak is usually dominant. 
    # Let's just refine the top 1 as before, but maybe check top 3 for safety?
    # The current program only checks top 1. Let's stick to that for speed in the final step,
    # but maybe do a second pass if we have time? 
    # Actually, the precise_sup_norm is only called a few times, so we can be thorough.
    
    # Let's refine the top 3 peaks to be safe.
    # Get indices of top 3 values
    # argsort is expensive for 20000 elements? No, it's fast.
    top_indices = np.argsort(mags)[-3:][::-1]
    
    best_refined = 0.0
    for idx in top_indices:
        if idx == idx_max:
            continue # Already refined
        
        def obj_local(t, idx=idx):
            z_loc = np.exp(1j * (theta[idx] + t))
            powers = 1.0 + 0j
            sum_val = 0.0 + 0.0j
            for k in range(n):
                sum_val += c[k] * powers
                powers *= z_loc
            return -np.abs(sum_val)
            
        res_local = minimize_scalar(obj_local, bounds=(-h, h), method='bounded', options={'xatol': 1e-10})
        if res_local.success:
            val = -res_local.fun
            if val > best_refined:
                best_refined = val
                
    if best_refined > max_val:
        max_val = best_refined
        
    return max_val

def construct():
    n = 24
    start_time = time.time()
    max_time = 115.0
    
    # 1. Fast Evaluator Setup
    # Use a smaller grid for the annealing loop to be fast
    M_fast = 1024
    theta_fast = np.linspace(0, 2 * np.pi, M_fast, endpoint=False)
    z_fast = np.exp(1j * theta_fast)
    V_fast = np.vander(z_fast, N=n, increasing=True)
    
    def fast_sup_norm(c):
        p_vals = V_fast @ c
        return np.max(np.abs(p_vals))
    
    # 2. Simulated Annealing
    # Initial random configuration
    coeffs = np.random.choice([-1, 1], size=n)
    current_energy = fast_sup_norm(coeffs)
    best_coeffs = coeffs.copy()
    best_energy = current_energy
    
    # Temperature schedule
    # Start high enough to accept bad moves, end low to converge
    T_start = 1.0
    T_end = 0.001
    # Number of iterations per temperature step
    steps_per_temp = 100
    # Number of temperature steps
    num_temps = 2000
    
    # Total iterations ~ 2,000,000. 
    # Each iteration: 1 flip, 1 matrix mult (1024x24), 1 max.
    # 1024*24 = 24576 ops. 2M * 25k = 50e9 ops? No, numpy is fast.
    # 2M iterations * ~10us/iter = 20s. This is feasible.
    
    # We will run the annealing loop until time is up, or if we finish the schedule.
    # We'll use a time-based loop to ensure we don't exceed the limit.
    
    rng = np.random.default_rng(42)
    
    # Precompute indices for flipping
    indices = np.arange(n)
    
    # Annealing Loop
    T = T_start
    cooling_rate = (T_end / T_start) ** (1.0 / num_temps)
    
    # We'll do a fixed number of iterations that fit in the time budget
    # Let's estimate: 1 million iterations should take ~5-10 seconds.
    # Let's aim for 1.5 million iterations.
    max_iters = 1500000
    
    for i in range(max_iters):
        if time.time() - start_time > max_time - 10.0:
            break
            
        # Choose a random index to flip
        k = rng.integers(n)
        
        # Flip
        old_val = coeffs[k]
        coeffs[k] = -old_val
        
        # Calculate new energy
        new_energy = fast_sup_norm(coeffs)
        delta_E = new_energy - current_energy
        
        # Acceptance criterion
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_coeffs = coeffs.copy()
        else:
            # Revert
            coeffs[k] = old_val
            
        # Cool down
        # We can cool down every few steps or continuously
        # Let's cool down every 100 steps to match the schedule
        if i % 100 == 0:
            T *= cooling_rate
    
    # 3. Final Polish and Precise Evaluation
    # Take the best candidate found by SA
    coeffs = best_coeffs
    
    # Perform a local search using the PRECISE sup norm to ensure we are at a local minimum
    # This is important because the fast norm might have a different local minimum structure.
    # We do a few passes of single-flip local search with precise evaluation.
    
    # To save time, we only do this if we have > 5 seconds left.
    if time.time() - start_time < max_time - 5.0:
        # Local search with precise norm
        # We limit the number of flips to avoid timeout
        max_local_steps = 500
        for step in range(max_local_steps):
            if time.time() - start_time > max_time - 3.0:
                break
                
            improved = False
            for k in range(n):
                old_val = coeffs[k]
                coeffs[k] = -old_val
                new_precise = _precise_sup_norm(coeffs)
                old_precise = _precise_sup_norm(best_coeffs) # This is expensive, we should cache it
                
                # Actually, calculating old_precise every time is O(1) if we store it.
                # Let's store current_precise.
                
                # Re-structure:
                # We need current_precise for best_coeffs.
                # Let's just do a simple greedy local search.
                
                # This inner loop is very slow because of _precise_sup_norm.
                # 24 flips * 500 steps * 20ms/eval = 240s. Too slow.
                # We should only do a few flips.
                
                # Let's limit to 20 steps of local search.
                # 20 * 24 * 20ms = 9.6s. Still tight.
                # Let's do 10 steps.
                # 10 * 24 * 20ms = 4.8s.
                
                # Better: Only evaluate precise norm for the best flip found in a pass?
                # No, we need to compare.
                
                # Alternative: Use fast norm for local search, then precise for final.
                # But fast norm might lead to a slightly worse precise norm.
                # Given the tight time, let's just use the SA result and do ONE precise evaluation.
                # Or, do a very small local search with fast norm, then precise.
                
                # Let's revert the previous idea and just do a fast local search.
                break
            break # Skip the expensive precise local search to be safe
            
    # Final precise evaluation
    final_score = _precise_sup_norm(best_coeffs)
    
    # Convert to list of ints
    return [int(c) for c in best_coeffs]
