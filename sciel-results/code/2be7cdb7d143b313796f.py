"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # We represent the step function as a sum of rectangular pulses.
    # Each pulse is defined by (center, width, height).
    # To keep the search space manageable and ensure non-negativity,
    # we use a fixed number of pulses and optimize their parameters.
    
    # The domain is [-1/4, 1/4].
    # We will use M pulses. Let's try a moderate number to allow flexibility.
    # Since the optimal shape for minimizing max autoconvolution is likely smooth or piecewise constant,
    # a few wide pulses might capture the essence. However, the task specifies 600 intervals.
    # The output must be a sequence of 600 heights.
    
    # Strategy:
    # 1. Define a parametric family of step functions using K rectangular pulses.
    # 2. Use differential evolution to find optimal (center, width, height) for each pulse.
    # 3. Convert the optimal parameters to the 600-bin step function.
    # 4. Optionally, do a local polish on the 600 bins using the knowledge from the pulses.
    
    # Let's try K = 10 pulses initially.
    K = 10
    
    # Parameters for each pulse:
    # center: in [-0.25, 0.25]
    # width: in [0, 0.5] (max width is the domain length)
    # height: in [0, 10] (arbitrary scale, since the objective is scale-invariant? 
    # Wait, the score is the normalized maximum of its autoconvolution.
    # Let's check the normalization. "normalized maximum" usually implies dividing by the max value 
    # or integrating. If it's just max(autoconv), then scaling h by c scales autoconv by c^2.
    # If the score is "upper bound on C1", and C1 is a constant, the problem likely involves 
    # a constraint like ||h||_1 = 1 or ||h||_2 = 1.
    # The prompt says "Choose a nonnegative step function... to minimize the normalized maximum of its autoconvolution".
    # If no constraint is given, the minimum is 0 (set all heights to 0). 
    # However, the seed returns [1.0]*N. This implies a normalization.
    # Common normalization: max(h) = 1 or sum(h) = 1.
    # Given "upper bound on C1", this sounds like a functional inequality.
    # Let's assume the normalization is such that the function is not trivially zero.
    # If I scale the function, the autoconvolution scales quadratically. 
    # If the score is "normalized maximum", it might be max(autoconv) / (max(h))^2 or similar.
    # Or perhaps the constraint is ||h||_1 = 1.
    # Let's look at the seed: [1.0]*600. 
    # If we assume the evaluator normalizes the input such that max(h)=1, then the scale of my output doesn't matter, only the shape.
    # If the evaluator computes max(autoconv) directly, I should minimize it. But I can make it arbitrarily small by scaling down.
    # Therefore, there MUST be a normalization in the evaluator.
    # Most likely: The evaluator takes my heights, normalizes them (e.g., max=1 or sum=1), computes autoconvolution, and returns the max.
    # Or, the "normalized maximum" is part of the definition of the bound C1.
    # Let's assume the evaluator normalizes the input to have max height 1.0. 
    # In that case, I can just output the shape, and the scale doesn't matter.
    # However, to be safe, I will ensure my output has max height 1.0.
    
    # Let's define the objective function for the pulse parameters.
    # Params: [c1, w1, h1, c2, w2, h2, ..., cK, wK, hK]
    # Bounds:
    # center: [-0.25, 0.25]
    # width: [0, 0.5]
    # height: [0, 1.0] (since we'll normalize to max=1 anyway, we can restrict to [0,1])
    
    bounds = []
    for _ in range(K):
        bounds.append((-0.25, 0.25))  # center
        bounds.append((0.0, 0.5))     # width
        bounds.append((0.0, 1.0))     # height
    
    def objective(params):
        if time.time() > deadline:
            return 1e10  # Penalty for timeout
        
        # Reconstruct the function on a fine grid for convolution?
        # Or directly on the 600 bins?
        # The autoconvolution of a step function on 600 bins can be computed via FFT.
        # But the pulses are continuous. We need to map them to the 600 bins.
        
        # Let's evaluate the function on the 600 bin centers or edges.
        # The bins are equal intervals on [-1/4, 1/4].
        # Bin width = (0.5) / 600 = 1/1200.
        # Bin centers: -0.25 + (i + 0.5) * (1/1200) for i in 0..599.
        
        bin_width = 0.5 / N
        centers = -0.25 + (np.arange(N) + 0.5) * bin_width
        
        h = np.zeros(N)
        for i in range(K):
            c = params[3*i]
            w = params[3*i+1]
            hgt = params[3*i+2]
            
            # Rectangular pulse from c - w/2 to c + w/2
            left = c - w/2
            right = c + w/2
            
            mask = (centers >= left) & (centers < right)
            h[mask] += hgt
        
        # Normalize h so max is 1 (if max > 0)
        max_h = np.max(h)
        if max_h > 1e-9:
            h = h / max_h
        else:
            return 1e10  # Degenerate case
        
        # Compute autoconvolution
        # Using FFT for efficiency
        n_fft = 2 * N
        h_fft = np.fft.rfft(h, n=n_fft)
        conv_fft = h_fft * np.conj(h_fft)
        conv = np.fft.irfft(conv_fft, n=n_fft)[:N]
        
        # The max of the autoconvolution
        return np.max(conv)
    
    # Run differential evolution
    # We need to be careful with the time budget.
    # DE can be slow. Let's limit the number of generations or use a smaller population.
    # popsize=10, maxiter=50 might be too slow if the objective is expensive.
    # Objective cost: N=600, FFT is O(N log N). Very fast.
    # So we can afford more iterations.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20, 
            tol=0.01, 
            polish=False,  # We'll do our own polish or just return the DE result
            init='latinhypercube'
        )
        
        best_params = result.x
        best_val = result.fun
        
        # Convert best_params to the 600-bin step function
        bin_width = 0.5 / N
        centers = -0.25 + (np.arange(N) + 0.5) * bin_width
        
        h = np.zeros(N)
        for i in range(K):
            c = best_params[3*i]
            w = best_params[3*i+1]
            hgt = best_params[3*i+2]
            
            left = c - w/2
            right = c + w/2
            
            mask = (centers >= left) & (centers < right)
            h[mask] += hgt
        
        # Normalize to max 1
        max_h = np.max(h)
        if max_h > 1e-9:
            h = h / max_h
        
        # Optional: Local polish on the 600 bins
        # We can try to adjust individual bins to reduce the max autoconvolution.
        # This is a non-smooth optimization. We can use a simple greedy or gradient-free method.
        # Given the time constraint, let's see if we have time.
        
        elapsed = time.time() - start_time
        if elapsed < 100:  # Plenty of time left
            # Simple local search: randomly perturb bins and keep if it improves
            # Or use scipy.optimize.minimize on the vector h?
            # The objective is non-differentiable (max of a smooth function).
            # We can use Nelder-Mead or Powell on the 600-dimensional space? That's too high dimensional.
            # Instead, let's try a few iterations of "smoothing" or "projection".
            # Actually, the pulse representation might be too restrictive.
            # Let's try to refine the 600 bins directly using a few steps of a simple algorithm.
            
            # We can use the fact that the optimal h should be "as flat as possible" in some sense?
            # No, we want to minimize the peak of the autoconvolution.
            
            # Let's try a simple coordinate descent on the 600 bins.
            # For each bin, try to change its height slightly.
            # This is O(N^2) per step if done naively, but we can update the convolution incrementally?
            # Too complex for now. Let's just return the pulse-based solution.
            
            # One improvement: ensure the function is non-negative (it is, by construction) 
            # and maybe clip small values to 0 to reduce noise?
            h[h < 1e-6] = 0.0
            
            # Final check: re-normalize
            max_h = np.max(h)
            if max_h > 0:
                h = h / max_h
        
        return list(h)
        
    except Exception as e:
        # Fallback to seed if DE fails
        return [1.0] * N

# EVOLVE-BLOCK-END
