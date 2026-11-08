import numpy as np
import time

N = 400
DEADLINE = 140.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 is defined as the maximum absolute value of the autoconvolution.
    """
    # Use FFT to compute the autoconvolution efficiently
    # Pad to at least 2*N to avoid circular convolution effects
    size = 1 << (2 * N - 1).bit_length()  # Next power of 2 >= 2*N
    
    # Build the signal on a fine grid using vectorized operations
    # Each interval has width h = 1/N
    # We represent the function on a grid of 'size' points
    # We can do this by repeating each value size//N times
    # To be safe and exact, we can use np.repeat if size is divisible by N, 
    # but size is a power of 2 and N=400, so it's not divisible.
    # The original code used a loop. Let's try to vectorize or use a better method.
    # Actually, for N=400, size is 1024. 1024/400 is not integer.
    # The original code:
    # start = i * (size // N)
    # end = (i + 1) * (size // N)
    # This effectively truncates the last bit of the interval for some i?
    # size // N = 1024 // 400 = 2.
    # So each step is 2 samples wide. 400 * 2 = 800 samples used out of 1024.
    # The rest are zeros? 
    # f = np.zeros(size)
    # for i in range(N):
    #    start = i * 2
    #    end = (i+1) * 2
    #    f[start:end] = signed_step[i]
    # This puts the function in the first 800 samples, leaving 224 zeros at the end.
    # This is a valid representation for linear convolution via FFT if we treat the zeros as padding.
    
    # Let's replicate the exact logic of the seed to ensure the metric is identical.
    f = np.zeros(size)
    # Vectorized assignment is tricky with non-divisible sizes if we want to match the loop exactly.
    # However, the loop does:
    # f[i*(size//N) : (i+1)*(size//N)] = signed_step[i]
    # Since size//N is constant (2), we can do:
    step_size = size // N
    # Create indices
    # idx = np.arange(N) * step_size
    # This would be 0, 2, 4, ...
    # We can just do a loop if it's fast enough, or use repeat and slice.
    # np.repeat(signed_step, step_size) gives length N*step_size = 800.
    f[:N*step_size] = np.repeat(signed_step, step_size)
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real
    
    # Scale by the width of each grid cell
    dx = 0.5 / size
    autoconv *= dx
    
    # The autoconvolution is defined on [-0.5, 0.5]
    # C3 = max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3

def construct():
    start_time = time.time()
    
    # Strategy: Continuous Gradient Descent + Discrete Local Search
    
    # 1. Continuous Relaxation Phase
    # We treat the step heights as continuous variables x_i in [-1, 1].
    # The objective function is C3(x). We want to minimize it.
    # Since C3 involves a max over absolute values, it is convex (max of convex functions).
    # We can use a simple subgradient or just use the fact that for small steps,
    # we can approximate the gradient.
    
    # However, a simpler and robust approach for "signed step" problems where the domain is discrete
    # but the landscape is smooth-ish:
    # Start with random continuous values in [-1, 1].
    # Perform gradient descent to minimize C3.
    # Since C3 is max |autocorr|, the gradient with respect to x_i is related to the autocorrelation
    # at the points where the max is achieved.
    
    # Let's implement a simple gradient descent step.
    # C3(x) = max_k | (f*f)(t_k) | where f is built from x.
    # Let y = f*f. C3 = max |y_k|.
    # If the max is at k* and y[k*] > 0, grad C3 wrt x is proportional to grad y[k*] wrt x.
    # y = ifft(fft(f) * conj(fft(f))).
    # This is complex to differentiate directly in Python without autograd.
    
    # Alternative Structural Change:
    # Instead of gradient descent on the continuous relaxation (which might be slow or tricky to implement correctly for the max-abs),
    # let's use a different search method: **Genetic Algorithm (GA)** or **Particle Swarm Optimization (PSO)**?
    # Or **Simulated Annealing with larger jumps**?
    
    # Let's look at the inspirations. Inspiration 3 mentions "models the problem as ...".
    # A very effective strategy for binary optimization with smooth underlying structure is **Adaptive Simulated Annealing** 
    # or **Iterated Local Search (ILS)** with a strong perturbation.
    
    # Let's try **Iterated Local Search (ILS)**.
    # 1. Start with a random solution.
    # 2. Perform local search (hill climbing) until a local minimum is found.
    # 3. Perturb the solution (e.g., flip a random block of bits).
    # 4. Repeat local search.
    # 5. Keep the best solution found.
    
    # Why ILS?
    # SA (current) uses a cooling schedule which might cool down too fast or too slow.
    # ILS focuses on finding local minima quickly and then escaping them via controlled perturbations.
    # This is often more effective for rugged landscapes.
    
    # Implementation Details:
    # Local Search: Flip bits that reduce C3. Repeat until no improvement.
    # Perturbation: Flip a random window of bits (e.g., size 10-50) to jump to a new basin.
    
    best_step = np.random.choice([-1, 1], size=N).astype(np.int8)
    best_c3 = _compute_c3(best_step)
    
    current_step = best_step.copy()
    current_c3 = best_c3
    
    def local_search(step, c3, time_limit):
        """
        Hill climbing: flip any bit that reduces C3.
        To speed up, we can check a random subset of indices per iteration, 
        or all indices if fast enough.
        With N=400, checking all 400 indices takes 400 FFTs.
        FFT 1024 is ~10us. 400 * 10us = 4ms.
        So one full pass of local search takes ~4ms.
        We can do many passes.
        """
        improved = True
        while improved:
            improved = False
            # Check all indices
            # To be faster, we can shuffle the order of indices to avoid bias
            indices = np.random.permutation(N)
            for idx in indices:
                if time.time() > time_limit:
                    break
                
                old_val = step[idx]
                step[idx] = -old_val
                new_c3 = _compute_c3(step)
                
                if new_c3 < c3:
                    c3 = new_c3
                    improved = True
                else:
                    step[idx] = old_val
            
            if time.time() > time_limit:
                break
        
        return step, c3

    # ILS Loop
    # We have ~140 seconds.
    # Each local search phase might take a few seconds if it converges slowly, or less if it's near a minimum.
    # We will run ILS until time runs out.
    
    while time.time() < DEADLINE - 5.0:
        # 1. Local Search
        # Set a sub-time-limit for local search to ensure we get back for perturbation
        # If local search takes too long, it's still exploring, but we want diversity.
        # Let's give it 2 seconds max per phase.
        local_limit = time.time() + 2.0
        current_step, current_c3 = local_search(current_step, current_c3, local_limit)
        
        # Update best
        if current_c3 < best_c3:
            best_c3 = current_c3
            best_step = current_step.copy()
        
        # 2. Perturbation
        # Flip a random contiguous block of bits
        # Block size: adaptive? Start small, grow if stuck?
        # Let's use a random block size between 5 and 50.
        block_size = np.random.randint(5, 50)
        start_idx = np.random.randint(0, N - block_size)
        
        # Flip the block
        current_step[start_idx:start_idx + block_size] *= -1
        
        # Optional: Randomly flip a few individual bits outside the block
        # num_extra = np.random.randint(0, 5)
        # extra_indices = np.random.choice(N, num_extra, replace=False)
        # current_step[extra_indices] *= -1
        
        # Calculate new C3 for the perturbed state
        current_c3 = _compute_c3(current_step)
        
        # If the perturbation made it worse, we still keep it for the next local search
        # because local search will try to fix it.
        # If we rejected worse moves, we'd just be stuck in the same basin.
        
        # Safety check: if we are running out of time, do one final local search
        if time.time() > DEADLINE - 10.0:
             final_step, final_c3 = local_search(current_step, current_c3, DEADLINE - 2.0)
             if final_c3 < best_c3:
                 best_c3 = final_c3
                 best_step = final_step.copy()
             break

    return best_step
