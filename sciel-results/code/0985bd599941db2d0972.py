import numpy as np
import time

N = 600
TIME_LIMIT = 140  # Safety margin before 150s kill

def compute_score_exact(f):
    """Compute the normalized maximum of the autoconvolution exactly."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def construct():
    start_time = time.time()
    
    # Initial state: uniform distribution
    # Normalize so that sum(f) = N. This makes the score = max(conv)/N^2.
    # Actually, let's keep sum(f) = 1 for simplicity in math, but the score is scale invariant.
    # Let's work with f such that sum(f) = 1.
    # The score is max(conv) / (sum(f))^2. If sum(f)=1, score = max(conv).
    
    f = np.ones(N) / N
    sum_f = 1.0
    
    # Compute initial convolution
    conv = np.convolve(f, f, mode='full')
    current_score = np.max(conv)
    
    # Precompute indices for incremental update?
    # conv[m] = sum_j f[j] f[m-j]
    # If we change f[k] by delta, how does conv change?
    # conv_new[m] = conv_old[m] + delta * (f[m-k] + f[k-m]) + delta^2 (if m-k==k-m i.e. m=2k)
    # Specifically:
    # delta_conv[m] = delta * f[m-k] + delta * f[k-m] + (delta^2 if m == 2k else 0)
    # Note: indices for f are 0..N-1. m is 0..2N-2.
    # f[index] is 0 if index < 0 or index >= N.
    
    # We will use a simulated annealing / random coordinate descent approach.
    # To make it efficient, we track the current conv vector.
    
    # Temperature schedule
    T = 0.01
    cooling_rate = 0.999
    
    # We want to minimize the score.
    # Let's try a mix of random restarts and local search.
    # Given the time limit, a pure local search from a good start might be best.
    
    # Better start: try a few simple shapes
    # 1. Uniform
    # 2. Triangle (linear ramp up then down)
    # 3. Cosine bump
    
    candidates = []
    
    # Candidate 1: Uniform
    f1 = np.ones(N) / N
    candidates.append(f1)
    
    # Candidate 2: Triangle
    f2 = np.linspace(0, 1, N//2)
    f2 = np.concatenate([f2, f2[::-1]])
    f2 = f2 / np.sum(f2)
    candidates.append(f2)
    
    # Candidate 3: Cosine bump in center
    x = np.linspace(0, 1, N)
    f3 = np.cos(np.pi * (x - 0.5)) ** 2
    f3 = np.maximum(f3, 0)
    f3 = f3 / np.sum(f3)
    candidates.append(f3)
    
    # Candidate 4: Two bumps
    f4 = np.zeros(N)
    f4[:N//4] = 1.0
    f4[3*N//4:] = 1.0
    f4 = f4 / np.sum(f4)
    candidates.append(f4)
    
    # Evaluate candidates and pick the best start
    best_start = None
    best_start_score = np.inf
    for c in candidates:
        s = np.sum(c)
        if s < 1e-12: continue
        c_norm = c / s
        conv_c = np.convolve(c_norm, c_norm, mode='full')
        score_c = np.max(conv_c)
        if score_c < best_start_score:
            best_start_score = score_c
            best_start = c_norm.copy()
            
    f = best_start
    sum_f = 1.0
    conv = np.convolve(f, f, mode='full')
    current_score = np.max(conv)
    
    # Local Search / Simulated Annealing
    # We will perturb random coordinates.
    # To be fast, we only update the conv array incrementally.
    
    # We need to be careful with the "max" operation. 
    # If the max changes, we need to know the new max.
    # Maintaining a sorted structure or heap for the max is complex with incremental updates.
    # However, N=600, so conv has 1199 elements.
    # Recomputing max(conv) is O(N). 
    # Updating conv is O(N) for a single coordinate change (since it affects all lags).
    # So each step is O(N). 
    # 120s / (N * cost_per_op). N=600. 
    # Python loop overhead is high. We should vectorize the conv update.
    
    # Let's try a vectorized update for a single coordinate change.
    # delta_conv = delta * (shift_left(f, k) + shift_right(f, k)) + delta^2 * (m == 2k)
    # shift_left(f, k)[m] = f[m-k]
    # shift_right(f, k)[m] = f[k-m]
    
    # Actually, let's just do a simple random restart + local refinement using numpy ops.
    # The previous program used FFT for scoring many random candidates.
    # This new strategy focuses on *intelligent* local search from good starts.
    
    # Let's implement a fast local search using the fact that we can update conv in O(N) using numpy slicing.
    
    best_f = f.copy()
    best_score = current_score
    
    # Define a function to evaluate a single coordinate change
    # We'll do a "greedy" hill climbing with random restarts of the perturbation direction.
    
    # To speed up, we can batch updates? No, max is non-linear.
    # Let's stick to single coordinate updates but make them fast.
    
    # Optimization: 
    # Instead of updating the whole conv array and taking max, 
    # we can estimate the change in max. 
    # But exactness is required.
    
    # Let's try a different angle: 
    # The objective is max_m sum_j f[j]f[m-j].
    # This is equivalent to finding the peak of the autocorrelation.
    
    # Let's use a "coordinate descent" where we optimize one bin at a time,
    # but we do it in a randomized order, and we use a "lookahead" or "gradient" step.
    
    # Given the constraints, let's try a robust Simulated Annealing with incremental updates.
    
    def update_conv_single(f, conv, k, new_val):
        """
        Update conv array when f[k] changes from old_val to new_val.
        Returns new conv array (in place or new).
        """
        old_val = f[k]
        delta = new_val - old_val
        if abs(delta) < 1e-15:
            return conv
        
        # conv[m] = sum_j f[j] f[m-j]
        # Change in conv[m] = delta * f[m-k] + delta * f[k-m] + delta^2 (if m==2k)
        
        # We need to construct the vector of changes.
        # m ranges from 0 to 2N-2.
        # f[m-k] is non-zero when 0 <= m-k < N => k <= m < k+N
        # f[k-m] is non-zero when 0 <= k-m < N => k-N < m <= k
        
        # Let's create a temporary array for the delta_conv
        # This is O(N) memory and O(N) time.
        
        # Vectorized approach:
        # The term delta * f[m-k] corresponds to shifting f by k.
        # The term delta * f[k-m] corresponds to shifting reversed f by k.
        
        # Let's just do it with slices to be safe and fast enough for N=600.
        
        # Create a zero array of length 2N-1
        # Actually, we can update conv in place if we are careful, or just create a new one.
        # Creating a new one is safer.
        
        # f_shifted_left[m] = f[m-k]
        # f_shifted_right[m] = f[k-m]
        
        # Let's compute the contributions.
        # contrib1 = delta * f[m-k]
        # contrib2 = delta * f[k-m]
        
        # We can use np.roll or slicing.
        
        # Let's create the full delta array.
        # m is 0..2N-2
        
        # For m in [0, 2N-2]:
        # idx1 = m - k. If 0 <= idx1 < N, add delta * f[idx1]
        # idx2 = k - m. If 0 <= idx2 < N, add delta * f[idx2]
        
        # This is effectively:
        # delta_conv = delta * (np.roll(f, k, 2*N-1)[:2*N-1] + np.roll(f, -k, 2*N-1)[:2*N-1])
        # Wait, np.roll wraps around. We need zero padding.
        
        # Let's use explicit slicing.
        delta_conv = np.zeros(2*N-1)
        
        # f[m-k] term
        # m-k in [0, N-1] => m in [k, k+N-1]
        if k < 2*N-1:
            start1 = k
            end1 = k + N
            if end1 > 2*N-1:
                end1 = 2*N-1
            # f[0..end1-k-1]
            # m goes from k to end1-1
            # m-k goes from 0 to end1-k-1
            # So delta_conv[k:end1] += delta * f[0:end1-k]
            delta_conv[start1:end1] += delta * f[0:end1-k]
            
        # f[k-m] term
        # k-m in [0, N-1] => m in [k-N+1, k]
        start2 = k - N + 1
        end2 = k + 1
        if start2 < 0:
            start2 = 0
        if end2 > 2*N-1:
            end2 = 2*N-1
        # m goes from start2 to end2-1
        # k-m goes from k-start2 to k-(end2-1)
        # So delta_conv[start2:end2] += delta * f[k-start2 : k-end2+2]
        # Note: f indices must be reversed? No.
        # f[k-m]. If m=start2, idx=k-start2. If m=end2-1, idx=k-end2+1.
        # So we take f[k-start2 : k-end2+2]
        delta_conv[start2:end2] += delta * f[k-start2 : k-end2+2]
        
        # delta^2 term at m=2k
        if 0 <= 2*k < 2*N-1:
            delta_conv[2*k] += delta * delta
            
        return conv + delta_conv

    # This function is O(N). 
    # Let's run the annealing loop.
    
    T = 0.005
    min_T = 1e-6
    step_size = 0.01
    
    current_f = f.copy()
    current_conv = conv.copy()
    current_score = np.max(current_conv)
    
    # We will perform many iterations.
    # Estimate time: 100,000 iterations * (O(N) update + O(N) max) 
    # N=600. 100k * 1200 ops = 120M ops.
    # Python numpy overhead is ~1us per op? Maybe 10us.
    # 100k * 20us = 2s. This should be fast enough.
    
    # Let's do 500,000 iterations if time permits.
    
    iterations = 0
    while time.time() - start_time < TIME_LIMIT - 5:
        iterations += 1
        
        # Random coordinate
        k = np.random.randint(0, N)
        
        # Propose a change
        # We want to keep sum(f) = 1.
        # If we change f[k], we must adjust another f[j] to keep sum constant?
        # Or we can just change f[k] and then renormalize?
        # Renormalizing changes ALL bins, which is expensive for incremental conv.
        
        # Alternative: Change f[k] by delta, and f[j] by -delta (for j != k).
        # This keeps sum constant.
        # This is a 2-coordinate move.
        
        # Let's pick a random j != k.
        j = np.random.randint(0, N)
        while j == k:
            j = np.random.randint(0, N)
            
        # Propose delta
        # Scale delta with T
        delta = np.random.normal(0, T) * step_size
        
        # Check bounds: f[k] >= 0, f[j] >= 0
        new_fk = current_f[k] + delta
        new_fj = current_f[j] - delta
        
        if new_fk < 0 or new_fj < 0:
            # Reject or clip
            # Let's just reject for simplicity in SA
            T *= cooling_rate
            if T < min_T:
                T = min_T
            continue
            
        # Calculate new score
        # We need to update conv for both k and j.
        # conv_new = conv + delta_conv(k, delta) + delta_conv(j, -delta)
        
        # To avoid creating multiple large arrays, we can compute the total delta_conv.
        # Or just do two updates.
        
        # Let's compute the change in conv.
        # We'll create a temporary delta array.
        
        # Actually, let's just compute the new conv values at the affected lags?
        # No, the max could be anywhere.
        
        # Let's use the update function twice.
        # But we need to apply them to the same base conv.
        
        # conv_temp = current_conv.copy()
        # conv_temp = update_conv_single(current_f, conv_temp, k, new_fk)
        # conv_temp = update_conv_single(current_f, conv_temp, j, new_fj)
        # new_score = np.max(conv_temp)
        
        # This is 2 * O(N) + O(N) max = 3N.
        
        # Let's optimize: combine the deltas.
        # delta_conv_total = delta_conv(k, delta) + delta_conv(j, -delta)
        
        # We can write a combined updater.
        
        # For now, let's just do the two updates. It's clear and correct.
        
        # Optimization: Only calculate max if the change is likely to be an improvement?
        # No, SA needs the exact score.
        
        # Let's try to speed up the update by doing it in one pass.
        
        # Let's just use the simple approach and see if it's fast enough.
        
        # We'll create a helper that returns the delta_conv for a single change, 
        # then sum them.
        
        def get_delta_conv(f, k, delta):
            delta_conv = np.zeros(2*N-1)
            
            # Term 1: delta * f[m-k]
            start1 = k
            end1 = k + N
            if end1 > 2*N-1:
                end1 = 2*N-1
            if start1 < 2*N-1:
                delta_conv[start1:end1] += delta * f[0:end1-k]
                
            # Term 2: delta * f[k-m]
            start2 = k - N + 1
            end2 = k + 1
            if start2 < 0:
                start2 = 0
            if end2 > 2*N-1:
                end2 = 2*N-1
            delta_conv[start2:end2] += delta * f[k-start2 : k-end2+2]
            
            # Term 3: delta^2 at 2k
            if 0 <= 2*k < 2*N-1:
                delta_conv[2*k] += delta * delta
                
            return delta_conv

        dc1 = get_delta_conv(current_f, k, delta)
        dc2 = get_delta_conv(current_f, j, -delta)
        
        new_conv = current_conv + dc1 + dc2
        new_score = np.max(new_conv)
        
        # Acceptance criterion
        if new_score < current_score:
            current_score = new_score
            current_f[k] = new_fk
            current_f[j] = new_fj
            current_conv = new_conv
            
            # Update global best
            if current_score < best_score:
                best_score = current_score
                best_f = current_f.copy()
                
            # Cool down
            T *= cooling_rate
            if T < min_T:
                T = min_T
        else:
            # Accept with probability exp(-(new_score - current_score) / T)
            # Since we minimize, delta_E = new_score - current_score > 0
            if T > min_T:
                prob = np.exp(-(new_score - current_score) / T)
                if np.random.rand() < prob:
                    current_score = new_score
                    current_f[k] = new_fk
                    current_f[j] = new_fj
                    current_conv = new_conv
                    T *= cooling_rate
                    if T < min_T:
                        T = min_T
            else:
                T = min_T
        
        # Periodic check for time
        if iterations % 1000 == 0:
            if time.time() - start_time > TIME_LIMIT - 5:
                break

    # Final refinement: greedy coordinate descent from best_f
    # Try to improve best_f by adjusting single coordinates (renormalizing is hard, 
    # so we stick to 2-coordinate moves or just accept the SA result).
    
    # Let's do a few passes of greedy 2-coordinate descent.
    current_f = best_f.copy()
    current_conv = np.convolve(current_f, current_f, mode='full')
    current_score = np.max(current_conv)
    
    improved = True
    passes = 0
    while improved and passes < 5 and time.time() - start_time < TIME_LIMIT - 2:
        improved = False
        passes += 1
        # Random order of coordinates
        indices = np.random.permutation(N)
        for k in indices:
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            # Find best j and delta?
            # This is expensive. 
            # Let's just try a few random j's and small deltas.
            for _ in range(10):
                j = np.random.randint(0, N)
                if j == k: continue
                # Try to move mass from j to k if it reduces score
                # Or just do a small step.
                # Let's try delta = 0.001 * (f[j] - f[k])
                delta = 0.001 * (current_f[j] - current_f[k])
                if abs(delta) < 1e-6: continue
                new_fk = current_f[k] + delta
                new_fj = current_f[j] - delta
                if new_fk < 0 or new_fj < 0: continue
                
                dc1 = get_delta_conv(current_f, k, delta)
                dc2 = get_delta_conv(current_f, j, -delta)
                new_conv = current_conv + dc1 + dc2
                new_score = np.max(new_conv)
                
                if new_score < current_score:
                    current_score = new_score
                    current_f[k] = new_fk
                    current_f[j] = new_fj
                    current_conv = new_conv
                    improved = True
    
    if current_score < best_score:
        best_f = current_f
        best_score = current_score

    # Final exact check
    final_score = compute_score_exact(best_f)
    if final_score < best_score:
        best_score = final_score
        
    # Ensure nonnegative and return
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    start = time.time()
    f = construct()
    elapsed = time.time() - start
    score = compute_score_exact(f)
    print(f"Score: {score}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
    print(f"Time: {elapsed:.2f}s")
