import numpy as np
import time

N = 100

def _overlap_max(h):
    """Compute max cyclic overlap of h with 1-h, scaled by step width."""
    w = 2.0 / N
    # overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    # = N/2 - autocorr[k]
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr
    return w * np.max(overlaps)

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Initial guess: all 0.5
    # This is a symmetric point. We need to break symmetry to find a better solution.
    # A random start projected to the constraint manifold is usually better.
    
    best_h = None
    best_val = float('inf')
    
    # Try multiple random restarts to avoid local minima
    # The landscape of "max overlap" is rugged.
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Generate random initial state
        h = np.random.uniform(0, 1, N)
        
        # Project to sum = N/2
        # Simple projection: h = h + (target_sum - current_sum)/N
        # Then clip. But clipping breaks the sum.
        # We need a projection that preserves sum and bounds.
        # For N=100, a simple iterative adjustment works well enough for initialization.
        
        target_sum = N / 2.0
        for _ in range(20):
            diff = target_sum - np.sum(h)
            if abs(diff) < 1e-12:
                break
            h += diff / N
            h = np.clip(h, 0, 1)
            
        # If still not close, force it by adjusting the largest/smallest elements
        # But for random uniform, the simple average adjustment usually works within a few iterations.
        # Let's ensure exactness for the optimizer start.
        # Actually, for the local search, we will maintain the constraint exactly.
        # Let's do a more robust projection:
        # Sort indices, adjust from extremes.
        # But for N=100, the simple method above is likely fine if we just accept a small error 
        # and let the search correct it, OR we use a proper projection.
        
        # Let's use a proper projection function to be safe.
        def project_to_sum_bounds(x, target_sum):
            x = np.clip(x, 0, 1)
            # Iterative water-filling style adjustment
            for _ in range(50):
                s = np.sum(x)
                diff = target_sum - s
                if abs(diff) < 1e-15:
                    break
                # Distribute diff to elements that can accept it
                if diff > 0:
                    mask = x < 1.0
                    if not np.any(mask):
                        break
                    x[mask] += diff / np.sum(mask)
                else:
                    mask = x > 0.0
                    if not np.any(mask):
                        break
                    x[mask] += diff / np.sum(mask)
            x = np.clip(x, 0, 1)
            return x

        h = project_to_sum_bounds(h, target_sum)
        
        # Ensure the sum is exactly target_sum for the search to work correctly
        # The search moves will preserve the sum exactly.
        # If the projection left a tiny error, we can adjust one element.
        err = target_sum - np.sum(h)
        if abs(err) > 1e-12:
             # Find an element not at bound to adjust
             idxs = np.where((h > 1e-9) & (h < 1 - 1e-9))[0]
             if len(idxs) > 0:
                 h[idxs[0]] += err
                 h = np.clip(h, 0, 1)
        
        val = _overlap_max(h)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
        # Local Search Phase
        # We perform hill climbing.
        # Move: Pick two indices i, j. Increase h[i] by d, decrease h[j] by d.
        # This preserves the sum.
        # We need to ensure 0 <= h[i]+d <= 1 and 0 <= h[j]-d <= 1.
        
        # To speed up, we can try several moves in parallel?
        # No, _overlap_max is FFT-based, so it's O(N log N).
        # N=100 is very small. FFT is fast.
        # We can do many iterations.
        
        local_best_h = h.copy()
        local_best_val = val
        no_improve_count = 0
        max_no_improve = 500
        
        # Adaptive step size
        step_size = 0.05
        
        while no_improve_count < max_no_improve and time.time() < deadline - 5:
            # Pick random pair (i, j) with i != j
            i, j = np.random.choice(N, 2, replace=False)
            
            # Determine max possible step in both directions
            # h[i] can increase by (1 - h[i])
            # h[j] can decrease by h[j]
            # So d must be in [-h[j], 1-h[i]]
            # We pick a random d in this range, scaled by step_size?
            # Or pick a random d from a distribution centered at 0?
            
            # Let's try a random perturbation magnitude
            # Uniform in [-step_size, step_size]
            d = np.random.uniform(-step_size, step_size)
            
            # Check bounds
            new_hi = local_best_h[i] + d
            new_hj = local_best_h[j] - d
            
            if new_hi < 0 or new_hi > 1 or new_hj < 0 or new_hj > 1:
                # Skip this move or try smaller step?
                # For efficiency, just skip.
                continue
                
            # Apply move
            local_best_h[i] = new_hi
            local_best_h[j] = new_hj
            
            new_val = _overlap_max(local_best_h)
            
            if new_val < local_best_val:
                local_best_val = new_val
                no_improve_count = 0
                # Update global best
                if local_best_val < best_val:
                    best_val = local_best_val
                    best_h = local_best_h.copy()
            else:
                # Revert
                local_best_h[i] -= d
                local_best_h[j] += d
                no_improve_count += 1
                
                # If stuck, reduce step size and maybe randomize a bit (simulated annealing lite?)
                if no_improve_count % 100 == 0:
                    step_size *= 0.9
                    if step_size < 0.001:
                        break
                        
        # After local search, if we haven't improved global best, we might try a "kick"
        # but for now, let's just rely on the restarts.
        
    return best_h
