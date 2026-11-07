import numpy as np

N = 100

def construct():
    # Start with seed
    h = np.full(N, 0.5)
    
    def get_value(h):
        corr = np.correlate(h, 1 - h, mode="full")
        return np.max(corr)
    
    def get_grad_for_max_lag(h):
        corr = np.correlate(h, 1 - h, mode="full")
        max_idx = np.argmax(corr)
        s = max_idx - (N - 1) # Lag corresponding to max correlation
        
        # We want to minimize C_s = sum(h_i) - sum(h_i * h_{i-s})
        # grad C_s / d h_j = 1 - (h_{j-s} + h_{j+s})
        # Note: indices must be valid for correlation (no wrap around)
        # h is 0..N-1.
        # h_{j-s} exists if 0 <= j-s < N
        # h_{j+s} exists if 0 <= j+s < N
        
        grad = np.ones(N)
        
        # Term h_{j-s}
        # j-s in [0, N-1] => s <= j <= N-1+s
        # If s >= 0: j in [s, N-1]. h_{j-s} = h[0:N-s] shifted?
        # Let's just compute it directly or via vectorization.
        
        # Contribution from h_{j-s}:
        # For j, we add h_{j-s} to the sum being subtracted.
        # So grad[j] -= h[j-s] if valid.
        
        # Contribution from h_{j+s}:
        # grad[j] -= h[j+s] if valid.
        
        if s >= 0:
            # j-s >= 0 => j >= s
            # j-s < N => j < N+s (always true for j<N)
            # So for j in [s, N-1], h_{j-s} is h[j-s]
            grad[s:] -= h[:N-s]
            
            # j+s < N => j < N-s
            # j+s >= 0 (always true)
            # So for j in [0, N-s-1], h_{j+s} is h[j+s]
            grad[:N-s] -= h[s:]
        else:
            # s < 0
            # Let k = -s > 0
            # j - (-k) = j+k
            # j+k < N => j < N-k
            # j+k >= 0 (always)
            # So for j in [0, N-k-1], h_{j+k} is h[j+k]
            # This corresponds to the "h_{j-s}" term in the formula?
            # Wait, the term is h_{j-s}. If s=-k, it's h_{j+k}.
            # So grad[j] -= h[j+k] for j in [0, N-k-1].
            k = -s
            grad[:N-k] -= h[k:]
            
            # Term h_{j+s} = h_{j-k}
            # j-k >= 0 => j >= k
            # j-k < N => j < N+k (always)
            # So for j in [k, N-1], h_{j-k} is h[j-k]
            grad[k:] -= h[:N-k]

        return grad, max_idx

    current_h = h.copy()
    
    # Learning rate schedule
    eta = 0.1
    eta_decay = 0.999
    steps = 2000
    
    for step in range(steps):
        grad, max_idx = get_grad_for_max_lag(current_h)
        
        # Gradient step
        new_h = current_h - eta * grad
        
        # Project to [0, 1]
        new_h = np.clip(new_h, 0, 1)
        
        # Project to sum N/2
        # Simple projection: adjust elements that are not at 0 or 1
        target_sum = N / 2
        current_sum = np.sum(new_h)
        diff = target_sum - current_sum
        
        if abs(diff) > 1e-9:
            # Find indices where we can adjust
            # We can increase if new_h < 1, decrease if new_h > 0
            # To minimize perturbation, adjust those closest to the middle?
            # Or just distribute evenly among adjustable elements?
            # Better: use a simple iterative correction or just add diff/N to all and clip again?
            # Adding diff/N might push some out of bounds.
            
            # Strategy:
            # 1. Identify free variables (0 < h < 1)
            free_mask = (new_h > 1e-6) & (new_h < 1 - 1e-6)
            
            if np.any(free_mask):
                # Distribute diff among free variables
                # This might push some to bounds.
                # Simple approach: add diff / count(free) to free vars
                n_free = np.sum(free_mask)
                adjustment = diff / n_free
                new_h[free_mask] += adjustment
                
                # Re-clip
                new_h = np.clip(new_h, 0, 1)
                
                # If sum is still off, iterate a few times or use a more robust projection
                # For now, one step is often enough if eta is small
                # But let's do a few iterations of this projection
                for _ in range(10):
                    current_sum = np.sum(new_h)
                    diff = target_sum - current_sum
                    if abs(diff) < 1e-12:
                        break
                    free_mask = (new_h > 1e-6) & (new_h < 1 - 1e-6)
                    if not np.any(free_mask):
                        break
                    n_free = np.sum(free_mask)
                    adjustment = diff / n_free
                    new_h[free_mask] += adjustment
                    new_h = np.clip(new_h, 0, 1)
            else:
                # All at bounds, cannot adjust sum easily without breaking box constraint
                # This implies the target sum is not reachable with current box config?
                # But sum 50 is always reachable with 100 vars in [0,1].
                # This case shouldn't happen unless we hit a corner that doesn't sum to 50.
                # If we hit a corner, we should move away from it.
                # For now, revert to previous if this happens?
                # Or just force a random perturbation.
                pass

        # Check improvement
        old_val = get_value(current_h)
        new_val = get_value(new_h)
        
        if new_val < old_val:
            current_h = new_h
        else:
            # Try smaller step or revert
            # If we don't improve, we might be stuck.
            # For now, just keep current_h if no improvement.
            # We can also try a random perturbation if stuck for many steps.
            pass
            
        eta *= eta_decay

    # Final check
    if abs(np.sum(current_h) - N/2) > 1e-9 * N:
        return [0.5] * N
        
    return list(current_h)
