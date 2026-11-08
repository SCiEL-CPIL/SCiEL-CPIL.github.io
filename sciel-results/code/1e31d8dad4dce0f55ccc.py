import numpy as np
import time

N = 400

def compute_R_fft(h):
    """Compute linear autocorrelation using FFT."""
    h_pad = np.zeros(2 * N)
    h_pad[:N] = h
    fft_h = np.fft.fft(h_pad)
    power = np.abs(fft_h) ** 2
    R_circ = np.fft.ifft(power).real
    return R_circ[:N]

def compute_cost(R):
    """Compute sum of squares of non-zero lag autocorrelations."""
    return np.sum(R[1:] ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with random sequence
    h = np.random.choice([-1, 1], N)
    R = compute_R_fft(h)
    current_cost = compute_cost(R)
    best_h = h.copy()
    best_cost = current_cost
    
    # Simulated Annealing parameters
    T = 1.0
    cooling = 0.999
    
    # Precompute indices for single flip update
    # We will update R in O(N)
    
    while time.time() < deadline:
        # Determine move type
        if np.random.random() < 0.8:
            # Single flip
            i = np.random.randint(N)
            old_h_i = h[i]
            h[i] = -old_h_i
            
            # Update R in O(N)
            # Delta R(k) = -2 * h_i * (h_{i+k} + h_{i-k})
            # Note: h[i] is now the new value.
            # The change is from old_h_i to -old_h_i.
            # Delta = -2 * old_h_i * ( ... )
            
            delta_R = np.zeros(N)
            # Terms where h_i is the first element: h_i * h_{i+k}
            # Valid for k such that i+k < N
            # k = 0 to N-1-i
            # R(k) includes h_i * h_{i+k}
            # So for k in 1..N-1-i:
            #   delta_R[k] += -2 * old_h_i * h[i+k]
            
            # Terms where h_i is the second element: h_{i-k} * h_i
            # Valid for k such that i-k >= 0
            # k = 1 to i
            # R(k) includes h_{i-k} * h_i
            # So for k in 1..i:
            #   delta_R[k] += -2 * old_h_i * h[i-k]
            
            # Vectorized update
            # k range 1 to N-1
            ks = np.arange(1, N)
            
            # Mask for first part: i+k < N => k < N-i
            mask1 = ks < (N - i)
            # Mask for second part: i-k >= 0 => k <= i
            mask2 = ks <= i
            
            # Compute contributions
            # For mask1: h[i+k]
            # We need h[i+1 : i+mask1.sum()+1]? No, just index h[i+ks] where valid.
            # Simpler:
            delta_R[mask1] += -2 * old_h_i * h[i + ks[mask1]]
            delta_R[mask2] += -2 * old_h_i * h[i - ks[mask2]]
            
            # Calculate new cost
            # E_new = sum (R(k) + delta_R(k))^2
            # We only need to sum over k where delta_R != 0?
            # Actually, delta_R is zero for k where no terms involve i.
            # But R(k) is non-zero everywhere.
            # We can just compute the sum over all k=1..N-1.
            # R_new = R + delta_R
            # Cost_new = sum(R_new[1:]**2)
            
            R_new = R + delta_R
            cost_new = compute_cost(R_new)
            
            delta_cost = cost_new - current_cost
            
            if delta_cost < 0 or np.random.random() < np.exp(-delta_cost / T):
                # Accept
                R = R_new
                current_cost = cost_new
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_h = h.copy()
            else:
                # Reject
                h[i] = old_h_i
        else:
            # Multi-flip (2 or 3 indices)
            n_flips = np.random.choice([2, 3])
            indices = np.random.choice(N, n_flips, replace=False)
            
            # Save old h
            h_old = h.copy()
            
            # Flip
            for idx in indices:
                h[idx] = -h[idx]
                
            # Recompute R via FFT
            R_new = compute_R_fft(h)
            cost_new = compute_cost(R_new)
            delta_cost = cost_new - current_cost
            
            if delta_cost < 0 or np.random.random() < np.exp(-delta_cost / T):
                # Accept
                R = R_new
                current_cost = cost_new
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_h = h.copy()
            else:
                # Reject
                h = h_old
        
        # Cool down
        T *= cooling
        if T < 1e-6:
            T = 1e-6

    return best_h

