"""Signed autocorrelation maximum absolute-value variant."""
import numpy as np
import time

N = 400


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    def score(x):
        """Compute max |autocorr| / (sum)^2. Lower is better."""
        S = np.sum(x)
        if abs(S) < 1e-12:
            return float('inf')
        n = len(x)
        size = 1
        while size < 2 * n - 1:
            size *= 2
        X = np.fft.rfft(x, size)
        acf = np.fft.irfft(X * np.conj(X), size)[:n]
        max_abs = np.max(np.abs(acf))
        return max_abs / (S * S)

    rng = np.random.RandomState(42)
    
    # Initial random sequence
    best_x = rng.choice([-1, 1], N)
    best_score = score(best_x)
    
    # Simulated Annealing / Local Search
    # We want to minimize max |A(k)|.
    # Since sum A(k)^2 is constant, we want to flatten the spectrum.
    
    current_x = best_x.copy()
    current_score = best_score
    
    # Temperature for Simulated Annealing
    T = 1.0
    cooling_rate = 0.99
    
    # Precompute size for FFT
    size = 1
    while size < 2 * N - 1:
        size *= 2

    while time.time() < deadline - 2.0:
        # Compute current autocorrelation
        X = np.fft.rfft(current_x, size)
        acf = np.fft.irfft(X * np.conj(X), size)[:N]
        
        # Find the peak
        abs_acf = np.abs(acf)
        k_max = np.argmax(abs_acf)
        max_val = abs_acf[k_max]
        
        # Greedy Step: Find best flip to reduce |A(k_max)|
        # A(k) = sum x_i x_{i+k}
        # Flip j: x_j -> -x_j
        # Change in A(k_max) = -2 x_j (x_{j+k_max} + x_{j-k_max})
        # We want to reduce |A(k_max)|.
        # If A(k_max) > 0, we want negative change.
        # If A(k_max) < 0, we want positive change.
        
        # Indices for neighbors
        # x_{j+k} is valid if j+k < N
        # x_{j-k} is valid if j-k >= 0
        
        # Vectorized calculation of change for all j
        # Note: A(k) is symmetric, A(k) = A(-k). 
        # The term is x_j * (x_{j+k} + x_{j-k})
        
        # Create masks for valid indices
        idx = np.arange(N)
        
        # For k_max, neighbors are j+k_max and j-k_max
        # We need to handle boundaries.
        # x_{j+k} exists if j < N - k_max
        # x_{j-k} exists if j >= k_max
        
        neighbor_sum = np.zeros(N)
        
        # Contribution from x_{j+k}
        mask_pos = idx < (N - k_max)
        if np.any(mask_pos):
            neighbor_sum[mask_pos] += current_x[idx[mask_pos] + k_max]
            
        # Contribution from x_{j-k}
        mask_neg = idx >= k_max
        if np.any(mask_neg):
            neighbor_sum[mask_neg] += current_x[idx[mask_neg] - k_max]
            
        # Change if we flip x_j:
        # delta_A = -2 * current_x[j] * neighbor_sum[j]
        # We want to minimize |A(k_max) + delta_A|
        
        target_sign = np.sign(acf[k_max])
        if target_sign == 0:
            target_sign = 1.0
            
        # We want delta_A to oppose A(k_max)
        # So we want current_x[j] * neighbor_sum[j] to have the same sign as A(k_max)
        # Because delta = -2 * (product). If product has same sign as A, delta opposes A.
        
        products = current_x * neighbor_sum
        # Score for each flip: how much it reduces the peak
        # Reduction = A - (A + delta) = -delta = 2 * product (if sign matches)
        # If sign doesn't match, it increases.
        
        # We want to maximize the reduction of |A|
        # Let's just evaluate the new magnitude for the top candidates to be safe, 
        # or pick the one with max product * sign(A)
        
        potential_reductions = products * target_sign
        # Find best candidate
        j_best = np.argmax(potential_reductions)
        
        # Calculate actual new score for this flip
        # To avoid O(N^2) or expensive FFT for every step, we only evaluate the global max 
        # after the flip. But we can estimate local improvement.
        
        # Let's just do the flip and check global score.
        # To allow escaping local minima, use SA.
        
        # Generate a random alternative with some probability
        if rng.random() < 0.1:
            j_candidate = rng.randint(N)
        else:
            j_candidate = j_best
            
        # Flip
        current_x[j_candidate] *= -1
        
        # Compute new score
        X_new = np.fft.rfft(current_x, size)
        acf_new = np.fft.irfft(X_new * np.conj(X_new), size)[:N]
        max_abs_new = np.max(np.abs(acf_new))
        S_new = np.sum(current_x)
        if abs(S_new) < 1e-12:
             new_score = float('inf')
        else:
             new_score = max_abs_new / (S_new * S_new)
        
        # Acceptance criterion
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = current_x.copy()
        else:
            # Metropolis criterion
            delta_E = new_score - current_score
            if delta_E > 0:
                prob = np.exp(-delta_E / T)
                if rng.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    current_x[j_candidate] *= -1
        
        # Cool down
        T *= cooling_rate
        if T < 0.01:
            T = 0.01 # Floor temperature to allow some exploration

    return best_x
