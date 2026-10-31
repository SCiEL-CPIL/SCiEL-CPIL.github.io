import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def compute_overlaps(h):
        # h is length N
        # overlap_k = sum_i h_i * (1 - h_{i-k})
        # = sum_i h_i - sum_i h_i h_{i-k}
        # sum_i h_i is constant N/2
        # We need C_k = sum_i h_i h_{i-k}
        # This is circular cross-correlation
        # Using FFT for efficiency
        H = np.fft.fft(h)
        C = np.fft.ifft(H * np.conj(H)).real
        # C[k] corresponds to shift k
        overlaps = N * 0.5 - C
        return overlaps
    
    def sinusoidal_h(A, phi):
        """Generate sinusoidal step heights."""
        i = np.arange(N)
        h = 0.5 + A * np.sin(2 * np.pi * i / N + phi)
        # Clip to [0, 1]
        h = np.clip(h, 0, 1)
        # Project to sum exactly N/2
        diff = N / 2 - np.sum(h)
        h = h + diff / N
        h = np.clip(h, 0, 1)
        return h
    
    def objective(A, phi):
        """Compute maximum overlap for given sinusoidal parameters."""
        h = sinusoidal_h(A, phi)
        overlaps = compute_overlaps(h)
        return np.max(overlaps)
    
    # Grid search over A and phi
    best_score = np.inf
    best_A = 0.0
    best_phi = 0.0
    
    # Coarse grid
    A_values = np.linspace(0.0, 0.5, 50)
    phi_values = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    
    for A in A_values:
        if time.time() > deadline:
            break
        for phi in phi_values:
            score = objective(A, phi)
            if score < best_score:
                best_score = score
                best_A = A
                best_phi = phi
    
    # Local refinement around best point
    if time.time() < deadline - 10:
        # Fine grid around best
        A_fine = np.linspace(max(0, best_A - 0.1), min(0.5, best_A + 0.1), 30)
        phi_fine = np.linspace(best_phi - 0.2, best_phi + 0.2, 30)
        
        for A in A_fine:
            if time.time() > deadline - 5:
                break
            for phi in phi_fine:
                score = objective(A, phi)
                if score < best_score:
                    best_score = score
                    best_A = A
                    best_phi = phi
    
    # Generate final h
    best_h = sinusoidal_h(best_A, best_phi)
    
    # Try to improve with direct local search on h
    if time.time() < deadline - 5:
        step_size = 0.02
        current_h = best_h.copy()
        current_score = best_score
        
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            
            overlaps = compute_overlaps(current_h)
            max_overlap = np.max(overlaps)
            
            if max_overlap < current_score:
                current_score = max_overlap
                best_h = current_h.copy()
            
            # Find the shift with max overlap
            k_max = np.argmax(overlaps)
            
            # Gradient: derivative of overlap_k wrt h_j
            # overlap_k = sum_i h_i (1 - h_{i-k})
            # d/dh_j = (1 - h_{j-k}) - h_{j+k}
            h_shifted_down = np.roll(current_h, k_max)
            h_shifted_up = np.roll(current_h, -k_max)
            grad = (1 - h_shifted_down) - h_shifted_up
            
            # Update
            new_h = current_h - step_size * grad
            new_h = np.clip(new_h, 0, 1)
            diff = N / 2 - np.sum(new_h)
            new_h = new_h + diff / N
            new_h = np.clip(new_h, 0, 1)
            
            new_overlaps = compute_overlaps(new_h)
            new_max_overlap = np.max(new_overlaps)
            
            if new_max_overlap < max_overlap:
                current_h = new_h
                step_size = min(step_size * 1.05, 0.1)
            else:
                step_size *= 0.9
                if step_size < 0.001:
                    break
    
    return best_h
# EVOLVE-BLOCK-END
