import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    N = 24
    # Precompute points on the unit circle
    # Use a dense grid to approximate the sup norm
    M = 4096
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    z = np.exp(1j * theta)
    
    # Powers of z: z^k for k=0..N-1
    # z_powers[i, k] = z[i] ** k
    z_powers = np.zeros((M, N), dtype=np.complex128)
    z_powers[:, 0] = 1.0
    for k in range(1, N):
        z_powers[:, k] = z_powers[:, k-1] * z
    
    def evaluate(coeffs):
        # coeffs is a numpy array of shape (N,) with values 1 or -1
        # P(z) = sum c_k z^k
        # vals = z_powers @ coeffs
        vals = z_powers @ coeffs
        return np.max(np.abs(vals))
    
    # Initialize with a random sequence
    best_coeffs = np.array([1 if random.random() < 0.5 else -1 for _ in range(N)])
    best_score = evaluate(best_coeffs)
    
    # Also try the Golay sequence as a starting point
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    golay_coeffs = np.array(p[:N])
    golay_score = evaluate(golay_coeffs)
    if golay_score < best_score:
        best_score = golay_score
        best_coeffs = golay_coeffs.copy()
    
    # Simulated Annealing
    T = 0.1  # Initial temperature
    T_min = 1e-4
    cooling_rate = 0.999
    
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    
    while time.time() < deadline:
        # Flip a random bit
        idx = random.randint(0, N - 1)
        old_val = current_coeffs[idx]
        current_coeffs[idx] = -old_val
        
        new_score = evaluate(current_coeffs)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_coeffs = current_coeffs.copy()
        else:
            # Accept with probability exp(-(new_score - current_score) / T)
            delta = new_score - current_score
            if delta < T:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Reject: revert
                    current_coeffs[idx] = old_val
            else:
                # Reject: revert
                current_coeffs[idx] = old_val
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Return as a list of integers
    return list(best_coeffs.astype(int))
