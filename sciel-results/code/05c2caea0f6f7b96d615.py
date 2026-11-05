import numpy as np
import time

def _eval_sup_norm(coeffs, N=4096):
    """
    Evaluate the sup norm of the polynomial P(z) = sum c_k z^k for |z|=1.
    Uses dense sampling.
    """
    # coeffs is a list of +1/-1
    # z = exp(i * theta)
    thetas = 2 * np.pi * np.arange(N) / N
    z = np.exp(1j * thetas)
    
    # P(z_j) = sum_{k=0}^{n-1} c_k z_j^k
    # We can compute this as coeffs @ z_powers
    # z_powers[k, j] = z_j^k
    # To save memory and time, we can compute iteratively or use broadcasting
    # n is small (24), N is 4096.
    # coeffs shape (n,), z shape (N,)
    # We want sum_k c_k * (z^k)
    
    # Efficient way:
    # Start with P = 0
    # For k from 0 to n-1:
    #   P += c_k * z^k
    
    # Or use matrix multiplication:
    # powers = np.power(z[None, :], np.arange(len(coeffs))[:, None])
    # This creates a (24, 4096) array, which is fine.
    
    powers = np.ones((len(coeffs), N), dtype=complex)
    for k in range(1, len(coeffs)):
        powers[k] = powers[k-1] * z
        
    # coeffs is real, powers is complex
    # P_vals = coeffs @ powers
    P_vals = np.dot(coeffs, powers)
    
    return np.max(np.abs(P_vals))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial candidate: Golay-Rudin-Shapiro truncated to 24
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=np.float64)
    
    # Precompute powers for all k up to 23
    N = 4096
    thetas = 2 * np.pi * np.arange(N) / N
    z = np.exp(1j * thetas)
    
    powers = np.ones((24, N), dtype=complex)
    for k in range(1, 24):
        powers[k] = powers[k-1] * z
        
    def fast_eval(coeffs):
        # coeffs is a numpy array or list
        return np.max(np.abs(np.dot(coeffs, powers)))
        
    best_score = fast_eval(best_coeffs)
    
    rng = np.random.default_rng(42)
    
    def local_search(coeffs):
        """
        Perform 1-bit and 2-bit flip hill climbing.
        Returns the improved coeffs and score.
        """
        current_score = fast_eval(coeffs)
        improved = True
        
        # 1-bit flips
        while improved:
            improved = False
            for i in range(24):
                # Flip bit i
                coeffs[i] *= -1
                new_score = fast_eval(coeffs)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    coeffs[i] *= -1
        
        # 2-bit flips (to escape local minima)
        improved = True
        while improved:
            improved = False
            for i in range(23):
                for j in range(i + 1, 24):
                    # Flip bits i and j
                    coeffs[i] *= -1
                    coeffs[j] *= -1
                    new_score = fast_eval(coeffs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        coeffs[i] *= -1
                        coeffs[j] *= -1
                        
        return coeffs, current_score

    # Run local search on the initial candidate
    best_coeffs, best_score = local_search(best_coeffs)
    
    # If we are already very close to the known optimum, we might stop early?
    # Known optimum ratio is 1.2583758.
    # Target max_norm = 1.2583758 * sqrt(24) ≈ 6.15633
    
    # List of deterministic seeds to try
    seeds = []
    # Original GRS
    seeds.append(np.array(p[:24], dtype=np.float64))
    
    # Shifted GRS (cyclic shifts)
    for shift in range(1, 24):
        seeds.append(np.roll(p[:24], shift).astype(np.float64))
        
    # Inverted GRS
    seeds.append(-np.array(p[:24], dtype=np.float64))
    
    # Random seeds
    # We will generate random seeds on the fly if time permits.
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        candidate = seed.copy()
        candidate, score = local_search(candidate)
        
        if score < best_score:
            best_score = score
            best_coeffs = candidate
            
        # Check if we found the optimum
        if abs(best_score / np.sqrt(24) - 1.2583758) < 1e-6:
            break

    # If we still have time, do random restarts
    while time.time() < deadline:
        candidate = rng.choice([-1, 1], size=24).astype(np.float64)
        candidate, score = local_search(candidate)
        
        if score < best_score:
            best_score = score
            best_coeffs = candidate
            
        # Check if we found the optimum
        if abs(best_score / np.sqrt(24) - 1.2583758) < 1e-6:
            break

    # Return the best coefficients found
    # The function must return 24 coefficients in {+1, -1}
    return [int(c) for c in best_coeffs]
