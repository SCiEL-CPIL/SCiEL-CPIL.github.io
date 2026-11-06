import numpy as np
import time

def construct():
    N = 24
    # Dense grid for accurate sup norm estimation
    M = 8192
    angles = 2 * np.pi * np.arange(M) / M
    k_idx = np.arange(N)
    Z = np.exp(1j * np.outer(angles, k_idx))
    
    # Precompute Z.T for efficient flip calculations if needed, 
    # but V - 2*c_k*Z[:,k] is efficient enough.
    
    def get_score(c):
        # c is shape (N,)
        # Returns max |P(z)|
        P = Z @ c
        return np.max(np.abs(P))
        
    def get_precise_score(c):
        # Refine the max by checking around the peak
        P = Z @ c
        abs_P = np.abs(P)
        peak_idx = np.argmax(abs_P)
        
        # Golden section search in a small window around peak_idx
        # Window of +/- 2 grid points
        delta = 2 * np.pi / M
        a = angles[peak_idx] - 2*delta
        b = angles[peak_idx] + 2*delta
        
        def f(theta):
            z = np.exp(1j * theta)
            return np.abs(np.dot(c, z ** k_idx))
            
        gr = (np.sqrt(5) - 1) / 2
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        f1 = f(x1)
        f2 = f(x2)
        
        for _ in range(30):
            if f1 < f2:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + gr * (b - a)
                f2 = f(x2)
            else:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - gr * (b - a)
                f1 = f(x1)
                
        return max(f1, f2)

    # Generate Golay-Rudin-Shapiro sequence
    def generate_grs(n):
        p = [1]
        q = [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:n], dtype=float)
        
    # Initial solution: GRS
    current_coeffs = generate_grs(N).copy()
    current_val = get_score(current_coeffs)
    
    best_coeffs = current_coeffs.copy()
    best_val = current_val
    
    # Simulated Annealing
    start_temp = 1.0 # Max possible change in score is roughly 2*sqrt(N)? No, max change is 2*|Z[:,k]| max = 2.
    # Score is max |P|. P changes by 2*c_k*z^k.
    # So delta score is bounded.
    # Let's set T0 based on typical delta.
    T0 = 0.5
    T_end = 0.001
    alpha = 0.999 # Cooling rate
    
    # Time limit management
    t_start = time.time()
    time_limit = 100 # Leave 20s for final polish and safety
    
    # Precompute Z columns for fast access
    # Z is (M, N)
    
    while T0 > T_end:
        if time.time() - t_start > time_limit:
            break
            
        # Random flip
        k = np.random.randint(N)
        old_coeff = current_coeffs[k]
        
        # Calculate new value
        # V_new = V - 2 * old_coeff * Z[:, k]
        # We don't store V explicitly in the loop to save memory, 
        # but recalculating Z @ c is O(M*N). 
        # Better: Keep V = Z @ current_coeffs.
        
        # Let's restructure to keep V.
        # But inside the loop, we need to be careful.
        
        # Actually, let's just calculate the new max directly.
        # It's O(M).
        
        # Flip
        current_coeffs[k] *= -1
        new_val = get_score(current_coeffs)
        delta = new_val - current_val
        
        if delta < 0 or np.random.random() < np.exp(-delta / T0):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_coeffs = current_coeffs.copy()
        else:
            # Revert
            current_coeffs[k] *= -1
            
        T0 *= alpha
        
    # Final Polish: Hill Climbing on best_coeffs
    current_coeffs = best_coeffs.copy()
    current_val = get_score(current_coeffs)
    
    improved = True
    while improved:
        improved = False
        # Try all flips
        for k in range(N):
            current_coeffs[k] *= -1
            new_val = get_score(current_coeffs)
            if new_val < current_val - 1e-12:
                current_val = new_val
                improved = True
            else:
                current_coeffs[k] *= -1
                
    # Check if polished solution is better
    if current_val < best_val:
        best_coeffs = current_coeffs.copy()
        best_val = current_val
        
    # Return integer list
    return best_coeffs.astype(int).tolist()
