import numpy as np
import math
import random

def construct():
    N = 24
    # Grid size for evaluation during search
    M = 4096
    angles = 2 * np.pi * np.arange(M) / M
    z = np.exp(1j * angles)
    
    # Precompute powers of z for fast dot product
    # powers[k, m] = z[m] ** k
    powers = np.zeros((N, M), dtype=complex)
    powers[0, :] = 1
    for k in range(1, N):
        powers[k, :] = powers[k-1, :] * z

    def eval_peak(coeffs):
        """
        Compute the maximum absolute value of P(z) on the grid.
        coeffs: array of shape (N,) with values +/-1
        """
        vals = powers.T @ coeffs
        return np.max(np.abs(vals))

    def refine_peak_value(coeffs):
        """
        Refine the peak value using golden section search on the top few local maxima
        found in the coarse grid. This provides higher accuracy for the final score.
        """
        vals = powers.T @ coeffs
        abs_vals = np.abs(vals)
        
        # Find local maxima in the coarse grid
        local_max_indices = []
        for i in range(M):
            if abs_vals[i] >= abs_vals[(i-1) % M] and abs_vals[i] >= abs_vals[(i+1) % M]:
                local_max_indices.append(i)
        
        if not local_max_indices:
            return np.max(abs_vals)
        
        max_val = 0
        # Refine the top 10 local maxima to be safe
        local_max_indices.sort(key=lambda i: abs_vals[i], reverse=True)
        
        for idx in local_max_indices[:10]:
            delta = 2 * np.pi / M
            center = angles[idx]
            a = center - delta
            b = center + delta
            
            def f(theta):
                z_t = np.exp(1j * theta)
                p_val = 0
                for k in range(N):
                    p_val += coeffs[k] * (z_t ** k)
                return np.abs(p_val)
            
            # Golden section search
            invphi = (1 + math.sqrt(5)) / 2
            invphi2 = invphi * invphi
            c = b - (b - a) / invphi
            d = a + (b - a) / invphi
            fc = f(c)
            fd = f(d)
            
            for _ in range(60):
                if fc < fd:
                    a = c
                    c = d
                    fc = fd
                    d = a + (b - a) / invphi2
                    fd = f(d)
                else:
                    b = d
                    d = c
                    fd = fc
                    c = b - (b - a) / invphi2
                    fc = f(c)
            
            val = max(fc, fd)
            if val > max_val:
                max_val = val
                
        return max_val

    # Simulated Annealing with 1 and 2 flips
    best_coeffs = None
    best_peak = float('inf')
    
    # Multiple restarts
    num_restarts = 10
    
    for r in range(num_restarts):
        if r == 0:
            # Start with Rudin-Shapiro or similar low-energy pattern
            # Rudin-Shapiro sequence generation
            p, q = [1], [1]
            while len(p) < N:
                p, q = p + q, p + [-c for c in q]
            current = np.array(p[:N], dtype=int)
        else:
            current = np.random.choice([-1, 1], size=N)
        
        current_peak = eval_peak(current)
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_coeffs = current.copy()
            
        T_start = 0.5
        T_end = 1e-6
        steps = 20000
        cooling_rate = (T_end / T_start) ** (1.0 / steps)
        T = T_start
        
        for i in range(steps):
            # Choose move type: 50% 1-flip, 50% 2-flip
            if random.random() < 0.5:
                # 1 flip
                idx1 = np.random.randint(N)
                old_val1 = current[idx1]
                current[idx1] = -old_val1
                new_peak = eval_peak(current)
                delta = new_peak - current_peak
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_coeffs = current.copy()
                else:
                    current[idx1] = old_val1
            else:
                # 2 flips
                idx1 = np.random.randint(N)
                idx2 = np.random.randint(N)
                if idx1 == idx2:
                    idx2 = (idx2 + 1) % N
                
                old_val1 = current[idx1]
                old_val2 = current[idx2]
                current[idx1] = -old_val1
                current[idx2] = -old_val2
                
                new_peak = eval_peak(current)
                delta = new_peak - current_peak
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_coeffs = current.copy()
                else:
                    current[idx1] = old_val1
                    current[idx2] = old_val2
                    
            T *= cooling_rate
            
    # Final Local Search: Greedy 1-flip and 2-flip
    current = best_coeffs.copy()
    current_peak = eval_peak(current)
    
    improved = True
    while improved:
        improved = False
        
        # 1-flip pass
        for i in range(N):
            old_val = current[i]
            current[i] = -old_val
            new_peak = eval_peak(current)
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
            else:
                current[i] = old_val
                
        # 2-flip pass
        for i in range(N):
            for j in range(i+1, N):
                old_val_i = current[i]
                old_val_j = current[j]
                current[i] = -old_val_i
                current[j] = -old_val_j
                new_peak = eval_peak(current)
                if new_peak < current_peak:
                    current_peak = new_peak
                    improved = True
                else:
                    current[i] = old_val_i
                    current[j] = old_val_j
                    
    return current.tolist()
