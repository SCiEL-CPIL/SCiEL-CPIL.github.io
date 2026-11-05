import numpy as np
import time
import sys

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    def compute_value(h):
        """Compute the objective value (lower is better)."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    def h_from_coeffs(params):
        """
        Reconstruct h from Fourier coefficients.
        params: [A0, c1, s1, c2, s2, ...]
        h[k] = A0 + sum_{m=1}^K (c_m cos(2*pi*m*k/N) + s_m sin(2*pi*m*k/N))
        """
        A0 = params[0]
        k = np.arange(N)
        h = np.full(N, A0, dtype=np.float64)
        
        num_pairs = (len(params) - 1) // 2
        for i in range(num_pairs):
            freq = i + 1
            c = params[1 + 2*i]
            s = params[2 + 2*i]
            h += c * np.cos(2 * np.pi * freq * k / N)
            h += s * np.sin(2 * np.pi * freq * k / N)
            
        return h

    def objective(params):
        """
        Objective function for the Fourier coefficients.
        Uses a soft-max surrogate to make the objective smoother for optimization.
        """
        h = h_from_coeffs(params)
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        conv = np.convolve(h, h)
        # Use log-sum-exp for smooth max
        tau = 0.005
        conv_max = np.max(conv)
        if conv_max > 0:
            sm = conv_max + tau * np.log(np.sum(np.exp((conv - conv_max) / tau)))
        else:
            sm = conv_max
        val = abs(2.0 * N * sm / (s * s))
        return val

    # Strategy: 
    # 1. Start with a good initial guess (low-frequency Fourier series)
    # 2. Use L-BFGS-B to minimize the smooth surrogate
    # 3. Refine with coordinate-wise descent on the original (non-smooth) objective
    # 4. Exploit the fact that we only care about the MAX of the convolution
    
    # Initial parameters: DC + a few harmonics
    K_max = 15  # Number of harmonic pairs
    num_params = 1 + 2 * K_max
    
    # Multiple random starts
    np.random.seed(42)
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Try a few random Fourier-based starts
    for start_idx in range(5):
        params = np.zeros(num_params)
        params[0] = 1.0  # DC component
        for i in range(K_max):
            params[1 + 2*i] = np.random.uniform(-0.05, 0.05)
            params[2 + 2*i] = np.random.uniform(-0.05, 0.05)
        
        h = h_from_coeffs(params)
        val = compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Phase 1: L-BFGS-B optimization on Fourier parameters
    if time.time() < deadline - 40.0:
        try:
            from scipy.optimize import minimize
            
            # Optimize from the best start found
            # First, find which start was best
            for start_idx in range(5):
                params = np.zeros(num_params)
                params[0] = 1.0
                np.random.seed(42 + start_idx)
                for i in range(K_max):
                    params[1 + 2*i] = np.random.uniform(-0.05, 0.05)
                    params[2 + 2*i] = np.random.uniform(-0.05, 0.05)
                
                h = h_from_coeffs(params)
                val = compute_value(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
                    best_params = params.copy()
                    break
            else:
                best_params = np.zeros(num_params)
                best_params[0] = 1.0
            
            result = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_val:
                best_params = result.x
                best_h = h_from_coeffs(best_params)
                best_val = compute_value(best_h)
                print(f"L-BFGS-B result: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"L-BFGS-B error: {e}", file=sys.stderr)
    
    # Phase 2: Coordinate-wise descent on the raw h vector
    # This allows us to fine-tune individual elements to reduce the max convolution value
    
    h = best_h.copy()
    current_val = best_val
    s = np.sum(h)
    
    # Precompute convolution for efficiency
    conv = np.convolve(h, h)
    
    # For each element h[j], if we change it by delta, the convolution changes.
    # conv[k] = sum_i h[i] * h[k-i] (with appropriate indexing for full convolution)
    # Changing h[j] affects conv[j + i] for all i (the terms involving h[j])
    
    max_coord_iter = 20000
    coord_iter = 0
    
    while time.time() < deadline - 10.0 and coord_iter < max_coord_iter:
        coord_iter += 1
        improved = False
        
        # Cycle through coordinates (random order for better exploration)
        for j in range(N):
            if time.time() > deadline - 10.0:
                break
            
            old_hj = h[j]
            
            # Try a few perturbations
            # The gradient of the max convolution w.r.t. h[j] can guide us
            # For now, try small random steps
            
            # Estimate which convolution entries are "active" (near the max)
            # If we can identify the argmax, we can focus on reducing that entry
            
            # Simple approach: try a few delta values
            deltas = np.array([0.01, -0.01, 0.005, -0.005, 0.02, -0.02])
            best_delta = 0.0
            best_local_val = current_val
            
            for delta in deltas:
                h[j] = old_hj + delta
                new_s = s + delta
                if abs(new_s) < 1e-6:
                    h[j] = old_hj
                    continue
                # Quick approximation: recompute only affected conv entries
                # Full conv is O(N^2) per eval which is 400^2 = 160000, times 6 deltas = 1M
                # That's too slow. Let's just recompute full conv for the best few.
                conv_new = np.convolve(h, h)
                val = abs(2.0 * N * np.max(conv_new) / (new_s * new_s))
                if val < best_local_val:
                    best_local_val = val
                    best_delta = delta
                h[j] = old_hj
            
            if best_delta != 0.0:
                h[j] = old_hj + best_delta
                s = s + best_delta
                current_val = best_local_val
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
        
        if not improved:
            break
    
    # Final check
    final_val = compute_value(h)
    if final_val < best_val:
        best_val = final_val
        best_h = h.copy()
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-6:
        return np.ones(N)
    
    return best_h

if __name__ == "__main__":
    h = construct()
    val = compute_value(h)
    print(f"Result value: {val}")
