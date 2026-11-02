import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def get_max_overlap(h):
        """
        Calculates the maximum cyclic overlap of h with 1-h.
        Overlap k = sum h_i (1 - h_{i+k}) = N/2 - autocorr[k]
        Max Overlap = N/2 - min(autocorr[1:])
        """
        H = np.fft.fft(h)
        autocorr = np.fft.ifft(np.abs(H)**2).real
        # autocorr[0] is sum h^2, we ignore it (k=0)
        min_autocorr = np.min(autocorr[1:])
        return N / 2.0 - min_autocorr

    best_h = None
    best_score = np.inf

    # Phase 1: Binary Local Search
    # The optimal solution for this problem is often binary or close to it.
    # We search the space of binary sequences with sum N/2.
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        # Initialize random binary sequence with exactly N/2 ones
        h = np.zeros(N)
        ones_indices = np.random.choice(N, N//2, replace=False)
        h[ones_indices] = 1.0
        
        current_score = get_max_overlap(h)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
        no_improve_steps = 0
        max_no_improve = 200
        
        while time.time() < deadline - 30 and no_improve_steps < max_no_improve:
            # Compute autocorrelation to find the worst lag
            H = np.fft.fft(h)
            autocorr = np.fft.ifft(np.abs(H)**2).real
            k_min = np.argmin(autocorr[1:]) + 1
            
            zeros = np.where(h == 0)[0]
            ones = np.where(h == 1)[0]
            
            best_swap_score = current_score
            best_swap = None
            
            # Evaluate all possible swaps (0->1 at j, 1->0 at m)
            # N=100, so 50*50 = 2500 pairs. Very fast.
            for j in zeros:
                # Pre-calculate terms for j
                hj_k = h[(j + k_min) % N]
                hj_mk = h[(j - k_min) % N]
                
                for m in ones:
                    # Delta A at k_min
                    # A_k = sum h_i h_{i+k}
                    # Change: h_j 0->1, h_m 1->0
                    # Delta = h_{j+k} + h_{j-k} - h_{m+k} - h_{m-k}
                    delta = (hj_k + hj_mk) - (h[(m + k_min) % N] + h[(m - k_min) % N])
                    
                    # If delta is not positive, it won't improve A_k_min
                    if delta <= 0:
                        continue
                        
                    # Tentatively apply swap
                    h[j] = 1.0
                    h[m] = 0.0
                    
                    new_score = get_max_overlap(h)
                    
                    # Revert
                    h[j] = 0.0
                    h[m] = 1.0
                    
                    if new_score < best_swap_score:
                        best_swap_score = new_score
                        best_swap = (j, m)
                        
            if best_swap:
                j, m = best_swap
                h[j] = 1.0
                h[m] = 0.0
                current_score = best_swap_score
                no_improve_steps = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                no_improve_steps += 1
                # If stuck, try a random move to escape local minimum
                if np.random.random() < 0.2 and len(zeros) > 0 and len(ones) > 0:
                    j = np.random.choice(zeros)
                    m = np.random.choice(ones)
                    h[j] = 1.0
                    h[m] = 0.0
                    new_score = get_max_overlap(h)
                    if new_score < current_score:
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_h = h.copy()
                    else:
                        # Revert
                        h[j] = 0.0
                        h[m] = 1.0
                        no_improve_steps += 1

    # Phase 2: Continuous Polish
    # Refine the best binary solution using gradient-based optimization
    if best_h is not None:
        h0 = best_h.copy()
        
        def objective(x):
            h = np.empty(N)
            h[:N-1] = x
            h[N-1] = N/2.0 - np.sum(x)
            
            # Soft penalty for bounds
            if np.any(h < -1e-6) or np.any(h > 1 + 1e-6):
                penalty = np.sum(np.maximum(0, -h)**2 + np.maximum(0, h-1)**2)
                return 1000.0 + penalty
            
            return get_max_overlap(h)
            
        bounds = [(0, 1)] * (N-1)
        
        try:
            res = minimize(objective, h0[:N-1], method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-6})
            
            if res.fun < best_score:
                h_candidate = np.empty(N)
                h_candidate[:N-1] = res.x
                h_candidate[N-1] = N/2.0 - np.sum(res.x)
                
                # Clip and fix sum
                h_candidate = np.clip(h_candidate, 0, 1)
                diff = N/2.0 - np.sum(h_candidate)
                h_candidate[N-1] += diff
                
                # Check validity
                if 0 <= h_candidate[N-1] <= 1:
                    final_score = get_max_overlap(h_candidate)
                    if final_score < best_score:
                        best_score = final_score
                        best_h = h_candidate
        except Exception:
            pass

    # Fallback
    if best_h is None:
        best_h = np.full(N, 0.5)

    # Ensure exact sum
    diff = N/2.0 - np.sum(best_h)
    best_h[N-1] += diff
    
    # If adjustment broke bounds, revert to constant
    if best_h[N-1] < -1e-9 or best_h[N-1] > 1 + 1e-9:
        best_h = np.full(N, 0.5)

    return best_h
