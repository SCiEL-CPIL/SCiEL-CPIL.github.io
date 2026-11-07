import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Strategy: Start with uniform 0.5, then apply gradient descent or simple perturbations
    # to minimize the max correlation.
    # The seed is 0.5.
    # A sinusoidal perturbation reduces C_0 significantly.
    
    # Try a few specific structured patterns
    best_h = np.full(N, 0.5)
    best_val = 0.5
    
    def evaluate(h):
        # h is array of 100
        # Check constraints
        if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
            return float('inf')
        if abs(np.sum(h) - 50.0) > 1e-6 * N:
            return float('inf')
        
        # Clip to [0, 1] for safety in calculation, though we should maintain feasibility
        h_safe = np.clip(h, 0.0, 1.0)
        
        # Calculate value
        # correlate(h, 1-h, 'full')
        # Note: np.correlate in 'full' mode returns length 2N-1
        # The max is taken over all shifts.
        corr = np.correlate(h_safe, 1.0 - h_safe, mode='full')
        max_corr = np.max(corr)
        return max_corr / N * 2.0

    # Test seed
    val_seed = evaluate(best_h)
    
    # Test sinusoidal patterns
    # h_i = 0.5 + A * cos(2*pi*k*i/N + phase)
    for A in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for k in [1, 2, 3, 4, 5]:
            for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
                if time.time() > deadline:
                    break
                h_test = 0.5 + A * np.cos(2 * np.pi * k * np.arange(N) / N + phase)
                # Ensure sum is 50
                # Sum of cos is 0 if k is integer and N is multiple of period?
                # Sum_{i=0}^{N-1} cos(2*pi*k*i/N + phase) = 0 for integer k >= 1.
                # So sum is 50.
                # Check bounds
                if np.min(h_test) < -1e-9 or np.max(h_test) > 1 + 1e-9:
                    continue
                
                val_test = evaluate(h_test)
                if val_test < best_val:
                    best_val = val_test
                    best_h = h_test.copy()
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # If we have time, do a simple local search (gradient step) from the best found
    if time.time() < deadline - 5.0:
        h_cur = best_h.copy()
        lr = 0.01
        for step in range(2000):
            if time.time() > deadline - 5.0:
                break
            
            # Calculate gradient approximation or just use finite difference on max term
            # Since max is non-differentiable, use subgradient or just random perturbations
            # Let's do a simple coordinate descent with random restarts if stuck
            
            # Perturb a few random indices
            idxs = np.random.choice(N, 5, replace=False)
            h_new = h_cur.copy()
            for idx in idxs:
                h_new[idx] += np.random.uniform(-0.1, 0.1)
            
            # Project to [0, 1]
            h_new = np.clip(h_new, 0.0, 1.0)
            
            # Project to sum 50
            # Simple projection: subtract mean error
            err = np.sum(h_new) - 50.0
            h_new -= err / N
            
            # Re-clip after sum projection? This might break sum.
            # Better: iterative projection or just accept small sum error if within tol?
            # Tolerance is 1e-9 * N = 1e-7.
            # If error is too big, skip.
            if abs(np.sum(h_new) - 50.0) > 1e-6:
                continue
            
            val_new = evaluate(h_new)
            if val_new < best_val:
                best_val = val_new
                h_cur = h_new.copy()
            else:
                # If not better, maybe try smaller step or different idx
                pass

    return best_h
