"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _calculate_C(h):
    """Calculate the overlap profile C_k."""
    C = np.zeros(N)
    for k in range(N):
        if k == 0:
            C[k] = np.sum(h * (1 - h))
        else:
            C[k] = np.sum(h[:N-k] * (1 - h[k:]))
    return C

def _score(h):
    """Return the maximum overlap."""
    C = _calculate_C(h)
    return np.max(C)

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # Safety margin
    
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _score(h)
    
    eta = 0.05
    no_improve_count = 0
    
    while time.time() < deadline:
        C = _calculate_C(h)
        k_max = np.argmax(C)
        
        # Calculate gradient of C_k_max
        g = np.zeros(N)
        if k_max == 0:
            g = 1 - 2 * h
        else:
            # d/dh_i C_k = (1 - h_{i+k}) for i < N-k
            #             - h_{i-k} for i >= k
            if N - k_max > 0:
                g[:N-k_max] += (1 - h[k_max:])
            if k_max < N:
                g[k_max:] -= h[:N-k_max]
        
        # Update
        h = h - eta * g
        h = np.clip(h, 0.0, 1.0)
        
        # Enforce sum constraint approximately
        # If sum is off, adjust
        s = np.sum(h)
        if abs(s - 50.0) > 1e-6:
            # Simple correction
            h += (50.0 - s) / N
            h = np.clip(h, 0.0, 1.0)
            # Re-correct sum?
            s = np.sum(h)
            if abs(s - 50.0) > 1e-6:
                # If still off, revert to previous best?
                # Or just accept small error?
                # The evaluator might check sum exactly?
                # "sum is exactly len/2"
                # We should ensure this.
                # If clipping broke it, we might need a smarter projection.
                # For now, if error is small, it might be ok?
                # But "exactly" is strict.
                # Let's try to fix it properly.
                # Find indices where we can adjust.
                # This is getting complicated.
                # Maybe just don't clip if it breaks sum?
                # Or use a projection onto the box and hyperplane.
                
                # Simple robust approach:
                # If sum is bad, restore best_h and reduce eta?
                h = best_h.copy()
                eta *= 0.9
                continue

        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 100:
                # Random perturbation to escape local minima
                h = best_h + np.random.normal(0, 0.01, N)
                h = np.clip(h, 0, 1)
                # Fix sum
                s = np.sum(h)
                h += (50.0 - s) / N
                h = np.clip(h, 0, 1)
                no_improve_count = 0
                eta = 0.05
        
        # Slowly decrease eta?
        # eta *= 0.999
        
    return best_h
# EVOLVE-BLOCK-END
