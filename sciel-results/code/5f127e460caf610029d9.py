import numpy as np
import time

N = 600
TIME_LIMIT = 115

def compute_score_fast(F):
    """
    Computes the normalized max of the autoconvolution for the signal
    obtained by hard-projecting the real part of ifft(F) to non-negative.
    """
    f_temp = np.fft.ifft(F).real
    f_hard = np.maximum(f_temp, 0.0)
    s = np.sum(f_hard)
    if s < 1e-15:
        return np.inf
    F_hard = np.fft.fft(f_hard)
    c = np.fft.ifft(np.abs(F_hard)**2).real
    return np.max(c) / (s * s)

def coordinate_descent(F_init, start_time, max_time):
    """
    Coordinate descent on frequency coefficients F to minimize the score.
    """
    F = F_init.copy()
    best_score = compute_score_fast(F)
    current_score = best_score
    step_size = 0.5
    half = N // 2
    
    # Indices to iterate
    indices = list(range(half + 1))
    
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        np.random.shuffle(indices)
        
        for k in indices:
            if time.time() - start_time > max_time:
                break
            
            old_Fk = F[k]
            # Save F[N-k] only if distinct
            if k != 0 and k != half:
                old_FNk = F[N - k]
            else:
                old_FNk = None
            
            # Try perturbations
            if k == 0 or k == half:
                deltas = [step_size, -step_size]
            else:
                deltas = [step_size, -step_size, 1j*step_size, -1j*step_size]
            
            for delta in deltas:
                # Apply perturbation
                F[k] = old_Fk + delta
                if old_FNk is not None:
                    F[N - k] = old_FNk + np.conj(delta)
                
                score = compute_score_fast(F)
                
                if score < current_score:
                    current_score = score
                    if score < best_score:
                        best_score = score
                    improved = True
                    break # Accept this delta and move to next k
                else:
                    # Revert
                    F[k] = old_Fk
                    if old_FNk is not None:
                        F[N - k] = old_FNk
            
            # If we accepted a delta, we break the delta loop.
            # We continue to the next k.
            
        if not improved:
            step_size *= 0.7
            if step_size < 1e-6:
                break
        else:
            # If improved, we might increase step size slightly to explore further?
            # Or keep it.
            pass

    return F, best_score

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    F_const = np.fft.fft(best_f)
    best_score = compute_score_fast(F_const)
    
    n_restarts = 100
    half = N // 2
    
    for r in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Create a random box filter initialization
        bandwidth = np.random.randint(10, half)
        F_init = np.zeros(N, dtype=np.complex128)
        
        # DC
        F_init[0] = 1.0
        
        # Random phases
        for k in range(1, bandwidth + 1):
            phase = np.random.uniform(0, 2 * np.pi)
            # Magnitude: uniform for "box"
            F_init[k] = np.exp(1j * phase)
            F_init[N - k] = np.conj(F_init[k])
            
        # Run coordinate descent
        F_opt, score_opt = coordinate_descent(F_init, start_time, TIME_LIMIT - 2)
        
        # Convert to time domain
        f_opt = np.fft.ifft(F_opt).real
        f_opt = np.maximum(f_opt, 0.0)
        
        # Verify
        F_verify = np.fft.fft(f_opt)
        score_verify = compute_score_fast(F_verify)
        
        if score_verify < best_score:
            best_score = score_verify
            best_f = f_opt
            
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score_fast(np.fft.fft(f))}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
