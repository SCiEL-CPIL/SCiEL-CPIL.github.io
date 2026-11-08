import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    np.random.seed(42)
    h = np.random.choice([-1, 1], size=N)
    best_h = h.copy()
    
    # Compute initial autocorrelation
    H = np.fft.fft(best_h)
    R = np.fft.ifft(np.abs(H)**2).real
    best_score = np.max(np.abs(np.concatenate([R[1:], R[:0]]))) / N
    
    # Local search
    for iteration in range(5000):
        if (time.time() - start_time) > deadline:
            break
        
        # Find worst lag (excluding 0)
        abs_R = np.abs(R)
        abs_R[0] = 0
        worst_lag = np.argmax(abs_R)
        
        # Try flipping each bit, compute delta in R
        best_delta_score = best_score
        best_flip_idx = -1
        
        for i in range(N):
            # Delta in R when flipping h[i]
            # R(k) changes by -2*h[i]*(h[(i+k)%N] + h[(i-k)%N])
            # But we only care about the max |R(k)| for k != 0
            # This is complex; let's just recompute for the worst lag and nearby
            
            # Actually, let's compute the new R for the worst lag only as a quick check
            # R_new(worst_lag) = R(worst_lag) - 2*h[i]*(h[(i+worst_lag)%N] + h[(i-worst_lag)%N])
            delta_worst = -2 * best_h[i] * (best_h[(i + worst_lag) % N] + best_h[(i - worst_lag) % N])
            new_R_worst = R[worst_lag] + delta_worst
            
            # Quick check: does this reduce |R(worst_lag)|?
            if abs(new_R_worst) < abs_R[worst_lag]:
                # Promising, let's do a full check
                h_trial = best_h.copy()
                h_trial[i] *= -1
                H_trial = np.fft.fft(h_trial)
                R_trial = np.fft.ifft(np.abs(H_trial)**2).real
                score_trial = np.max(np.abs(np.concatenate([R_trial[1:], R_trial[:0]]))) / N
                if score_trial < best_delta_score:
                    best_delta_score = score_trial
                    best_flip_idx = i
        
        if best_flip_idx >= 0:
            best_h[best_flip_idx] *= -1
            H = np.fft.fft(best_h)
            R = np.fft.ifft(np.abs(H)**2).real
            best_score = best_delta_score
        else:
            # Try random flips
            for _ in range(200):
                if (time.time() - start_time) > deadline:
                    break
                idx = np.random.randint(N)
                h_trial = best_h.copy()
                h_trial[idx] *= -1
                H_trial = np.fft.fft(h_trial)
                R_trial = np.fft.ifft(np.abs(H_trial)**2).real
                score_trial = np.max(np.abs(np.concatenate([R_trial[1:], R_trial[:0]]))) / N
                if score_trial < best_score:
                    best_h = h_trial
                    best_score = score_trial
                    H = H_trial
                    R = R_trial
                    break
    
    return best_h
