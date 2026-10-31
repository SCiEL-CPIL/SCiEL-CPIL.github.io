import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    def autocorr_max_abs(f):
        n = len(f)
        nfft = 2 * n
        F = np.fft.rfft(f, n=nfft)
        acf = np.fft.irfft(F * np.conj(F), n=nfft)[:n]
        return np.max(np.abs(acf))
    
    def eval_score(f):
        s = np.sum(f)
        if abs(s) < 1e-8:
            return np.inf
        return autocorr_max_abs(f) / (s * s)
    
    np.random.seed(42)
    
    best_f = np.ones(N)
    best_score = eval_score(best_f)
    
    # Try multiple random restarts with local search
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Different initializations
        if restart == 0:
            f = np.ones(N)
        elif restart % 3 == 1:
            f = np.random.randn(N)
        else:
            f = np.random.choice([-1, 1], size=N).astype(float)
        
        # Local search
        for it in range(5000):
            if time.time() > deadline:
                break
            
            idx = np.random.randint(N)
            f_new = f.copy()
            if np.random.random() < 0.3:
                f_new[idx] = -f_new[idx]
            else:
                f_new[idx] += np.random.randn() * 0.5
            
            sc = eval_score(f_new)
            if sc < best_score:
                best_score = sc
                best_f = f_new.copy()
                f = f_new
            elif sc < eval_score(f) * 1.01:
                # Accept with some probability (simulated annealing-ish)
                if np.random.random() < 0.1:
                    f = f_new
    
    return best_f
