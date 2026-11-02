import numpy as np
import time

N = 600
DX = (1.0/2) / N  # 1/1200

def compute_score(h):
    h = np.asarray(h, dtype=np.float64)
    total = np.sum(h) * DX
    if total < 1e-12:
        return 1e10
    nfft = 1
    while nfft < 2*N:
        nfft <<= 1
    F = np.fft.rfft(h, n=nfft)
    auto = np.fft.irfft(F * F.conj(), n=nfft) * DX
    max_auto = np.max(auto)
    return max_auto / total

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    best_h = np.ones(N, dtype=np.float64)
    best_score = compute_score(best_h)
    
    np.random.seed(123)
    
    n_restarts = 15
    max_iters = 30000
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Diverse initializations
        if r == 0:
            h = np.ones(N)
        elif r == 1:
            h = np.cos(2*np.pi*np.arange(N)/N) ** 2 + 0.5
        elif r == 2:
            h = np.abs(np.sin(np.pi*np.arange(N)/N)) + 0.1
        elif r == 3:
            h = np.random.exponential(1.0, N)
        elif r == 4:
            h = np.ones(N) + 0.5*np.cos(4*np.pi*np.arange(N)/N)
        elif r == 5:
            h = np.ones(N) + 0.3*np.sin(6*np.pi*np.arange(N)/N)
        elif r == 6:
            h = np.maximum(0, np.random.normal(1.0, 0.5, N))
        elif r == 7:
            h = np.abs(np.fft.irfft(np.exp(1j*np.random.uniform(0,2*np.pi,N//2+1)))) + 0.1
        else:
            h = np.maximum(0, np.random.normal(1.0, 1.0, N))
        
        h = np.maximum(0, h)
        cur_score = compute_score(h)
        if cur_score < best_score:
            best_score = cur_score
            best_h = h.copy()
        
        T = 0.005
        cooling = (1e-5 / T) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Perturb random subset
            k = np.random.randint(1, 30)
            idx = np.random.choice(N, k, replace=False)
            sigma = 0.05 + 0.1 * T / 0.005
            h_new = h.copy()
            h_new[idx] = np.maximum(0, h[idx] + np.random.normal(0, sigma, k))
            
            new_score = compute_score(h_new)
            d = new_score - cur_score
            
            if d < 0 or np.random.random() < np.exp(-d / max(T, 1e-8)):
                h = h_new
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_h = h.copy()
            
            T *= cooling
    
    # Final local polish: coordinate descent
    if time.time() < deadline - 5:
        h = best_h.copy()
        cur_score = compute_score(h)
        for _ in range(500):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 3:
                    break
                old = h[i]
                for trial in [old*0.9, old*1.1, max(0, old-0.01), old+0.01]:
                    h[i] = trial
                    s = compute_score(h)
                    if s < cur_score:
                        cur_score = s
                        improved = True
                        break
                if not improved:
                    h[i] = old
            if not improved:
                break
        if cur_score < best_score:
            best_score = cur_score
            best_h = h
    
    return list(best_h)
