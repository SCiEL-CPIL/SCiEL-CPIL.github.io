import time
import numpy as np
from scipy.fft import fft, ifft
from scipy.optimize import minimize

N = 100
DEADLINE_S = 145.0

def _time_remaining():
    return DEADLINE_S - time.time()

def _max_overlap_binary(h):
    """Compute max overlap for binary h."""
    # C_h(s) = sum h_i h_{i-s}
    H = fft(h)
    C = np.real(ifft(np.conj(H) * H)) # Autocorrelation
    # C[0] is sum h_i^2 = N/2 = 50
    # We care about s != 0
    C[0] = -np.inf
    min_C = np.min(C)
    return 0.5 * N - min_C

def _binary_local_search(h0, time_limit):
    """Simulated annealing / local search on binary sequence."""
    h = h0.copy()
    best_h = h.copy()
    best_val = _max_overlap_binary(h)
    current_val = best_val
    
    # Precompute shift indices? Or just use FFT for evaluation.
    # FFT evaluation is O(N log N). Flipping is O(N) to update h, then O(N log N) to eval.
    # This is fast enough.
    
    t_start = time.time()
    temp = 1.0
    cooling = 0.999
    
    while _time_remaining() > time_limit:
        # Pick a random bit to flip
        j = np.random.randint(N)
        h[j] = 1 - h[j]
        
        # Check balance: sum must be N/2
        # If we flip, sum changes by +/- 1.
        # We need to flip another bit to restore balance?
        # Or just reject if sum != N/2.
        # Better: pick two bits j, k with h[j]!=h[k] and swap them.
        
        # Let's do swap strategy.
        h[j] = 1 - h[j] # Undo
        # Pick j where h[j]=1 and k where h[k]=0
        ones = np.where(h == 1)[0]
        zeros = np.where(h == 0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        j = ones[np.random.randint(len(ones))]
        k = zeros[np.random.randint(len(zeros))]
        
        # Swap
        h[j] = 0
        h[k] = 1
        
        new_val = _max_overlap_binary(h)
        
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Accept with probability exp(-(new_val - current_val) / temp)
            # Since values are integers, delta is integer.
            delta = new_val - current_val
            if delta > 0:
                prob = np.exp(-delta / temp)
                if np.random.rand() < prob:
                    current_val = new_val
                else:
                    # Revert
                    h[j] = 1
                    h[k] = 0
            else:
                # Should not happen if logic is correct, but handle just in case
                current_val = new_val
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    return best_h, best_val

def _soft_min_obj_grad(h, beta):
    H = fft(h)
    F = H * (1.0 - np.exp(-2j * np.pi * np.arange(N) / N))
    Ov = np.real(ifft(F))
    
    Ov_max = np.max(Ov)
    exp_Ov = np.exp(beta * (Ov - Ov_max))
    sum_exp = np.sum(exp_Ov)
    w = exp_Ov / sum_exp
    
    obj_val = Ov_max + np.log(sum_exp) / beta
    
    W = fft(w)
    H1 = fft(1.0 - h)
    term1 = np.real(ifft(W * H1))
    term2 = np.real(ifft(W * np.conj(H)))
    
    grad = term1 - term2
    return obj_val, grad

def _polish_continuous(h0, time_limit):
    h = h0.copy()
    
    def func(h_flat):
        val, _ = _soft_min_obj_grad(h_flat, 100.0)
        return val
        
    def jac(h_flat):
        _, g = _soft_min_obj_grad(h_flat, 100.0)
        return g
        
    bounds = [(0.0, 1.0)] * N
    try:
        res = minimize(func, h, method='L-BFGS-B', jac=jac, bounds=bounds,
                       options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5})
        h_opt = res.x
    except Exception:
        h_opt = h
        
    h_opt = np.clip(h_opt, 0.0, 1.0)
    # Fix sum
    diff = h_opt.sum() - 0.5 * N
    if diff > 0:
        order = np.argsort(h_opt)[::-1]
        for i in order:
            take = min(diff, h_opt[i])
            h_opt[i] -= take
            diff -= take
            if diff <= 1e-12: break
    else:
        diff = -diff
        order = np.argsort(h_opt)
        for i in order:
            take = min(diff, 1.0 - h_opt[i])
            h_opt[i] += take
            diff -= take
            if diff <= 1e-12: break
            
    return h_opt

def construct():
    t0 = time.time()
    
    # 1. Generate initial binary candidates
    candidates = []
    
    # Random
    h_rand = np.zeros(N)
    h_rand[:N//2] = 1
    np.random.shuffle(h_rand)
    candidates.append(h_rand)
    
    # Sawtooth
    h_saw = np.zeros(N)
    h_saw[::4] = 1
    h_saw[1::4] = 1
    candidates.append(h_saw)
    
    # Spectral (continuous)
    EPS = 1e-4
    A = 0.25 * N - (N - 3) * EPS
    mags = np.full(N, EPS)
    mags[0] = 0.5 * N
    mags[1] = A
    mags[N - 1] = A
    H = mags * np.exp(1j * 2 * np.pi * np.arange(N) / N)
    h_spec = np.real(ifft(H))
    h_spec = np.clip(h_spec, 0.0, 1.0)
    diff = h_spec.sum() - 0.5 * N
    if diff > 0:
        order = np.argsort(h_spec)[::-1]
        for i in order:
            take = min(diff, h_spec[i])
            h_spec[i] -= take
            diff -= take
            if diff <= 1e-12: break
    else:
        diff = -diff
        order = np.argsort(h_spec)
        for i in order:
            take = min(diff, 1.0 - h_spec[i])
            h_spec[i] += take
            diff -= take
            if diff <= 1e-12: break
    candidates.append(h_spec)
    
    best_h = candidates[0]
    best_val = _max_overlap_binary(candidates[0]) if np.all((candidates[0]==0)|(candidates[0]==1)) else 100.0
    
    # 2. Binary Local Search on binary candidates
    for h_cand in candidates:
        if _time_remaining() < 10.0:
            break
        if np.all((h_cand==0)|(h_cand==1)):
            h_opt, v_opt = _binary_local_search(h_cand, 5.0)
            if v_opt < best_val:
                best_val = v_opt
                best_h = h_opt
    
    # 3. Continuous Polish on the best binary candidate
    if _time_remaining() > 5.0:
        h_polished = _polish_continuous(best_h, 5.0)
        # Evaluate
        H = fft(h_polished)
        F = H * (1.0 - np.exp(-2j * np.pi * np.arange(N) / N))
        Ov = np.real(ifft(F))
        v_polished = np.max(Ov)
        
        if v_polished < best_val:
            best_val = v_polished
            best_h = h_polished
    
    return best_h
