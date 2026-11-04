"""Rewritten program: minimize max overlap of h with shifted 1-h via spectral shaping."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_SECONDS = 115.0  # safety margin before 120s kill

def _overlap_max(h):
    """
    Compute the maximum over all cyclic shifts k of the overlap between h and 1-h shifted by k.
    Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    We want to minimize the maximum of this over k.
    """
    # h is length N, values in [0,1], sum = N/2
    # For each shift k, compute sum_i h[i] * (1 - h[(i+k)%N])
    # This equals sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    # = N/2 - autocorr(k)
    # So maximizing overlap = minimizing autocorr(k)
    # We want to minimize max_k overlap = minimize max_k (N/2 - autocorr(k))
    # = N/2 - min_k autocorr(k)
    # So we want to maximize min_k autocorr(k)
    
    # Compute autocorrelation via FFT
    # autocorr[k] = sum_i h[i] * h[(i+k)%N]
    # Using FFT: autocorr = ifft(|fft(h)|^2) but need to be careful with circular correlation
    
    H = np.fft.fft(h)
    power = H * np.conj(H)
    autocorr = np.real(np.fft.ifft(power))
    
    # autocorr[k] = sum_i h[i] * h[(i - k) % N] (circular)
    # We want min over k of autocorr[k]
    min_autocorr = np.min(autocorr)
    
    # max overlap = N/2 - min_autocorr
    max_overlap = N / 2.0 - min_autocorr
    return max_overlap

def _objective(params):
    """Objective: max overlap of h with shifted 1-h."""
    h = params
    # Clip to [0,1] and project to sum = N/2
    h = np.clip(h, 0, 1)
    # Project to sum constraint
    s = np.sum(h)
    if s > 0:
        # Simple projection: scale, but that may violate bounds
        # Better: use a soft projection
        # For now, just return a penalty if sum is off
        pass
    return _overlap_max(h)

def _project(params):
    """Project params to feasible set: h in [0,1], sum(h) = N/2."""
    h = np.clip(params, 0, 1)
    s = np.sum(h)
    target = N / 2.0
    if s != 0:
        # If sum is too high, reduce largest values
        # If sum is too low, increase smallest values
        diff = target - s
        if abs(diff) < 1e-12:
            return h
        if diff > 0:
            # Need to add mass; increase values that are below 1
            mask = h < 1.0
            if np.sum(mask) == 0:
                return h
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
        else:
            # Need to remove mass; decrease values that are above 0
            mask = h > 0.0
            if np.sum(mask) == 0:
                return h
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    # Final clip
    h = np.clip(h, 0, 1)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with a multi-harmonic initial guess
    # h[i] = 0.5 + sum_j a_j * cos(2*pi*j*i/N + phi_j)
    # Choose harmonics that spread energy
    
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Try several initializations with different harmonic mixes
    initializations = []
    
    # 1. Pure 0.5
    initializations.append(np.full(N, 0.5))
    
    # 2. Single harmonic
    for k in [1, 2, 3, 5, 7, 11, 13]:
        if k < N // 2:
            phase = 0
            amp = 0.3
            h_init = 0.5 + amp * np.cos(2 * np.pi * k * np.arange(N) / N + phase)
            h_init = np.clip(h_init, 0, 1)
            # Adjust to sum = N/2
            s = np.sum(h_init)
            if s > 0:
                h_init *= (N / 2.0) / s
                h_init = np.clip(h_init, 0, 1)
            initializations.append(h_init)
    
    # 3. Two harmonics
    for k1 in [1, 2, 3]:
        for k2 in [5, 7, 11]:
            if k2 < N // 2:
                amp1 = 0.25
                amp2 = 0.15
                h_init = 0.5 + amp1 * np.cos(2 * np.pi * k1 * np.arange(N) / N) + \
                          amp2 * np.cos(2 * np.pi * k2 * np.arange(N) / N + 0.5)
                h_init = np.clip(h_init, 0, 1)
                s = np.sum(h_init)
                if s > 0:
                    h_init *= (N / 2.0) / s
                    h_init = np.clip(h_init, 0, 1)
                initializations.append(h_init)
    
    # Optimize each initialization
    for h0 in initializations:
        if time.time() > deadline - 5:
            break
        
        # Use projected gradient descent
        h = h0.copy()
        h = _project(h)
        
        lr = 0.05
        for step in range(500):
            if time.time() > deadline - 2:
                break
            
            # Numerical gradient
            eps = 1e-5
            grad = np.zeros(N)
            for i in range(N):
                hp = h.copy()
                hp[i] += eps
                hp = _project(hp)
                hmin = h.copy()
                hmin[i] -= eps
                hmin = _project(hmin)
                grad[i] = (_overlap_max(hp) - _overlap_max(hmin)) / (2 * eps)
            
            # Update
            h_new = h - lr * grad
            h_new = _project(h_new)
            
            score_new = _overlap_max(h_new)
            if score_new < _overlap_max(h):
                h = h_new
                lr = min(lr * 1.05, 0.2)
            else:
                lr *= 0.8
            
            if lr < 1e-6:
                break
        
        score = _overlap_max(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Try scipy minimize on the best candidate
    if time.time() < deadline - 10:
        h0 = best_h.copy()
        # Parameterize with small perturbations
        def obj(params):
            h = _project(params)
            return _overlap_max(h)
        
        res = minimize(obj, h0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6,
                               'adaptive': True})
        if time.time() < deadline:
            h_opt = _project(res.x)
            score_opt = _overlap_max(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
    
    return best_h

# Ensure the function returns a sequence
def construct():
    """Return step heights in [0,1] with sum = N/2."""
    h = _construct_impl()
    return h

def _construct_impl():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with a multi-harmonic initial guess
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Try several initializations with different harmonic mixes
    initializations = []
    
    # 1. Pure 0.5
    initializations.append(np.full(N, 0.5))
    
    # 2. Single harmonics
    for k in [1, 2, 3, 5, 7, 11, 13, 17, 23]:
        if k < N // 2:
            for amp in [0.2, 0.3, 0.4]:
                for phase in [0, 0.5, 1.0]:
                    h_init = 0.5 + amp * np.cos(2 * np.pi * k * np.arange(N) / N + phase)
                    h_init = np.clip(h_init, 0, 1)
                    s = np.sum(h_init)
                    if s > 0:
                        h_init *= (N / 2.0) / s
                        h_init = np.clip(h_init, 0, 1)
                    initializations.append(h_init.copy())
    
    # 3. Two harmonics
    for k1 in [1, 2, 3, 5]:
        for k2 in [7, 11, 13, 17]:
            if k2 < N // 2:
                for amp1 in [0.2, 0.3]:
                    for amp2 in [0.1, 0.2]:
                        h_init = 0.5 + amp1 * np.cos(2 * np.pi * k1 * np.arange(N) / N) + \
                                  amp2 * np.cos(2 * np.pi * k2 * np.arange(N) / N + 0.5)
                        h_init = np.clip(h_init, 0, 1)
                        s = np.sum(h_init)
                        if s > 0:
                            h_init *= (N / 2.0) / s
                            h_init = np.clip(h_init, 0, 1)
                        initializations.append(h_init.copy())
    
    # Optimize each initialization with projected gradient descent
    for h0 in initializations:
        if time.time() > deadline - 5:
            break
        
        h = _project(h0)
        
        lr = 0.02
        for step in range(300):
            if time.time() > deadline - 2:
                break
            
            # Numerical gradient (central difference)
            eps = 1e-4
            grad = np.zeros(N)
            for i in range(N):
                hp = h.copy()
                hp[i] += eps
                hp = _project(hp)
                hmin = h.copy()
                hmin[i] -= eps
                hmin = _project(hmin)
                grad[i] = (_overlap_max(hp) - _overlap_max(hmin)) / (2 * eps)
            
            # Update
            h_new = h - lr * grad
            h_new = _project(h_new)
            
            score_new = _overlap_max(h_new)
            score_curr = _overlap_max(h)
            if score_new < score_curr:
                h = h_new
                lr = min(lr * 1.03, 0.1)
            else:
                lr *= 0.7
            
            if lr < 1e-7:
                break
        
        score = _overlap_max(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Try scipy minimize on the best candidate
    if time.time() < deadline - 10:
        h0 = best_h.copy()
        def obj(params):
            h = _project(params)
            return _overlap_max(h)
        
        try:
            res = minimize(obj, h0, method='Nelder-Mead', 
                          options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-6,
                                   'adaptive': True})
            if time.time() < deadline:
                h_opt = _project(res.x)
                score_opt = _overlap_max(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt
        except Exception:
            pass
    
    return best_h
