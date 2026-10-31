import numpy as np
import time

def _compute_score(U):
    U = sorted(set(U))
    if len(U) < 2:
        return 1.0
    M = max(U)
    if M == 0:
        return 1.0
    
    f = np.zeros(M + 1, dtype=np.float64)
    for x in U:
        f[x] = 1.0
    
    n_fft = 1
    while n_fft < 2 * M + 1:
        n_fft *= 2
    
    F = np.fft.fft(f, n=n_fft)
    
    conv = np.fft.ifft(F * F).real
    sumset_size = int(np.sum(conv[:2*M+1] > 0.5))
    
    autocorr = np.fft.ifft(F * np.conj(F)).real
    diff_nonzero = 0
    if autocorr[0] > 0.5:
        diff_nonzero += 1
    for d in range(1, M + 1):
        if autocorr[d] > 0.5:
            diff_nonzero += 1
        if autocorr[n_fft - d] > 0.5:
            diff_nonzero += 1
    
    if sumset_size == 0:
        return 1.0
    
    ratio = diff_nonzero / sumset_size
    if ratio <= 1.0:
        return 1.0
    
    return 1.0 + np.log(ratio) / np.log(2 * M + 1)


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # Try structured initial sets
    for base in [2, 3, 4, 5]:
        for k in range(3, 15):
            U = [0]
            x = 1
            for i in range(k-1):
                x *= base
                if x > 5*10**6:
                    break
                U.append(x)
            if len(U) >= 3:
                s = _compute_score(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
    
    # Local search
    U = best_U[:]
    
    for it in range(50000):
        if time.time() > deadline - 20:
            break
        
        U = sorted(set(U))
        M = max(U)
        if M > 5*10**6:
            break
        
        cur = _compute_score(U)
        if cur > best_score:
            best_score = cur
            best_U = U[:]
        
        r = np.random.random()
        U_new = U[:]
        ok = False
        
        if r < 0.4 and len(U) < 200:
            if np.random.random() < 0.3:
                nx = np.random.randint(M+1, min(2*M+100, 5*10**6+1))
            else:
                idx = np.random.randint(len(U))
                sp = max(1, M//20)
                nx = U[idx] + np.random.randint(-sp, sp+1)
            if 0 < nx <= 5*10**6 and nx not in U_new:
                U_new.append(nx)
                U_new = sorted(set(U_new))
                ok = True
        elif r < 0.6 and len(U) > 4:
            idx = np.random.randint(1, len(U))
            U_new.pop(idx)
            U_new = sorted(set(U_new))
            ok = True
        else:
            idx = np.random.randint(1, len(U))
            sp = max(1, M//30)
            nx = U[idx] + np.random.randint(-sp, sp+1)
            if 0 < nx <= 5*10**6 and nx not in U_new:
                U_new[idx] = nx
                U_new = sorted(set(U_new))
                ok = True
        
        if ok:
            sn = _compute_score(U_new)
            if sn >= cur or np.random.random() < 0.03:
                U = U_new
    
    return best_U
