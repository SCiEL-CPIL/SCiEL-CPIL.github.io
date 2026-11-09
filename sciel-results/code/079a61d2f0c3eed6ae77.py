import numpy as np
import time

def _eval_at(c, theta):
    k = np.arange(len(c))
    return float(np.abs(np.sum(c * np.exp(1j * theta * k))))

def _find_peak(c, M=4096):
    k = np.arange(len(c))
    thetas = np.linspace(0, 2*np.pi, M, endpoint=False)
    Z = np.exp(1j * thetas[:, None] * k[None, :])
    vals = np.abs(Z @ c)
    peak_idx = int(np.argmax(vals))
    dtheta = 2*np.pi / M
    lo = peak_idx * dtheta
    hi = (peak_idx + 1) * dtheta
    # Golden section refinement
    inv_phi = (1 + 5**0.5) / 2
    phi = 1.0 / inv_phi
    a, b = lo, hi
    c1 = b - phi * (b - a)
    c2 = a + phi * (b - a)
    f1 = _eval_at(c, c1)
    f2 = _eval_at(c, c2)
    for _ in range(40):
        if f1 < f2:
            b = c2
            c2 = c1
            f2 = f1
            c1 = b - phi * (b - a)
            f1 = _eval_at(c, c1)
        else:
            a = c1
            c1 = c2
            f1 = f2
            c2 = a + phi * (b - a)
            f2 = _eval_at(c, c2)
    return max(f1, f2)

def _grs(n):
    a = [1] * n
    for i in range(1, n):
        k = i >> 1
        if i & 1:
            a[i] = -a[k]
    return np.array(a, dtype=np.float64)

def construct():
    start = time.time()
    deadline = start + 140.0
    N = 24
    
    # Multiple starting points
    starts = []
    starts.append(_grs(N))
    # Reversed GRS
    starts.append(_grs(N)[::-1].copy())
    # Negated GRS (same peak)
    # Random starts
    rng = np.random.RandomState(42)
    for _ in range(5):
        starts.append(rng.choice([-1.0, 1.0], size=N))
    
    best_c = None
    best_val = float('inf')
    
    for s in starts:
        if time.time() > deadline - 20.0:
            break
        c = s.copy()
        val = _find_peak(c)
        if val < best_val:
            best_val = val
            best_c = c.copy()
    
    # Local search from best
    if time.time() < deadline - 20.0:
        thetas_coarse = np.linspace(0, 2*np.pi, 2048, endpoint=False)
        k_arr = np.arange(N)
        
        def quick_max(trial):
            Z = np.exp(1j * thetas_coarse[:, None] * k_arr[None, :])
            return float(np.max(np.abs(Z @ trial)))
        
        improved = True
        passes = 0
        while improved and passes < 20 and time.time() < deadline - 15.0:
            improved = False
            passes += 1
            cur = _find_peak(best_c)
            
            for j in range(N):
                if time.time() > deadline - 15.0:
                    break
                trial = best_c.copy()
                trial[j] = -trial[j]
                if quick_max(trial) < cur + 0.005:
                    pv = _find_peak(trial)
                    if pv < cur - 1e-10:
                        best_c = trial
                        cur = pv
                        improved = True
            
            if time.time() < deadline - 25.0:
                cur = _find_peak(best_c)
                for j in range(N):
                    for l in range(j+1, N):
                        if time.time() > deadline - 15.0:
                            break
                        trial = best_c.copy()
                        trial[j] = -trial[j]
                        trial[l] = -trial[l]
                        if quick_max(trial) < cur + 0.005:
                            pv = _find_peak(trial)
                            if pv < cur - 1e-10:
                                best_c = trial
                                cur = pv
                                improved = True
    
    out = [1 if v >= 0 else -1 for v in best_c]
    return out
