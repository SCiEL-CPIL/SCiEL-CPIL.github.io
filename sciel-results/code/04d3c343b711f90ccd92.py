import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/1/4]

def _score_raw(h):
    """
    Compute the ratio: L2^2 / (L1 * Sup(AutoConv))
    where AutoConv is the continuous autoconvolution of the step function.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _line_search(h, i, current_score, max_iter=25):
    """
    Golden-section line search along coordinate i.
    """
    h = h.copy()
    
    # Determine bracket [lower, upper]
    if h[i] == 0:
        upper = 0.05
        while upper < 50:
            h[i] = upper
            if _score_raw(h) > current_score:
                upper *= 1.5
            else:
                break
        lower = upper / 1.5
    else:
        lower = 0.0
        upper = h[i] * 2
        h[i] = upper
        if _score_raw(h) < current_score:
            pass
        else:
            while upper < 50:
                upper *= 1.5
                h[i] = upper
                if _score_raw(h) < current_score:
                    break
            lower = upper / 1.5
    
    if lower >= upper:
        return h[i]
    
    gr = (np.sqrt(5) - 1) / 2
    a, b = lower, upper
    
    for _ in range(max_iter):
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        h[i] = c
        s_c = _score_raw(h)
        h[i] = d
        s_d = _score_raw(h)
        
        if s_c > s_d:
            b = d
        else:
            a = c
        
        if b - a < 1e-8 * max(1.0, abs(a)):
            break
    
    best_val = (a + b) / 2
    h[i] = best_val
    return best_val

def _coordinate_descent(h, max_iter=100, tol=1e-10):
    """
    Coordinate descent with line search to maximize L2^2 / (L1 * AutoConvMax).
    """
    h = h.copy()
    h = np.maximum(0, h)
    l1 = np.sum(h)
    if l1 == 0:
        return h
    
    # Normalize to L1=1
    h = h / l1
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            best_val = h[i]
            best_score = _score_raw(h)
            
            new_val = _line_search(h, i, best_score, max_iter=20)
            h[i] = new_val
            s_new = _score_raw(h)
            
            if s_new > best_score + tol:
                improved = True
                best_score = s_new
                best_val = new_val
            else:
                h[i] = best_val
        
        if not improved:
            break
    
    return h

def _sidon_support(N, k):
    """
    Generate a Sidon-like set of size k from {0, ..., N-1}.
    """
    support = []
    used_sums = set()
    
    for _ in range(k):
        best_pos = None
        best_collisions = float('inf')
        
        for pos in range(N):
            if pos in support:
                continue
            new_collisions = 0
            for s in support:
                s_sum = pos + s
                if s_sum in used_sums:
                    new_collisions += 1
            if new_collisions < best_collisions:
                best_collisions = new_collisions
                best_pos = pos
        
        if best_pos is None:
            break
        
        support.append(best_pos)
        for s in support[:-1]:
            used_sums.add(best_pos + s)
        used_sums.add(2 * best_pos)
    
    return sorted(support)

def _exponential_profile(support, decay_rate):
    """
    Create an exponentially decaying profile on the given support.
    """
    h = np.zeros(N)
    for idx, pos in enumerate(support):
        h[pos] = decay_rate ** idx
    return h

def _optimize_exponential(support, decay_range=(0.01, 0.99), n_trials=80):
    """
    Optimize the decay rate for a given support.
    """
    best_score = 0
    best_decay = 0.5
    
    log_rates = np.linspace(np.log(decay_range[0]), np.log(decay_range[1]), n_trials)
    
    for log_r in log_rates:
        r = np.exp(log_r)
        h = _exponential_profile(support, r)
        s = _score_raw(h)
        if s > best_score:
            best_score = s
            best_decay = r
    
    # Refine with golden section search
    if best_score > 0:
        a, b = decay_range[0], decay_range[1]
        gr = (np.sqrt(5) - 1) / 2
        for _ in range(25):
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            h_c = _exponential_profile(support, c)
            s_c = _score_raw(h_c)
            h_d = _exponential_profile(support, d)
            s_d = _score_raw(h_d)
            
            if s_c > s_d:
                b = d
            else:
                a = c
        
        best_decay = (a + b) / 2
        h_best = _exponential_profile(support, best_decay)
        best_score = _score_raw(h_best)
    
    return best_score, best_decay, _exponential_profile(support, best_decay)

def _greedy_support_search(h_init, max_supports=50):
    """
    Greedy search over support sets with exponential profiles.
    """
    best_h = h_init.copy()
    best_score = _score_raw(h_init)
    
    x_idx = np.arange(N)
    
    for k in range(3, min(N, 25)):
        if time.time() > 140:
            break
        
        # Try Sidon-like supports
        sidon = _sidon_support(N, k)
        if len(sidon) >= 3:
            s, decay, h_exp = _optimize_exponential(sidon)
            if s > best_score:
                best_score = s
                best_h = h_exp.copy()
        
        # Try random subsets
        for trial in range(5):
            if time.time() > 140:
                break
            
            subset = np.random.choice(N, k, replace=False)
            h = np.zeros(N)
            h[subset] = 1.0
            h = _coordinate_descent(h, max_iter=20)
            s = _score_raw(h)
            
            if s > best_score:
                best_score = s
                best_h = h.copy()
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Parametric initializations ---
    
    # 1. Uniform
    h_init = np.ones(N)
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # 2. Gaussian
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # 3. Triangular
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()

    # 4. Exponential decay profiles on various supports
    for k in range(3, 15):
        if time.time() > deadline - 40:
            break
        sidon = _sidon_support(N, k)
        if len(sidon) >= 3:
            s, decay, h_exp = _optimize_exponential(sidon)
            if s > best_score:
                best_score = s
                best_h = h_exp.copy()
        
        for start in range(0, N - k + 1, max(1, (N - k) // 5)):
            support = list(range(start, start + k))
            s, decay, h_exp = _optimize_exponential(support)
            if s > best_score:
                best_score = s
                best_h = h_exp.copy()

    # 5. Random sparse initialization
    for _ in range(100):
        if time.time() > deadline - 20: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        h_rand += np.random.rand(N) * 0.1
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # --- Phase 2: Coordinate Descent from best initialization ---
    
    if time.time() < deadline - 30:
        h_cd = _coordinate_descent(best_h, max_iter=150)
        s_cd = _score_raw(h_cd)
        if s_cd > best_score:
            best_score = s_cd
            best_h = h_cd.copy()
    
    # --- Phase 3: Greedy support search ---
    
    if time.time() < deadline - 20:
        h_greedy, s_greedy = _greedy_support_search(best_h, max_supports=30)
        if s_greedy > best_score:
            best_score = s_greedy
            best_h = h_greedy.copy()
    
    # --- Phase 4: Final polish with coordinate descent ---
    
    if time.time() < deadline - 5:
        h_final = _coordinate_descent(best_h, max_iter=200)
        s_final = _score_raw(h_final)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final.copy()
    
    return list(best_h)
