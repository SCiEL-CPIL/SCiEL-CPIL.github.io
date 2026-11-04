import numpy as np
import time

N = 50
dx = 0.5 / N

def _compute_ratio(h):
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return -np.inf
    
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    
    l2_sq = np.sum(h * h)
    if l2_sq <= 0:
        return 0.0
    
    # Discrete convolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv <= 0:
        return 0.0
    
    return l2_sq / (dx * l1 * max_conv)

def _find_sidon_set(size, max_iter=1000):
    """
    Find a set of 'size' indices in [0, N-1] such that pairwise sums are distinct.
    This is a greedy randomized search.
    """
    for _ in range(max_iter):
        current_set = []
        used_sums = set()
        
        # Try to build the set
        for _ in range(size):
            found = False
            for _ in range(50):
                p = np.random.randint(0, N)
                if p in current_set:
                    continue
                
                # Check if adding p causes sum collision
                new_sums = set()
                valid = True
                for q in current_set:
                    s = p + q
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.add(s)
                
                if not valid:
                    continue
                
                # Self-sum p+p
                s_self = 2 * p
                if s_self in used_sums:
                    valid = False
                    continue
                new_sums.add(s_self)
                
                if valid:
                    current_set.append(p)
                    used_sums.update(new_sums)
                    found = True
                    break
            
            if not found:
                break
        
        if len(current_set) == size:
            return current_set
            
    return None

def _optimize_gaussian():
    """
    Optimize a Gaussian profile h_k = exp(-alpha * (k - mu)^2).
    We fix mu to the center and optimize alpha.
    """
    k = np.arange(N)
    mu = (N - 1) / 2.0
    
    def ratio_alpha(alpha):
        h = np.exp(-alpha * (k - mu) ** 2)
        return _compute_ratio(h)
    
    # Golden section search for alpha in [0.01, 5.0]
    # A small alpha is wide, large alpha is narrow.
    # Narrow pulses have high L2/L1 but low conv max? 
    # Wide pulses have low L2/L1 but high conv max?
    # We want to balance.
    
    a = 0.01
    b = 5.0
    gr = (np.sqrt(5) - 1) / 2
    
    # Initial points
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    
    fc = ratio_alpha(c)
    fd = ratio_alpha(d)
    
    # Maximize
    for _ in range(50):
        if fc < fd:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = ratio_alpha(d)
        else:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = ratio_alpha(c)
            
    best_alpha = (a + b) / 2.0
    h = np.exp(-best_alpha * (k - mu) ** 2)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_h = None
    best_val = -np.inf
    
    # 1. Try Sidon sets
    for size in [8, 7, 6, 5, 4, 3, 2, 1]:
        if time.time() > deadline - 10:
            break
            
        sidon_set = _find_sidon_set(size, max_iter=200)
        
        if sidon_set is None:
            continue
            
        h = np.zeros(N)
        for p in sidon_set:
            h[p] = 1.0
            
        val = _compute_ratio(h)
        
        if val > best_val:
            best_val = val
            best_h = h.copy()
            
        if size >= 7:
            break

    # 2. Try Gaussian optimization
    if time.time() < deadline - 5:
        h_gauss = _optimize_gaussian()
        val_gauss = _compute_ratio(h_gauss)
        if val_gauss > best_val:
            best_val = val_gauss
            best_h = h_gauss

    if best_h is None:
        best_h = np.zeros(N)
        best_h[25] = 1.0
        best_val = _compute_ratio(best_h)

    return [float(v) for v in best_h]

if __name__ == "__main__":
    res = construct()
    h = np.array(res)
    val = _compute_ratio(h)
    print(f"Final score: {val}")
