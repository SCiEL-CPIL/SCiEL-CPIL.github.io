import numpy as np

N = 400

def construct():
    # 1. Generate random anti-symmetric base
    # a[i] = -a[N-1-i]
    # For even N, all pairs are distinct.
    rng = np.random.default_rng(42)
    a_half = rng.standard_normal(N // 2)
    a = np.zeros(N)
    a[:N//2] = a_half
    a[N//2:] = -a_half[::-1]
    
    # 2. Optimize the bias b
    # h = a + b
    # c_k(b) = c_k_a + 2b * s_k + b^2 * m_k
    # where c_k_a = conv(a, a), s_k = conv(a, ones), m_k = conv(ones, ones)
    
    c_a = np.convolve(a, a)
    ones = np.ones(N)
    s_a = np.convolve(a, ones)
    m_ones = np.convolve(ones, ones)
    
    # We want to minimize max_k (c_a[k] + 2b s_a[k] + b^2 m_ones[k])
    # m_ones[k] is a triangle: 1, 2, ..., N, ..., 2, 1
    
    # Use scipy minimize to find optimal b
    from scipy.optimize import minimize_scalar
    
    def obj(b):
        c_b = c_a + 2 * b * s_a + (b ** 2) * m_ones
        return np.max(c_b)
        
    # Search range for b
    # b should be small relative to std(a)
    # std(a) ~ 1
    res = minimize_scalar(obj, bounds=(-2, 2), method='bounded')
    best_b = res.x
    
    h = a + best_b
    
    # 3. Refine by local gradient descent on h?
    # The bias optimization is exact for fixed a.
    # We can try a few random restarts for a.
    
    best_val = np.inf
    best_h = h
    
    for restart in range(10):
        rng = np.random.default_rng(restart)
        a_half = rng.standard_normal(N // 2)
        a = np.zeros(N)
        a[:N//2] = a_half
        a[N//2:] = -a_half[::-1]
        
        c_a = np.convolve(a, a)
        s_a = np.convolve(a, ones)
        
        def obj(b):
            c_b = c_a + 2 * b * s_a + (b ** 2) * m_ones
            return np.max(c_b)
            
        res = minimize_scalar(obj, bounds=(-2, 2), method='bounded')
        b = res.x
        h_candidate = a + b
        
        # Calculate actual value
        s = np.sum(h_candidate)
        if s == 0: continue
        c_final = np.convolve(h_candidate, h_candidate)
        max_c = np.max(c_final)
        val = abs(2 * N * max_c / (s ** 2))
        
        if val < best_val:
            best_val = val
            best_h = h_candidate
            
    return best_h
