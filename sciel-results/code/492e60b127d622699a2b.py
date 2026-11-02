import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.array(h)
    if np.sum(h) == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    # Compute autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv == 0:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    best_score = 0.0
    best_h = [1.0] * N
    
    # Try various structured approaches
    # 1. Two-element support optimization
    # For h = [a, b, 0, ...], ratio = (a^2+b^2)/(a+b) / max(a^2, 2ab, b^2)
    # By symmetry, assume a >= b
    # If a^2 >= 2ab (i.e., a >= 2b), ratio = (a^2+b^2)/(a+b)/a^2 = (1+(b/a)^2)/(1+b/a)
    # Let t = b/a, ratio = (1+t^2)/(1+t)
    # Maximize over t in [0, 0.5]: derivative of (1+t^2)/(1+t) = (2t(1+t)-(1+t^2))/(1+t)^2 = (2t+2t^2-1-t^2)/(1+t)^2 = (t^2+2t-1)/(1+t)^2
    # Zero at t = -1+sqrt(2) ≈ 0.414
    # Check if t=0.414 is in [0,0.5]: yes
    # At t=0.414: ratio = (1+0.1716)/(1.414) = 1.1716/1.414 ≈ 0.828
    # But wait, we need to check the case where 2ab > a^2, i.e., 2b > a, i.e., t > 0.5
    # If 2ab >= a^2 and 2ab >= b^2 (always true if a>=b), then sup = 2ab
    # ratio = (a^2+b^2)/(a+b)/2ab = (a^2+b^2)/(2ab(a+b))
    # Let t = b/a, ratio = (1+t^2)/(2t(1+t))
    # For t in [0.5, 1]: at t=0.5, ratio = (1+0.25)/(1*1.5) = 1.25/1.5 = 0.833
    # At t=1, ratio = 2/4 = 0.5
    # So max in this region is at t=0.5, giving 0.833
    
    # Hmm, but the observed score was 1.1944 for [2/3, 1/3], which is t=0.5
    # My computed ratio is 0.833, but score is 1.1944. There's a factor of ~1.434
    
    # Let me just do numerical optimization
    
    # Try different support sizes
    for support_size in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        if time.time() > deadline:
            break
        
        # Initialize with a triangular or uniform distribution over the support
        h_init = np.zeros(N)
        for i in range(support_size):
            h_init[i] = 1.0 / support_size
        
        # Optimize using scipy
        def objective(x):
            h = np.zeros(N)
            h[:support_size] = np.abs(x)  # enforce nonnegativity
            return -compute_ratio(h)
        
        try:
            result = minimize(objective, h_init[:support_size], method='Nelder-Mead', 
                             options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            h_opt = np.zeros(N)
            h_opt[:support_size] = np.abs(result.x)
            score = compute_ratio(h_opt)
            
            if score > best_score:
                best_score = score
                best_h = list(h_opt)
        except Exception:
            pass
    
    # Also try some hand-crafted patterns
    # Uniform on a block
    for block_size in range(1, 25):
        if time.time() > deadline:
            break
        h = np.zeros(N)
        h[:block_size] = 1.0
        score = compute_ratio(h)
        if score > best_score:
            best_score = score
            best_h = list(h)
    
    # Triangular
    for peak_pos in range(0, N, 2):
        if time.time() > deadline:
            break
        h = np.zeros(N)
        for i in range(N):
            h[i] = max(0, 1 - abs(i - peak_pos) / 10.0)
        if np.sum(h) > 0:
            score = compute_ratio(h)
            if score > best_score:
                best_score = score
                best_h = list(h)
    
    # Return the best found
    return best_h
