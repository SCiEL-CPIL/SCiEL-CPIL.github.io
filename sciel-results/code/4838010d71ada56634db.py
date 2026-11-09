import numpy as np
import time

N = 100

def _score(h):
    # Calculate overlap using LINEAR correlation
    # Overlap_k = sum_{i=0}^{N-1-k} h_i * (1 - h_{i+k})
    # We want to minimize max overlap.
    # Overlap_k = (N-k)*0.5 - sum h_i h_{i+k}
    # Let R_k = sum h_i h_{i+k}
    # Minimize max_k (0.5*(N-k) - R_k)
    
    g = h - 0.5
    # Linear autocorrelation of g
    # R_k = sum_{i=0}^{N-1-k} g_i g_{i+k}
    # Note: h_i h_{i+k} = (0.5+g_i)(0.5+g_{i+k}) = 0.25 + 0.5(g_i+g_{i+k}) + g_i g_{i+k}
    # Sum h_i h_{i+k} = 0.25(N-k) + 0.5 sum(g_i+g_{i+k}) + R_k^g
    # Sum(g_i+g_{i+k}) = sum_{i=0}^{N-1-k} g_i + sum_{i=0}^{N-1-k} g_{i+k}
    # This term is not zero unless sum g is zero and we are careful with boundaries.
    # But sum g = 0 is enforced.
    # Let's just compute directly to be safe.
    
    max_overlap = 0
    for k in range(N):
        # h_shifted is 1-h shifted by k?
        # "overlap of h with shifted 1-h"
        # Usually means sum h_i * (1-h)_{i+k}
        # (1-h)_{i+k} = 1 - h_{i+k}
        # Term = h_i * (1 - h_{i+k})
        
        # Linear shift: indices i and i+k must be in [0, N-1]
        # i ranges from 0 to N-1-k
        h_part = h[:N-k]
        h_shifted = h[k:]
        overlap_k = np.sum(h_part * (1 - h_shifted))
        if overlap_k > max_overlap:
            max_overlap = overlap_k
            
    return max_overlap

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    # Initial best
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Random search
    np.random.seed(42)
    
    # Try Gaussian perturbations
    # We want to maximize min R_k (linear), which minimizes max overlap.
    
    for _ in range(5000):
        if time.time() > deadline:
            break
            
        # Generate random g
        # Start with small noise to explore local minima
        # Or large noise to jump?
        # Let's mix.
        
        # Method 1: Add Gaussian noise to current best
        if np.random.rand() < 0.5:
            noise = np.random.randn(N) * 0.05
        else:
            noise = np.random.randn(N) * 0.2
            
        g = best_h - 0.5 + noise
        # Enforce sum 0
        g -= np.mean(g)
        # Clip h to [0, 1]
        h = 0.5 + g
        h = np.clip(h, 0, 1)
        
        # Re-enforce sum exactly N/2?
        # Clipping might change sum.
        # We need sum(h) = N/2.
        # Adjust one element? Or scale?
        # Scaling might push out of bounds.
        # Simple fix: adjust the last element.
        current_sum = np.sum(h)
        diff = N/2 - current_sum
        h[-1] += diff
        # If h[-1] is out of bounds, we need a more robust fix.
        # But for small noise, it should be fine.
        if h[-1] < 0 or h[-1] > 1:
            # Fallback: distribute diff evenly
            h += diff / N
            h = np.clip(h, 0, 1)
            # This might break sum again. 
            # For robustness, if this happens, skip or accept with penalty?
            # Let's just accept if close enough, or skip.
            # Actually, the constraint is strict.
            # Let's use a projection method.
            # For now, if invalid, skip.
            if np.sum(h) != N/2:
                 continue

        score = _score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    return best_h
