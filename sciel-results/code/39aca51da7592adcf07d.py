import numpy as np
import time
from itertools import product

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate candidate vectors: all vectors with entries in {-1, 0, 1}
    # with small L2 norm, prioritized by norm
    candidates = []
    
    # Type A: single coordinate (norm 1)
    for i in range(D):
        for s in (1, -1):
            v = [0]*D
            v[i] = s
            candidates.append(v)
    
    # Type B: two coordinates (norm sqrt(2))
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0]*D
                    v[i], v[j] = si, sj
                    candidates.append(v)
    
    # Type C: three coordinates (norm sqrt(3))
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0]*D
                            v[i], v[j], v[k] = si, sj, sk
                            candidates.append(v)

    # Sort candidates by norm to prioritize smaller norms (often easier to fit)
    # Note: Greedy order can affect result. Smaller norms generally have smaller "footprint".
    def get_norm_sq(v):
        return sum(x*x for x in v)
    
    candidates.sort(key=get_norm_sq)

    selected = []
    selected_norms_sq = [] # Store squared norms for speed
    
    for v in candidates:
        if time.time() > deadline:
            break
        v = np.array(v)
        v_norm_sq = float(np.dot(v, v))
        
        # Check compatibility with all selected
        compatible = True
        for idx, u in enumerate(selected):
            dot = np.dot(v, u)
            u_norm_sq = selected_norms_sq[idx]
            
            # Condition: dist^2 >= max(norm_v^2, norm_u^2)
            # norm_v^2 + norm_u^2 - 2*dot >= max(norm_v^2, norm_u^2)
            # Let A = norm_v^2, B = norm_u^2
            # A + B - 2*dot >= max(A, B)
            # If A >= B: A + B - 2*dot >= A => B >= 2*dot => dot <= B/2
            # If B > A: A + B - 2*dot >= B => A >= 2*dot => dot <= A/2
            # So dot <= min(A, B) / 2
            
            min_norm_sq = min(v_norm_sq, u_norm_sq)
            if dot > min_norm_sq / 2.0 + 1e-9:
                compatible = False
                break
        
        if compatible:
            selected.append(v)
            selected_norms_sq.append(v_norm_sq)
    
    if not selected:
        # Fallback: return a single valid vector if greedy failed (shouldn't happen)
        return np.array([[1] + [0]*10], dtype=np.int64)
        
    return np.array(selected, dtype=np.int64)
