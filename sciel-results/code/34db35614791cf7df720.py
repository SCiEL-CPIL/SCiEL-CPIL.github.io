import numpy as np
from itertools import product


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    best_P = -1.0
    best_a, best_c, best_M = 0.0, 0.0, 2.0
    
    M_values = np.concatenate([
        np.linspace(1.0, 2.0, 30),
        np.linspace(2.0, 5.0, 30),
        np.linspace(5.0, 20.0, 30),
        np.linspace(20.0, 100.0, 30),
        np.array([200.0, 500.0, 1000.0, 5000.0, 10000.0])
    ])
    
    n_grid = 400
    
    pts = np.array([0.0, 1.0])  # base points, M added separately
    
    for M in M_values:
        if M <= 1.0 + 1e-9:
            continue
        s = np.array([0.0, 1.0, M])
        # Precompute which (i,j,k,l) satisfy s_i+s_j+s_k < 2*s_l
        # P = sum_{i,j,k,l: cond} w_i w_j w_k w_l
        # This is a degree-4 polynomial in (a,b,c)
        # Coefficient of a^i b^j c^k is the number of (i,j,k,l) with the right index counts and condition
        
        # Actually, let's just compute P for each (a,c) directly but vectorized
        a_vals = np.linspace(0.0, 1.0, n_grid)
        best_local_P = -1.0
        best_local_a, best_local_c = 0.0, 0.0
        
        for a in a_vals:
            c_max = 1.0 - a
            if c_max < 0:
                continue
            c_vals = np.linspace(0.0, c_max, n_grid)
            b_vals = 1.0 - a - c_vals
            # Filter valid
            valid = b_vals >= -1e-12
            if not np.any(valid):
                continue
            c_vals = c_vals[valid]
            b_vals = np.maximum(b_vals[valid], 0.0)
            
            wts = np.stack([np.full_like(c_vals, a), b_vals, c_vals], axis=1)
            # Shape: (n_c, 3)
            
            # Compute P = sum_{i,j,k,l} w_i w_j w_k w_l * I(s_i+s_j+s_k < 2 s_l)
            # Vectorize: for each l, compute sum_{i,j,k} w_i w_j w_k * I(s_i+s_j+s_k < 2 s_l) * w_l
            P_vals = np.zeros(len(c_vals))
            for l in range(3):
                # sum_{i,j,k} w_i w_j w_k * I(s_i+s_j+s_k < 2 s_l)
                # This is a function of the weights
                inner = 0.0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            if s[i] + s[j] + s[k] < 2.0 * s[l]:
                                inner += wts[:, i] * wts[:, j] * wts[:, k]
                P_vals += inner * wts[:, l]
            
            max_idx = np.argmax(P_vals)
            if P_vals[max_idx] > best_local_P:
                best_local_P = P_vals[max_idx]
                best_local_a = a
                best_local_c = c_vals[max_idx]
        
        if best_local_P > best_P:
            best_P = best_local_P
            best_a = best_local_a
            best_c = best_local_c
            best_M = M
    
    a = best_a
    c = best_c
    b = 1.0 - a - c
    if b < 0:
        b = 0.0
    
    support = np.array([0.0, 1.0, best_M], dtype=np.float64)
    weights = np.array([a, b, c], dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    
    return support, weights
