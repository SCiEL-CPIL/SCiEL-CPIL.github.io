import numpy as np


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Find optimal 3-point discrete measure on {0, 1, M} maximizing
    P(X1+X2+X3 < 2*X4) where X_i are i.i.d. from the measure.
    
    Uses analytical insight: for fixed M, the probability is a polynomial
    in the weights (a, b, c) where a=P(X=0), b=P(X=1), c=P(X=M), a+b+c=1.
    We do a 2D grid search over M and b (probability mass at 1), then solve
    for optimal a,c given M and b.
    """
    
    # Precompute the indicator conditions for each M
    # For points s = [0, 1, M], we need to check s_i + s_j + s_k < 2*s_l
    # There are 3^4 = 81 combinations
    
    # Strategy: For each M, compute P as a function of (a, b, c) with a+b+c=1
    # P = sum_{i,j,k,l: s_i+s_j+s_k < 2*s_l} w_i w_j w_k w_l
    # Since a = 1 - b - c, P becomes a cubic polynomial in (b, c)
    
    # For fixed M, let's enumerate all (i,j,k,l) tuples and their conditions
    # Then P = sum of terms where each term is a product of weights
    
    # Let's be more efficient: for fixed M, precompute which (i,j,k,l) satisfy the condition
    # Then P is a sum of monomials in (a, b, c)
    
    # For a given M, the condition s_i + s_j + s_k < 2*s_l depends on M
    # Let's categorize:
    # s = [0, 1, M]
    # 2*s_l for l=0: 0, l=1: 2, l=2: 2M
    # s_i+s_j+s_k can be: 0, 1, M, 2, 1+M, 2M, 3, 2+M, 1+2M, 3M
    
    # Let's just do a fine grid search over M and b, and for each compute optimal (a,c)
    
    # Actually, let's think more carefully.
    # For fixed M, P(a,b,c) is a polynomial. Since a = 1-b-c, we can write P as a function of (b,c).
    # This is a cubic polynomial in 2 variables over the triangle b>=0, c>=0, b+c<=1.
    
    # The maximum of a cubic polynomial over a triangle occurs either at critical points
    # (where gradient = 0) or on the boundary.
    
    # For simplicity and robustness, let's do a dense grid search.
    
    # Grid over M: logarithmic spacing from ~1 to large values
    M_values = np.concatenate([
        np.linspace(1.001, 2.0, 50),
        np.linspace(2.0, 10.0, 50),
        np.linspace(10.0, 100.0, 50),
        np.linspace(100.0, 1000.0, 30),
        np.array([5000.0, 10000.0, 50000.0, 100000.0])
    ])
    
    n_b = 200  # grid points for b
    n_c = 200  # grid points for c
    
    best_P = -1.0
    best_a, best_b, best_c, best_M = 0.0, 0.0, 0.0, 2.0
    
    for M in M_values:
        if M <= 1.0 + 1e-9:
            continue
        
        s = np.array([0.0, 1.0, M])
        
        # Precompute the coefficient matrix: for each (i,j,k,l), if condition holds,
        # add to the appropriate monomial coefficient
        # P = sum_{i,j,k,l: cond} w_i w_j w_k w_l
        # = sum over all monomials a^alpha b^beta c^gamma with alpha+beta+gamma=4
        
        # Let's compute P for a grid of (b, c) values
        b_vals = np.linspace(0.0, 1.0, n_b)
        
        best_local_P = -1.0
        best_local_b, best_local_c = 0.0, 0.0
        
        for b in b_vals:
            c_max = 1.0 - b
            if c_max < 1e-12:
                continue
            c_vals = np.linspace(0.0, c_max, n_c)
            a_vals = 1.0 - b - c_vals
            
            # Filter valid
            valid = (a_vals >= -1e-12) & (c_vals >= -1e-12)
            if not np.any(valid):
                continue
            a_vals = np.maximum(a_vals[valid], 0.0)
            c_vals = c_vals[valid]
            
            # Compute P for each (a, b, c)
            # P = sum_{i,j,k,l: s_i+s_j+s_k < 2*s_l} w_i w_j w_k w_l
            P_vals = np.zeros(len(a_vals))
            
            for l in range(3):
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            if s[i] + s[j] + s[k] < 2.0 * s[l] - 1e-15:
                                w_i = np.where(i == 0, a_vals, np.where(i == 1, b, c_vals))
                                w_j = np.where(j == 0, a_vals, np.where(j == 1, b, c_vals))
                                w_k = np.where(k == 0, a_vals, np.where(k == 1, b, c_vals))
                                w_l = np.where(l == 0, a_vals, np.where(l == 1, b, c_vals))
                                P_vals += w_i * w_j * w_k * w_l
            
            max_idx = np.argmax(P_vals)
            if P_vals[max_idx] > best_local_P:
                best_local_P = P_vals[max_idx]
                best_local_b = b
                best_local_c = c_vals[max_idx]
        
        if best_local_P > best_P:
            best_P = best_local_P
            best_a = 1.0 - best_local_b - best_local_c
            best_b = best_local_b
            best_c = best_local_c
            best_M = M
    
    # Refine around the best found point
    # Do a finer search in a small neighborhood
    M_range = max(best_M * 0.9, 1.001)
    M_range2 = best_M * 1.1
    
    M_fine = np.linspace(M_range, M_range2, 50)
    b_center = best_b
    c_center = best_c
    
    best_P_fine = best_P
    best_a_fine, best_b_fine, best_c_fine, best_M_fine = best_a, best_b, best_c, best_M
    
    for M in M_fine:
        if M <= 1.0 + 1e-9:
            continue
        
        s = np.array([0.0, 1.0, M])
        
        b_min = max(0.0, best_b - 0.1)
        b_max = min(1.0, best_b + 0.1)
        c_min = max(0.0, best_c - 0.1)
        c_max = min(1.0 - best_b, best_c + 0.1)
        
        n_refine = 50
        b_fine = np.linspace(b_min, b_max, n_refine)
        
        for b in b_fine:
            c_max_local = min(c_max, 1.0 - b)
            c_min_local = max(c_min, 0.0)
            if c_min_local > c_max_local:
                continue
            c_fine = np.linspace(c_min_local, c_max_local, n_refine)
            a_fine_vals = 1.0 - b - c_fine
            
            valid = (a_fine_vals >= -1e-12) & (c_fine >= -1e-12)
            if not np.any(valid):
                continue
            a_fine_vals = np.maximum(a_fine_vals[valid], 0.0)
            c_fine = c_fine[valid]
            
            P_vals = np.zeros(len(a_fine_vals))
            
            for l in range(3):
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            if s[i] + s[j] + s[k] < 2.0 * s[l] - 1e-15:
                                w_i = np.where(i == 0, a_fine_vals, np.where(i == 1, b, c_fine))
                                w_j = np.where(j == 0, a_fine_vals, np.where(j == 1, b, c_fine))
                                w_k = np.where(k == 0, a_fine_vals, np.where(k == 1, b, c_fine))
                                w_l = np.where(l == 0, a_fine_vals, np.where(l == 1, b, c_fine))
                                P_vals += w_i * w_j * w_k * w_l
            
            max_idx = np.argmax(P_vals)
            if P_vals[max_idx] > best_P_fine:
                best_P_fine = P_vals[max_idx]
                best_a_fine = a_fine_vals[max_idx]
                best_b_fine = b
                best_c_fine = c_fine[max_idx]
                best_M_fine = M
    
    a = best_a_fine
    b = best_b_fine
    c = best_c_fine
    
    # Ensure non-negative and normalized
    a = max(a, 0.0)
    b = max(b, 0.0)
    c = max(c, 0.0)
    total = a + b + c
    if total < 1e-9:
        a, b, c = 1.0, 0.0, 0.0
    else:
        a /= total
        b /= total
        c /= total
    
    support = np.array([0.0, 1.0, best_M_fine], dtype=np.float64)
    weights = np.array([a, b, c], dtype=np.float64)
    
    return support, weights
