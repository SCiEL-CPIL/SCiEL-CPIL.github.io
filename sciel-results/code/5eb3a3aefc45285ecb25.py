import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x8 matrix multiplication.
    Attempts to find a rank lower than 28, but falls back to the known rank-28 Strassen solution.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Guaranteed rank-28 solution
    best_result = _strassen_28()
    best_rank = 28
    
    # Try to find lower rank solutions
    # The theoretical lower bound is 4, but 28 is the known rank.
    # We try 27 and 26 quickly.
    for target_rank in [27, 26]:
        if time.time() > deadline:
            break
        
        result = _search_lower_rank(target_rank, start_time, deadline)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
            
    return best_result

def _strassen_28():
    """Returns rank-28 solution using Strassen per 2x2 block."""
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    A_forms = [
        {0: 1, 3: 1},       # M1: a+d
        {2: 1, 3: 1},       # M2: c+d
        {0: 1},             # M3: a
        {3: 1},             # M4: d
        {0: 1, 1: 1},       # M5: a+b
        {2: 1, 0: -1},      # M6: c-a
        {1: 1, 3: -1}       # M7: b-d
    ]
    
    B_forms = [
        {0: 1, 3: 1},       # M1: x0+y1
        {0: 1},             # M2: x0
        {1: 1, 3: -1},      # M3: x1-y1
        {2: 1, 0: -1},      # M4: y0-x0
        {3: 1},             # M5: y1
        {0: 1, 1: 1},       # M6: x0+x1
        {2: 1, 3: 1}        # M7: y0+y1
    ]
    
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # M1, M4, M5, M7
        {2: 1, 4: 1},              # M3, M5
        {1: 1, 3: 1},              # M2, M4
        {0: 1, 1: -1, 2: 1, 5: 1}  # M1, M2, M3, M6
    ]

    for block in range(4):
        k0 = block * 2
        
        for s in range(7):
            r = block * 7 + s
            
            for idx, val in A_forms[s].items():
                i = idx // 2
                j = idx % 2
                U[i * m + j][r] = val
            
            for idx, val in B_forms[s].items():
                j = idx // 2
                k_local = idx % 2
                k_global = k0 + k_local
                V[j * p + k_global][r] = val
            
            for c_idx, coeffs in enumerate(C_coeffs):
                if s in coeffs:
                    coeff = coeffs[s]
                    i = c_idx // 2
                    k_local = c_idx % 2
                    k_global = k0 + k_local
                    W[k_global * n + i][r] = coeff

    return {"U": U, "V": V, "W": W}

def _search_lower_rank(target_rank, start_time, deadline):
    """
    Searches for a rank-target_rank decomposition using a constrained randomized approach.
    Returns None if not found or time limit exceeded.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor T[a,b,c] = 1 if indices consistent, else 0
    # a = i*m + j, b = j*p + k, c = k*n + i
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # We use a fixed seed for reproducibility in the search attempts
    np.random.seed(42)
    
    # Limit attempts to ensure we stay within time budget
    # Each attempt involves einsum which is fast, but we don't want to do millions.
    max_attempts = 2000 
    attempt = 0
    
    while attempt < max_attempts:
        if time.time() > deadline:
            return None
        
        # Initialize with random half-integers
        # Using a small set of values to keep it simple
        vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        U = np.random.choice(vals, size=(nm, target_rank))
        V = np.random.choice(vals, size=(mp, target_rank))
        W = np.random.choice(vals, size=(pn, target_rank))
        
        # Compute decomposed tensor
        # T_decomposed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        T_decomposed = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check if close to target
        # Using a strict tolerance for "exact"
        if np.allclose(T_decomposed, T_target, atol=1e-9):
            # Verify entries are half-integers (they are by construction)
            # Convert to lists for return format
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
        attempt += 1
        
    return None

# EVOLVE-BLOCK-END
