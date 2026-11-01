"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T with shape (n*p, n*m, m*p).
    T[output, A, B] = 1 if indices are compatible, else 0.
    output index: i*p + k (row i, col k of C)
    A index: i*m + j (row i, col j of A)
    B index: j*p + k (row j, col k of B)
    Compatible if: A row == output row, A col == B row, B col == output col
    """
    T = np.zeros((n * p, n * m, m * p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                out_idx = i * p + k
                a_idx = i * m + j
                b_idx = j * p + k
                T[out_idx, a_idx, b_idx] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Verify that sum_c U[o,c]*V[a,c]*W[b,c] == T[o,a,b] for all o,a,b."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Contract: result[o,a,b] = sum_c U[o,c] * V[a,c] * W[b,c]
    # Use einsum: 'oc,ac,bc->oab'
    result = np.einsum('oc,ac,bc->oab', U, V, W)
    return np.allclose(result, T, atol=tol)

def _round_to_half_int(x, tol=1e-6):
    """Round to nearest half-integer if close, else return x."""
    rounded = np.round(2 * x) / 2.0
    if np.allclose(x, rounded, atol=tol):
        return rounded
    return x

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Build the tensor
    T = _build_tensor(n, m, p)
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try to find lower rank decompositions
    # Strategy: For a given rank r, we need U (np x r), V (nm x r), W (mp x r)
    # such that T[o,a,b] = sum_c U[o,c]*V[a,c]*W[b,c]
    
    # Try ranks from 24 up to 35 (skip 36 since we have schoolbook)
    for r in range(24, rank_school):
        if time.time() > deadline - 10:
            break
        
        # Randomized approach: try multiple random initializations
        found = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values in {-1, 0, 1} or {-0.5, 0, 0.5, 1}
            vals = [-1, -0.5, 0, 0.5, 1]
            U = np.array([[np.random.choice(vals) for _ in range(r)] for _ in range(n * p)])
            V = np.array([[np.random.choice(vals) for _ in range(r)] for _ in range(n * m)])
            W = np.array([[np.random.choice(vals) for _ in range(r)] for _ in range(m * p)])
            
            if _verify(U, V, W, T):
                found = True
                if r < best_rank:
                    best_rank = r
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
        
        if found:
            # Found a decomposition at this rank, try lower ranks
            continue
    
    # Convert to lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
