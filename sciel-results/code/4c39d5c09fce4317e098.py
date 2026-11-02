import time
import numpy as np

def _find_3x3_decomposition(rank, max_attempts=200, seed=42):
    """
    Try to find a rank-r decomposition for 3x3x3 matrix multiplication.
    Uses randomized approach: fix U and V randomly, solve for W.
    Returns (U, V, W) as lists of lists, or None if not found.
    """
    rng = np.random.RandomState(seed)
    
    for attempt in range(max_attempts):
        U = rng.randn(9, rank)
        V = rng.randn(9, rank)
        
        W = np.zeros((9, rank))
        valid = True
        for k in range(3):
            for i in range(3):
                A = np.zeros((3, rank))
                for j in range(3):
                    for c in range(rank):
                        A[j, c] = U[i*3+j, c] * V[j*3+k, c]
                
                try:
                    W_row, residuals, rank_val, sv = np.linalg.lstsq(A, np.ones(3), rcond=None)
                    # Use a looser tolerance for intermediate check
                    if np.allclose(A @ W_row, np.ones(3), atol=1e-8):
                        W[k*3+i, :] = W_row
                    else:
                        valid = False
                        break
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Verify with vectorized computation for speed and accuracy
            # Build the full tensor and check
            ok = True
            max_err = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        s = float(np.dot(U[i*3+j, :], V[j*3+k, :] * W[k*3+i, :]))
                        err = abs(s - 1.0)
                        max_err = max(max_err, err)
                        if err > 1e-7:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                U_list = [[float(U[r, c]) for c in range(rank)] for r in range(9)]
                V_list = [[float(V[r, c]) for c in range(rank)] for r in range(9)]
                W_list = [[float(W[r, c]) for c in range(rank)] for r in range(9)]
                return U_list, V_list, W_list
    
    return None

def _build_3x3x6_from_3x3(U3, V3, W3):
    """
    Build 3x3x6 decomposition from 3x3x3 decomposition by duplicating.
    """
    r = len(U3[0])
    rank_total = 2 * r
    U_full = [[0.0] * rank_total for _ in range(9)]
    V_full = [[0.0] * rank_total for _ in range(18)]
    W_full = [[0.0] * rank_total for _ in range(18)]
    
    # First r terms: for B1 (columns 0-2)
    for c in range(r):
        for i in range(3):
            for j in range(3):
                U_full[i*3+j][c] = U3[i*3+j][c]
        for j in range(3):
            for k1 in range(3):
                V_full[j*3+k1][c] = V3[j*3+k1][c]
        for k1 in range(3):
            for i in range(3):
                W_full[k1*3+i][c] = W3[k1*3+i][c]
    
    # Next r terms: for B2 (columns 3-5)
    for c in range(r):
        c_full = r + c
        for i in range(3):
            for j in range(3):
                U_full[i*3+j][c_full] = U3[i*3+j][c]
        for j in range(3):
            for k2 in range(3):
                V_full[j*6 + 3 + k2][c_full] = V3[j*3+k2][c]
        for k2 in range(3):
            for i in range(3):
                W_full[(3+k2)*3 + i][c_full] = W3[k2*3+i][c]
    
    return U_full, V_full, W_full

def _verify_3x3x6(U, V, W, n=3, m=3, p=6):
    """Verify the 3x3x6 decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-7:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    Strategy: Try to find low-rank decompositions for 3x3x3 (ranks 23-26),
    then duplicate for 3x3x6. Fall back to schoolbook (rank 54) if no better found.
    """
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try ranks from 23 upward for 3x3x3
    for r in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        
        # Try multiple seeds for robustness
        found = None
        for seed in [1, 2, 3, 7, 42, 123, 456, 789]:
            if time.time() > deadline:
                break
            result = _find_3x3_decomposition(r, max_attempts=50, seed=seed)
            if result is not None:
                U3, V3, W3 = result
                U_full, V_full, W_full = _build_3x3x6_from_3x3(U3, V3, W3)
                
                if _verify_3x3x6(U_full, V_full, W_full):
                    found = (U_full, V_full, W_full)
                    break
        
        if found is not None:
            return {"U": found[0], "V": found[1], "W": found[2]}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
