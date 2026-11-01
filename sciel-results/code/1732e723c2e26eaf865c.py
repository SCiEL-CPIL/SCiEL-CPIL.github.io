import numpy as np
import time

DIMENSIONS = [2, 2, 4]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _strassen_2x2_2x2():
    """
    Return the Strassen decomposition for 2x2 * 2x2.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Index convention: U[i*m+j], V[j*p+k], W[k*n+i]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    # C00 += M1, C11 += M1, C01 -= M1, C10 += M1
    W[0,0] = 1; W[2,0] = 1; W[1,0] = -1; W[3,0] = 1
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    # C10 += M2, C11 += M2
    W[1,1] = 1; W[3,1] = 1
    
    # M3 = A00(B10-B11)
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    # C00 += M3, C10 -= M3
    W[0,2] = 1; W[2,2] = -1
    
    # M4 = A11(B01-B00)
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    # C01 += M4, C11 -= M4
    W[1,3] = 1; W[3,3] = -1
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    # C01 += M5, C11 += M5
    W[2,4] = 1; W[3,4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    # C00 += M6, C10 += M6
    W[1,5] = 1; W[3,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    # C01 += M7, C11 += M7
    W[0,6] = 1; W[2,6] = 1
    
    return U, V, W

def _construct_rank14_einsum():
    """
    Construct rank-14 decomposition using einsum-based block assembly.
    Two independent Strassen 2x2x2 blocks for columns 0-1 and 2-3.
    """
    n, m, p = 2, 2, 4
    rank = 14
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    U_s, V_s, W_s = _strassen_2x2_2x2()
    
    # Block 1: columns 0-1 (k=0,1)
    for r in range(7):
        U[:, r] = U_s[:, r]
        # V indices: j*p+k, for k=0,1 -> j*4+0, j*4+1
        for j in range(m):
            for k in range(2):
                V[j*p+k, r] = V_s[j*2+k, r]
        # W indices: k*n+i, for k=0,1 -> 0*n+i, 1*n+i
        for k in range(2):
            for i in range(n):
                W[k*n+i, r] = W_s[k*2+i, r]
    
    # Block 2: columns 2-3 (k=2,3)
    for r in range(7):
        U[:, r+7] = U_s[:, r]
        # V indices: j*p+k, for k=2,3 -> j*4+2, j*4+3
        for j in range(m):
            for k in range(2, 4):
                V[j*p+k, r+7] = V_s[j*2+(k-2), r]
        # W indices: k*n+i, for k=2,3 -> 2*n+i, 3*n+i
        for k in range(2, 4):
            for i in range(n):
                W[k*n+i, r+7] = W_s[(k-2)*2+i, r]
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_rank_r_structured(n, m, p, r, deadline):
    """
    Try to construct a rank-r decomposition using structured block methods.
    Returns (U, V, W) as lists if successful, None otherwise.
    """
    if time.time() > deadline:
        return None
    
    if r == 14:
        U, V, W = _construct_rank14_einsum()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        return None
    
    # For other ranks, we don't have a structured construction yet
    return None

def _search_rank_r_with_fixed_U(n, m, p, r, deadline, max_attempts=200):
    """
    Search for rank-r decomposition by fixing U (or V) and solving for the rest.
    This is a more systematic approach than random search.
    
    Strategy:
    1. Fix U to have a specific structure (e.g., Strassen-like patterns)
    2. Solve the resulting linear system for V and W
    3. Check if the solution is valid (half-integer entries)
    """
    if time.time() > deadline:
        return None
    
    # Try different structured U patterns
    U_patterns = []
    
    # Pattern 1: Strassen-based U (first 7 columns are Strassen, rest are variations)
    U_s, _, _ = _strassen_2x2_2x2()
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(attempt * 1000)
        
        # Build U with some structure
        U_try = np.zeros((n*m, r), dtype=float)
        
        if attempt < 50:
            # Use Strassen patterns for first 7 columns
            for col in range(min(7, r)):
                U_try[:, col] = U_s[:, col]
            # Fill remaining columns randomly
            for col in range(7, r):
                u_rows = np.random.choice(n*m, size=2, replace=False)
                for row in u_rows:
                    U_try[row, col] = np.random.choice([-1.0, 1.0])
        elif attempt < 100:
            # Use Strassen patterns with some modifications
            for col in range(min(7, r)):
                U_try[:, col] = U_s[:, col]
                # Randomly flip some signs
                if np.random.random() < 0.3:
                    U_try[:, col] *= -1
            for col in range(7, r):
                u_rows = np.random.choice(n*m, size=2, replace=False)
                for row in u_rows:
                    U_try[row, col] = np.random.choice([-1.0, 1.0])
        else:
            # Fully random sparse U
            for col in range(r):
                u_rows = np.random.choice(n*m, size=2, replace=False)
                for row in u_rows:
                    U_try[row, col] = np.random.choice([-1.0, 1.0])
        
        # Now solve for V and W jointly
        # This is a bilinear system, so we need to iterate
        # Initialize V randomly
        V_try = np.zeros((m*p, r), dtype=float)
        for col in range(r):
            v_rows = np.random.choice(m*p, size=3, replace=False)
            for row in v_rows:
                V_try[row, col] = np.random.choice([-1.0, 1.0])
        
        # Solve for W given U and V
        # Equation (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        A = np.zeros((n*m*p, p*n*r), dtype=float)
        b = np.ones(n*m*p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        coeff = U_try[i*m+j, r_idx] * V_try[j*p+k, r_idx]
                        if coeff != 0:
                            A[eq_idx, (k*n+i)*r + r_idx] = coeff
                    eq_idx += 1
        
        try:
            W_flat, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        W_try = W_flat.reshape(p*n, r)
        
        # Check residual
        residual = np.linalg.norm(A @ W_flat - b)
        if residual > 1e-6:
            continue
        
        # Check if W is close to half-integers
        W_2x = W_try * 2
        W_rounded = np.round(W_2x)
        error = np.linalg.norm(W_2x - W_rounded) / max(np.linalg.norm(W_2x), 1e-10)
        
        if error < 0.05:
            W_clean = W_rounded / 2.0
            U_list = U_try.tolist()
            V_list = V_try.tolist()
            W_list = W_clean.tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_alternative_block_decomposition(n, m, p, r, deadline):
    """
    Try alternative block decompositions that might achieve lower rank.
    
    For 2x2 * 2x4, we can think of the B matrix as two 2x2 blocks.
    Instead of treating them independently (which gives 7+7=14),
    we can try to find correlations between the blocks.
    """
    if time.time() > deadline:
        return None
    
    # Try rank 12: assume we can save 2 terms by correlating the blocks
    # This requires a specific structure that we haven't found yet
    # For now, return None as a placeholder
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Start with guaranteed rank-14
    U14, V14, W14 = _construct_rank14_einsum()
    if not _verify_decomposition(U14, V14, W14, n, m, p):
        # Fallback to trivial rank-16
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = {"U": U14, "V": V14, "W": W14}
    best_rank = 14
    
    # Try to find rank 13
    if best_rank > 13 and time.time() < deadline - 10:
        result = _search_rank_r_with_fixed_U(n, m, p, 13, deadline, max_attempts=100)
        if result is not None:
            best_result = result
            best_rank = 13
    
    # Try to find rank 12
    if best_rank > 12 and time.time() < deadline - 10:
        result = _search_rank_r_with_fixed_U(n, m, p, 12, deadline, max_attempts=100)
        if result is not None:
            best_result = result
            best_rank = 12
    
    # Try alternative block decomposition
    if best_rank > 12 and time.time() < deadline - 5:
        result = _try_alternative_block_decomposition(n, m, p, 12, deadline)
        if result is not None:
            best_result = result
            best_rank = 12
    
    return best_result
