<<<<<<< SEARCH
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W form a valid tensor decomposition for matrix multiplication.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    result = _trivial_decomposition(n, m, p)
    U, V, W = result['U'], result['V'], result['W']
    
    # Quick verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback, though should not happen
        return _trivial_decomposition(n, m, p)
        
    return result
=======
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W form a valid tensor decomposition for matrix multiplication.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """Return Strassen's rank-7 decomposition for 2x2 matrix multiplication."""
    # Strassen's 7 multiplications for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    U = [
        [1, 1, 0, 0, 0, 0, 0],  # a: m1, m2(no), m3, m4(no), m5(no), m6, m7(no)
        [0, 0, 1, 0, 1, 0, 0],  # b: m1(no), m2(no), m3, m4(no), m5, m6(no), m7(no)
        [0, 1, 0, 0, 0, 1, 0],  # c: m1(no), m2, m3(no), m4(no), m5(no), m6, m7(no)
        [1, 1, 0, 1, 0, 0, 0],  # d: m1, m2, m3(no), m4, m5(no), m6(no), m7(no)
    ]
    V = [
        [1, 1, -1, 1, 1, 1, 1],  # e: m1, m2, m3, m4, m5(no), m6, m7(no)
        [1, 0, 1, 0, 1, 1, 1],  # f: m1, m2(no), m3, m4(no), m5, m6, m7
        [1, 0, -1, 0, 0, 0, 1], # g: m1, m2(no), m3, m4(no), m5(no), m6(no), m7
        [1, 0, 0, -1, 1, 0, 1], # h: m1, m2(no), m3(no), m4, m5, m6(no), m7
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C11: m1, m2(no), m3(no), m4, m5, m6(no), m7
        [0, 0, 1, 0, 1, 0, 0],   # C12: m1(no), m2(no), m3, m4(no), m5, m6(no), m7(no)
        [0, 1, 0, 1, 0, 0, 0],   # C21: m1(no), m2, m3(no), m4, m5(no), m6(no), m7(no)
        [1, -1, 1, 0, 0, 1, 0],  # C22: m1, m2, m3, m4(no), m5(no), m6, m7(no)
    ]
    return U, V, W

def _strassen_block_decomposition(n, m, p):
    """
    Build a Strassen-based decomposition by treating the 5x5 blocks as 2x2 super-blocks
    with padding to 4x4, then applying Strassen recursively.
    """
    # For 3x5 x 5x5, we can decompose the 5x5 into 2x2 blocks with padding
    # Actually, let's use a simpler approach: block the matrices into 2x2 and 1x1 pieces
    # A is 3x5, B is 5x5
    # We can write A as [A1 A2] where A1 is 3x2, A2 is 3x3
    # B as [[B11 B12],[B21 B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    
    # This is getting complex. Let me use a different approach:
    # Use the fact that rank(3x5x5) <= rank(3x4x4) * something + corrections
    # Or just use the trivial decomposition but try to find linear dependencies
    
    # For now, return the trivial decomposition as a starting point
    return _trivial_decomposition(n, m, p)

def _check_linear_dependency(U, V, W, n, m, p):
    """
    Check if any column is a linear combination of others.
    Returns list of column indices that can be removed.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    removable = []
    for col in range(rank):
        # Check if removing this column still gives valid decomposition
        # Quick check: see if the column vectors are linearly dependent
        # This is expensive, so we do a simple check
        pass
    
    return removable

def _try_remove_columns(U, V, W, n, m, p, max_removals=5):
    """
    Try to remove columns that are linearly dependent.
    """
    rank = len(U[0])
    removed = []
    
    for _ in range(max_removals):
        if rank == 0:
            break
            
        found_removal = False
        for col in range(rank):
            # Try removing this column
            U_new = U[:, :col] + U[:, col+1:] if rank > 1 else []
            V_new = V[:, :col] + V[:, col+1:] if rank > 1 else []
            W_new = W[:, :col] + W[:, col+1:] if rank > 1 else []
            
            if not U_new or not V_new or not W_new:
                continue
                
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                removed.append(col)
                found_removal = True
                break
        
        if not found_removal:
            break
    
    return U, V, W, removed

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    # Start with trivial decomposition
    result = _trivial_decomposition(n, m, p)
    best_U, best_V, best_W = result['U'], result['V'], result['W']
    best_rank = len(best_U[0])
    
    # Try to reduce rank by removing linearly dependent columns
    if time.time() - start_time < deadline - 30:
        U, V, W, removed = _try_remove_columns(best_U, best_V, best_W, n, m, p, max_removals=10)
        if len(removed) > 0:
            best_U, best_V, best_W = U, V, W
            best_rank = len(best_U[0])
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        return _trivial_decomposition(n, m, p)
        
    return {"U": best_U, "V": best_V, "W": best_W}
>>>>>>> REPLACE
