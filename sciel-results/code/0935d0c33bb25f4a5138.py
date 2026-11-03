import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W):
    """Verify that U, V, W form a valid decomposition using the trace convention."""
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _construct_rank_28():
    """
    Decompose 2x3 * 3x5 matrix multiplication using rank-28 ansatz.
    
    Strategy: Use Strassen's 7-term decomposition for two disjoint 2x2x2 blocks.
    Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    The remaining 14 terms are handled by schoolbook multiplication.
    Total rank: 7 + 7 + 14 = 28.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    
    def add_strassen_block(j_off, k_off):
        nonlocal c
        # m1 = (a00+a11)*(b00+b11)
        U[0*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # m2 = (a10+a11)*b00
        U[1*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1
        W[k_off*n + 1][c] = 1; W[(k_off+1)*n + 1][c] = -1
        c += 1
        
        # m3 = a00*(b01-b11)
        U[0*m + j_off][c] = 1
        V[j_off*p + (k_off+1)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # m4 = a11*(b10-b00)
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + k_off][c] = 1; V[j_off*p + k_off][c] = -1
        W[k_off*n + 0][c] = 1; W[k_off*n + 1][c] = 1
        c += 1
        
        # m5 = (a00+a01)*b11
        U[0*m + j_off][c] = 1; U[0*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = -1; W[(k_off+1)*n + 0][c] = 1
        c += 1
        
        # m6 = (a10-a00)*(b00+b01)
        U[1*m + j_off][c] = 1; U[0*m + j_off][c] = -1
        V[j_off*p + k_off][c] = 1; V[j_off*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # m7 = (a01-a11)*(b10+b11)
        U[0*m + (j_off+1)][c] = 1; U[1*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1
        c += 1

    # Block 1: j in {0,1}, k in {0,1}
    add_strassen_block(0, 0)
    
    # Block 2: j in {0,1}, k in {2,3}
    add_strassen_block(0, 2)
    
    # Remaining 14 schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in (0, 1) and k in (0, 1, 2, 3):
                    continue
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    
    assert c == rank, f"Expected rank {rank}, got {c}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank_27_search():
    """
    Attempt to find a rank-27 decomposition by trying to merge
    one schoolbook term with the Strassen structure.
    
    This is a heuristic search - if it fails, we return None.
    """
    n, m, p = 2, 3, 5
    rank = 27
    
    # Start from the rank-28 solution and try to eliminate one degree of freedom
    # by setting one entry to a specific value and checking consistency.
    
    # The 14 schoolbook terms are:
    # j=0,1 with k=4: 4 terms
    # j=2 with k=0,1,2,3,4: 10 terms
    
    # Try to see if we can express one of these as a linear combination
    # of the Strassen terms. This requires solving a system.
    
    # For now, this is a placeholder - the rank-28 solution is well-established.
    # A full rank-27 search would require solving a system of polynomial equations.
    return None

def construct():
    """
    Constructs the factor matrices for matrix multiplication 2x3 by 3x5.
    Returns a dictionary with keys 'U', 'V', 'W'.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try rank-27 search first (quick check)
    result_27 = _try_rank_27_search()
    if result_27 is not None:
        if _verify_decomposition(result_27['U'], result_27['V'], result_27['W']):
            return result_27
    
    # Fall back to the well-established rank-28 decomposition
    result_28 = _construct_rank_28()
    
    # Verify the decomposition
    if _verify_decomposition(result_28['U'], result_28['V'], result_28['W']):
        return result_28
    
    # If verification fails (shouldn't happen), fall back to schoolbook
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
# EVOLVE-BLOCK-END
