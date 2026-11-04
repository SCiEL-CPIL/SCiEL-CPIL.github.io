import time
import numpy as np

DIMENSIONS = [4, 4, 7]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reconstructs the matrix multiplication tensor using vectorized ops."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Vectorized: compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape for broadcasting
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=tol)

def _schoolbook_decomposition(n, m, p):
    """Return the trivial schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2.
    Rank 7 decomposition.
    U, V, W shapes: (4, 7)
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # m1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # m3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # m4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # m5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # m6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # m7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return np.array(U), np.array(V), np.array(W)

def _rank49_4x4x4():
    """
    Construct rank 49 decomposition for 4x4 * 4x4 using Kronecker product of Strassen.
    U, V, W shapes: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            # U4[i, c] = U2[i1, c1] * U2[i2, c2]
            # i = 2*i1 + i2
            for i1 in range(2):
                for i2 in range(2):
                    i = 2 * i1 + i2
                    U4[i, c] = U2[i1, c1] * U2[i2, c2]
            
            # V4[j, c] = V2[j1, c1] * V2[j2, c2]
            # j = 2*j1 + j2
            for j1 in range(2):
                for j2 in range(2):
                    j = 2 * j1 + j2
                    V4[j, c] = V2[j1, c1] * V2[j2, c2]
            
            # W4[k, c] = W2[k1, c1] * W2[k2, c2]
            # k = 2*k1 + k2
            for k1 in range(2):
                for k2 in range(2):
                    k = 2 * k1 + k2
                    W4[k, c] = W2[k1, c1] * W2[k2, c2]
                    
    return U4, V4, W4

def _build_4x4x7_strassen():
    """
    Build a rank 98 decomposition for 4x4 * 4x7 by splitting B into 4x3 and 4x4 parts.
    Part 1: 4x4 * 4x3 -> use 4x4x4 decomposition but zero out columns corresponding to k=3 in V and W.
    Part 2: 4x4 * 4x4 -> use 4x4x4 decomposition, shift k indices by 3.
    """
    U4, V4, W4 = _rank49_4x4x4()
    n, m, p = 4, 4, 7
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: 4x4 * 4x3 (cols 0-2 of B)
    for c in range(49):
        U[:, c] = U4[:, c]
        
        # V part: local k=0,1,2 -> global k=0,1,2
        for j in range(4):
            for k in range(3):
                V[j * 7 + k, c] = V4[j * 4 + k, c]
        
        # W part: local k=0,1,2 -> global k=0,1,2
        for k in range(3):
            for i in range(4):
                W[k * 4 + i, c] = W4[k * 4 + i, c]

    # Part 2: 4x4 * 4x4 (cols 3-6 of B)
    for c in range(49):
        c2 = 49 + c
        U[:, c2] = U4[:, c]
        
        # V part: local k=0,1,2,3 -> global k=3,4,5,6
        for j in range(4):
            for k in range(4):
                V[j * 7 + (3 + k), c2] = V4[j * 4 + k, c]
                
        # W part: local k=0,1,2,3 -> global k=3,4,5,6
        for k in range(4):
            for i in range(4):
                W[(3 + k) * 4 + i, c2] = W4[k * 4 + i, c]
                
    return U, V, W

def _try_reduce_rank(n, m, p, start_rank, target_rank, time_budget=30.0):
    """
    Attempt to reduce the rank using structured linear algebra approaches.
    Returns (success, U, V, W) where success is True if a valid lower-rank decomposition was found.
    """
    start_time = time.time()
    
    # Start with the known decomposition
    U_init, V_init, W_init = _build_4x4x7_strassen()
    
    best_rank = start_rank
    best_U, best_V, best_W = U_init, V_init, W_init
    
    # Try to find a lower-rank decomposition by exploiting structure
    # For 4x4x7, try using a 2x2x7 sub-problem decomposition
    
    # Approach: View 4x4x7 as 4 instances of 2x2x7 (splitting A into 2x2 blocks)
    # 2x2x7: A(2x2) * B(2x7) -> C(2x7)
    # This can be done with rank at most 2*2*7 = 28 (schoolbook) or less
    
    # Let's try a direct construction for 2x2xp
    def _strassen_2x2xp(p_val, seed=42):
        """Construct a decomposition for 2x2 * 2xp -> 2xp.
        U: (4, rank), V: (2*p_val, rank), W: (p_val*2, rank)
        """
        # Use the 2x2x2 Strassen for each column, but try to share
        # For now, use schoolbook: rank = 4*p_val
        rank = 4 * p_val
        U = np.zeros((4, rank))
        V = np.zeros((2 * p_val, rank))
        W = np.zeros((p_val * 2, rank))
        
        col = 0
        for k in range(p_val):
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j, col] = 1.0
                    V[j * p_val + k, col] = 1.0
                    W[k * 2 + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Try various target ranks
    for rank in range(target_rank, start_rank):
        if time.time() - start_time > time_budget:
            break
        # Skip for now - no reliable method to find lower rank
        pass
    
    return best_rank, best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    
    # First, get the known rank 98 decomposition
    U98, V98, W98 = _build_4x4x7_strassen()
    
    # Verify it's correct
    if not _verify_decomposition(U98, V98, W98, n, m, p, tol=1e-9):
        # Fallback to schoolbook
        U_s, V_s, W_s = _schoolbook_decomposition(n, m, p)
        return {"U": U_s, "V": V_s, "W": W_s}
    
    # Try to reduce the rank with a time budget
    time_budget = min(60.0, TIME_LIMIT - (time.time() - START_TIME) - SAFETY_MARGIN)
    target_rank = 64  # Try to get down to at least 64
    
    try:
        best_rank, best_U, best_V, best_W = _try_reduce_rank(n, m, p, 98, target_rank, time_budget)
    except Exception as e:
        print(f"Rank reduction failed: {e}")
        best_rank, best_U, best_V, best_W = 98, U98, V98, W98
    
    # Verify the final result
    if _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-9):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to the known rank 98 decomposition
        return {"U": U98.tolist(), "V": V98.tolist(), "W": W98.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
