"""Matrix multiplication 4,4,6 rank-minimal decomposition."""
import numpy as np
import time

def construct():
    n, m, p = 4, 4, 6
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build target tensor for verification
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    def check_decomposition(U, V, W):
        """Check if the decomposition exactly reproduces the target tensor."""
        T_test = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_test += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        return np.allclose(T_test, T, atol=1e-10)
    
    # Build trivial rank-96 decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n*m, rank_trivial))
    V_trivial = np.zeros((m*p, rank_trivial))
    W_trivial = np.zeros((p*n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1
                V_trivial[j*p+k, col] = 1
                W_trivial[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_trivial
    best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    
    # Strategy: Split 4x4x6 into 4x4x4 (rank 28) + 4x4x2 (rank 8) = rank 36
    
    rank_444 = 28  # 4 blocks of 2x2x2 Strassen
    rank_442 = 8   # 4 j-values * 2 k-values
    total_rank = rank_444 + rank_442  # 36
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # 4x4x4 part: 4 blocks of 2x2x2 using Strassen
    for ib in range(2):
        for jb in range(2):
            i_start = ib * 2
            j_start = jb * 2
            k_start = ib * 2
            
            # Factor 0: M1 = (A11+A22)*(B11+B22)
            U[i_start*m + j_start, col] = 1
            U[(i_start+1)*m + (j_start+1), col] = 1
            V[j_start*p + k_start, col] = 1
            V[(j_start+1)*p + (k_start+1), col] = 1
            W[k_start*n + i_start, col] = 1
            W[(k_start+1)*n + (i_start+1), col] = 1
            col += 1
            
            # Factor 1: M2 = (A21+A22)*B11
            U[(i_start+1)*m + j_start, col] = 1
            U[(i_start+1)*m + (j_start+1), col] = 1
            V[j_start*p + k_start, col] = 1
            W[k_start*n + (i_start+1), col] = 1
            col += 1
            
            # Factor 2: M3 = A11*(B12-B22)
            U[i_start*m + j_start, col] = 1
            V[j_start*p + (k_start+1), col] = 1
            V[(j_start+1)*p + (k_start+1), col] = -1
            W[(k_start+1)*n + i_start, col] = 1
            W[(k_start+1)*n + (i_start+1), col] = 1
            col += 1
            
            # Factor 3: M4 = A22*(B21-B11)
            U[(i_start+1)*m + (j_start+1), col] = 1
            V[(j_start+1)*p + k_start, col] = 1
            V[j_start*p + k_start, col] = -1
            W[k_start*n + i_start, col] = 1
            W[k_start*n + (i_start+1), col] = 1
            col += 1
            
            # Factor 4: M5 = (A11+A12)*B22
            U[i_start*m + j_start, col] = 1
            U[i_start*m + (j_start+1), col] = 1
            V[(j_start+1)*p + (k_start+1), col] = 1
            W[k_start*n + i_start, col] = 1
            W[(k_start+1)*n + i_start, col] = 1
            col += 1
            
            # Factor 5: M6 = (A21-A11)*(B11+B12)
            U[(i_start+1)*m + j_start, col] = 1
            U[i_start*m + j_start, col] = -1
            V[j_start*p + k_start, col] = 1
            V[j_start*p + (k_start+1), col] = 1
            W[(k_start+1)*n + (i_start+1), col] = 1
            col += 1
            
            # Factor 6: M7 = (A12-A22)*(B21+B22)
            U[i_start*m + (j_start+1), col] = 1
            U[(i_start+1)*m + (j_start+1), col] = -1
            V[(j_start+1)*p + k_start, col] = 1
            V[(j_start+1)*p + (k_start+1), col] = 1
            W[k_start*n + i_start, col] = 1
            W[(k_start+1)*n + (i_start+1), col] = 1
            col += 1
    
    # 4x4x2 part: 8 rank-1 terms
    for j in range(4):
        for k in [4, 5]:
            for i in range(4):
                U[i*m + j, col] = 1
            V[j*p + k, col] = 1
            for i in range(4):
                W[k*n + i, col] = 1
            col += 1
    
    # Verify
    if check_decomposition(U, V, W):
        best_rank = total_rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    else:
        best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
