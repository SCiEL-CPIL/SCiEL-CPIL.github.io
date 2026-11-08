import numpy as np
import time

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140

def _build_expected_tensor(n, m, p):
    """Build the expected matmul tensor T[a,b,c] = 1 if indices consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _verify_decomposition(U, V, W, T_expected, atol=1e-8):
    """Verify that sum_r U[:,r]⊗V[:,r]⊗W[:,r] == T_expected."""
    nm = U.shape[0]
    mp = V.shape[0]
    pn = W.shape[0]
    rank = U.shape[1]
    T_computed = np.zeros((nm, mp, pn))
    for r in range(rank):
        T_computed += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_computed, T_expected, atol=atol)

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses a block decomposition with Strassen for 2x2x2 blocks.
    Rank is 56.
    """
    n, m, p = DIMENSIONS
    
    if time.time() > _DEADLINE:
        return _schoolbook_fallback(n, m, p)
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_idx, v_idx, w_idx, U_block, V_block, W_block):
        rank = U_block.shape[1]
        for r in range(rank):
            U_parts.append(np.zeros(n * m))
            for i, idx in enumerate(u_idx):
                U_parts[-1][idx] = U_block[i, r]
            
            V_parts.append(np.zeros(m * p))
            for i, idx in enumerate(v_idx):
                V_parts[-1][idx] = V_block[i, r]
            
            W_parts.append(np.zeros(p * n))
            for i, idx in enumerate(w_idx):
                W_parts[-1][idx] = W_block[i, r]
    
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W

    # Block 1: rows 0-1 of A, cols 0-1 of B (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([0, 1, 4, 5], [0, 1, 5, 6], [0, 3, 1, 4], U_blk, V_blk, W_blk)
    
    # Block 2: rows 0-1 of A, cols 2-3 of B (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([0, 1, 4, 5], [2, 3, 7, 8], [6, 9, 7, 10], U_blk, V_blk, W_blk)
    
    # Block 3: rows 0-1 of A, col 4 of B (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block([0, 1, 4, 5], [4, 9], [12, 13], U_blk, V_blk, W_blk)
    
    # Block 4: rows 0-1 of A, cols 2-3 of B (2x2x2) -- Wait, this is A[0:2, 2:4]
    # The previous block 1 was A[0:2, 0:2].
    # Block 4 uses u_idx [2, 3, 6, 7] which are A[0,2], A[0,3], A[1,2], A[1,3].
    # So this is correct.
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([2, 3, 6, 7], [10, 11, 15, 16], [0, 3, 1, 4], U_blk, V_blk, W_blk)
    
    # Block 5: rows 0-1 of A, cols 2-3 of B (2x2x2) -- A[0:2, 2:4] x B[2:4, 2:4]
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([2, 3, 6, 7], [12, 13, 17, 18], [6, 9, 7, 10], U_blk, V_blk, W_blk)
    
    # Block 6: rows 0-1 of A, col 4 of B (2x2x1) -- A[0:2, 2:4] x B[2:4, 4]
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block([2, 3, 6, 7], [14, 19], [12, 13], U_blk, V_blk, W_blk)
    
    # Block 7: row 2 of A, cols 0-1 of B (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block([8, 9], [0, 1, 5, 6], [2, 5], U_blk, V_blk, W_blk)
    
    # Block 8: row 2 of A, cols 2-3 of B (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block([8, 9], [2, 3, 7, 8], [8, 11], U_blk, V_blk, W_blk)
    
    # Block 9: row 2 of A, col 4 of B (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block([8, 9], [4, 9], [14], U_blk, V_blk, W_blk)
    
    # Block 10: row 2 of A, cols 0-1 of B (1x2x2) -- Wait, u_idx [10, 11] is A[2,2], A[2,3]
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block([10, 11], [10, 11, 15, 16], [2, 5], U_blk, V_blk, W_blk)
    
    # Block 11: row 2 of A, cols 2-3 of B (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block([10, 11], [12, 13, 17, 18], [8, 11], U_blk, V_blk, W_blk)
    
    # Block 12: row 2 of A, col 4 of B (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block([10, 11], [14, 19], [14], U_blk, V_blk, W_blk)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    rank = U.shape[1]
    
    # Verify correctness
    if time.time() < _DEADLINE - 5:
        T_computed = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_computed += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        T_expected = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T_expected[a_idx, b_idx, c_idx] = 1.0
        
        if not np.allclose(T_computed, T_expected, atol=1e-10):
            return _schoolbook_fallback(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook_fallback(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    U_parts = []
    V_parts = []
    W_parts = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                u[i * m + j] = 1
                v = np.zeros(m * p)
                v[j * p + k] = 1
                w = np.zeros(p * n)
                w[k * n + i] = 1
                U_parts.append(u)
                V_parts.append(v)
                W_parts.append(w)
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
