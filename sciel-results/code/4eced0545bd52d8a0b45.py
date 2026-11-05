import time
import numpy as np

N, M, P = 3, 4, 5
SCHOOLBOOK_RANK = N * M * P  # 60

START_TIME = time.time()
DEADLINE = START_TIME + 110

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n=N, m=M, p=P):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9, rtol=0)

def _schoolbook():
    rank = SCHOOLBOOK_RANK
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _structured_3x4x5():
    """
    Construct a structured decomposition for 3x4x5 matrix multiplication.
    Rank 42.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    rank = 42
    U = np.zeros((12, rank))  # 3*4
    V = np.zeros((20, rank))  # 4*5
    W = np.zeros((15, rank))  # 5*3
    
    col = 0
    
    # Helper to add a block of size (ni x nj x nk) at offset (i0, j0, k0)
    # ni, nj, nk are the dimensions of the block
    # We assume the block decomposition is known (U_blk, V_blk, W_blk)
    # and we embed it into the global matrices.
    
    def add_block(i0, j0, k0, ni, nj, nk, U_blk, V_blk, W_blk):
        nonlocal col
        r_count = U_blk.shape[1]
        for r in range(r_count):
            for ii in range(ni):
                for jj in range(nj):
                    U[(i0 + ii) * M + (j0 + jj), col] += U_blk[ii * nj + jj, r]
            for jj in range(nj):
                for kk in range(nk):
                    V[(j0 + jj) * P + (k0 + kk), col] += V_blk[jj * nk + kk, r]
            for kk in range(nk):
                for ii in range(ni):
                    W[(k0 + kk) * N + (i0 + ii), col] += W_blk[kk * ni + ii, r]
            col += 1
        
    # Part 1: i in {0, 1}
    # j splits: {0,1}, {2,3}
    # k splits: {0,1}, {2,3}, {4}
    
    # Block 1: i=0,1; j=0,1; k=0,1 (2x2x2)
    add_block(0, 0, 0, 2, 2, 2, U2, V2, W2)
    
    # Block 2: i=0,1; j=0,1; k=2,3 (2x2x2)
    add_block(0, 0, 2, 2, 2, 2, U2, V2, W2)
    
    # Block 3: i=0,1; j=0,1; k=4 (2x2x1) Rank 2
    # T_{i,j,k} = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # For 2x2x1, U is 4x2, V is 2x2, W is 1x2
    U_blk_221 = np.zeros((4, 2))
    V_blk_221 = np.zeros((2, 2))
    W_blk_221 = np.zeros((1, 2))
    # Term 0: j=0
    U_blk_221[0, 0] = 1 # i=0, j=0
    V_blk_221[0, 0] = 1 # j=0, k=0
    W_blk_221[0, 0] = 1 # k=0, i=0
    # Term 1: j=1
    U_blk_221[1, 1] = 1 # i=0, j=1? No, U index is i*nj+j.
    # Wait, U shape is (ni*nj, rank).
    # For 2x2x1:
    # U indices: (0,0), (0,1), (1,0), (1,1)
    # V indices: (0,0), (1,0)
    # W indices: (0,0), (0,1)
    
    # We need T_{i,j,k} = delta_{j, k_local}? No, k is single.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Here k is fixed to global k=4.
    # So C_{i,4} = A_{i,0} B_{0,4} + A_{i,1} B_{1,4}.
    # Tensor T_{i,j,0} (local k=0) should be 1 if j matches?
    # No, T_{i,j,k} is coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # So T_{i,j,0} = 1 for all i? No.
    # T_{i,j,k} = delta_{i,i} delta_{j,j} delta_{k,k}?
    # The tensor is simply the identity on the j index for each i,k pair?
    # Actually, for a single k column, the multiplication is:
    # C_{i, k} = \sum_j A_{i,j} B_{j,k}
    # This is a sum of products.
    # The tensor T_{i,j,k} has 1s at (i, j, k) such that ...?
    # No, the tensor for matrix multiplication is T_{i,j,k} = 1 if indices are consistent.
    # Specifically, T_{i,j,k} = 1 for all i,j,k? No.
    # T_{i,j,k} is the structure tensor.
    # For 1x2x1: T_{0,0,0}=1, T_{0,1,0}=0?
    # C = A B. C_0 = A_0 B_0 + A_1 B_1.
    # So T_{0,0,0} = 1, T_{0,1,0} = 1.
    # This means U_{0,0,0}=1, V_{0,0,0}=1, W_{0,0,0}=1
    # and U_{0,1,1}=1, V_{1,0,1}=1, W_{0,0,1}=1?
    # Let's check:
    # Term 0: U_{0,0,0}=1, V_{0,0,0}=1, W_{0,0,0}=1 -> T_{0,0,0} += 1.
    # Term 1: U_{0,1,1}=1, V_{1,0,1}=1, W_{0,0,1}=1 -> T_{0,1,0} += 1.
    # This works. Rank 2.
    
    # For 2x2x1:
    # C_{i,0} = A_{i,0} B_{0,0} + A_{i,1} B_{1,0}
    # T_{i,0,0} = 1, T_{i,1,0} = 1.
    # Term 0: U_{i,0,0}=1, V_{0,0,0}=1, W_{0,i,0}=1.
    # Term 1: U_{i,1,1}=1, V_{1,0,1}=1, W_{0,i,1}=1.
    
    U_blk_221 = np.zeros((4, 2))
    V_blk_221 = np.zeros((2, 2))
    W_blk_221 = np.zeros((2, 2)) # W shape is (nk*ni, rank) = (1*2, 2)
    
    # Term 0 (j=0)
    U_blk_221[0, 0] = 1 # i=0, j=0
    U_blk_221[2, 0] = 1 # i=1, j=0
    V_blk_221[0, 0] = 1 # j=0, k=0
    W_blk_221[0, 0] = 1 # k=0, i=0
    W_blk_221[1, 0] = 1 # k=0, i=1
    
    # Term 1 (j=1)
    U_blk_221[1, 1] = 1 # i=0, j=1
    U_blk_221[3, 1] = 1 # i=1, j=1
    V_blk_221[1, 1] = 1 # j=1, k=0
    W_blk_221[0, 1] = 1 # k=0, i=0
    W_blk_221[1, 1] = 1 # k=0, i=1
    
    add_block(0, 0, 4, 2, 2, 1, U_blk_221, V_blk_221, W_blk_221)
    
    # Block 4: i=0,1; j=2,3; k=0,1 (2x2x2)
    add_block(0, 2, 0, 2, 2, 2, U2, V2, W2)
    
    # Block 5: i=0,1; j=2,3; k=2,3 (2x2x2)
    add_block(0, 2, 2, 2, 2, 2, U2, V2, W2)
    
    # Block 6: i=0,1; j=2,3; k=4 (2x2x1)
    add_block(0, 2, 4, 2, 2, 1, U_blk_221, V_blk_221, W_blk_221)
    
    # Part 2: i=2
    # j splits: {0,1}, {2,3}
    # k splits: {0,1}, {2,3}, {4}
    
    # Block 7: i=2; j=0,1; k=0,1 (1x2x2) Rank 2
    # C_{0,0} = A_{0,0}B_{0,0} + A_{0,1}B_{1,0}
    # C_{0,1} = A_{0,0}B_{0,1} + A_{0,1}B_{1,1}
    # T_{0,0,0}=1, T_{0,0,1}=0? No.
    # T_{i,j,k}. i=0.
    # T_{0,0,0}=1, T_{0,1,0}=0?
    # C_k = sum_j A_j B_{j,k}.
    # T_{0,j,k} = 1 if j matches?
    # T_{0,0,0}=1, T_{0,0,1}=0?
    # No, T_{0,j,k} is coeff of A_j B_{j,k} in C_k.
    # So T_{0,0,0}=1, T_{0,0,1}=0? No, B_{0,1} is in C_1.
    # T_{0,0,1}=1? No, B_{0,1} is j=0, k=1.
    # So T_{0,0,1}=1.
    # T_{0,1,0}=1 (A_1 B_{1,0} in C_0).
    # T_{0,1,1}=1 (A_1 B_{1,1} in C_1).
    # So T_{0,j,k} = delta_{j,k}.
    # Rank 2.
    # Term 0: j=0, k=0. U_{0,0,0}=1, V_{0,0,0}=1, W_{0,0,0}=1.
    # Term 1: j=1, k=1. U_{0,1,1}=1, V_{1,1,1}=1, W_{1,0,1}=1.
    
    U_blk_122 = np.zeros((2, 2)) # 1x2
    V_blk_122 = np.zeros((4, 2)) # 2x2
    W_blk_122 = np.zeros((2, 2)) # 2x1
    
    U_blk_122[0, 0] = 1
    V_blk_122[0, 0] = 1
    W_blk_122[0, 0] = 1
    
    U_blk_122[1, 1] = 1
    V_blk_122[3, 1] = 1 # j=1, k=1 -> index 1*2+1=3
    W_blk_122[1, 1] = 1 # k=1, i=0 -> index 1
    
    add_block(2, 0, 0, 1, 2, 2, U_blk_122, V_blk_122, W_blk_122)
    
    # Block 8: i=2; j=0,1; k=2,3 (1x2x2)
    add_block(2, 0, 2, 1, 2, 2, U_blk_122, V_blk_122, W_blk_122)
    
    # Block 9: i=2; j=0,1; k=4 (1x2x1) Rank 1
    # C_0 = A_0 B_0 + A_1 B_1
    # T_{0,0,0}=1, T_{0,1,0}=1
    # Term 0: U_{0,0,0}=1, V_{0,0,0}=1, V_{1,0,0}=1, W_{0,0,0}=1
    # Wait, Rank 1 means 1 term.
    # U_{0,j,0} = 1 for all j?
    # V_{j,k,0} = 1 for all j?
    # W_{k,i,0} = 1?
    # T_{0,j,0} = sum_r U_{0,j,r} V_{j,0,r} W_{0,0,r}
    # If r=0: U_{0,j,0}=1, V_{j,0,0}=1, W_{0,0,0}=1.
    # T_{0,j,0} = 1.
    # This gives T_{0,0,0}=1, T_{0,1,0}=1.
    # Correct.
    
    U_blk_121 = np.zeros((2, 1))
    V_blk_121 = np.zeros((2, 1))
    W_blk_121 = np.zeros((1, 1))
    
    U_blk_121[0, 0] = 1
    U_blk_121[1, 0] = 1
    V_blk_121[0, 0] = 1
    V_blk_121[1, 0] = 1
    W_blk_121[0, 0] = 1
    
    add_block(2, 0, 4, 1, 2, 1, U_blk_121, V_blk_121, W_blk_121)
    
    # Block 10: i=2; j=2,3; k=0,1 (1x2x2)
    add_block(2, 2, 0, 1, 2, 2, U_blk_122, V_blk_122, W_blk_122)
    
    # Block 11: i=2; j=2,3; k=2,3 (1x2x2)
    add_block(2, 2, 2, 1, 2, 2, U_blk_122, V_blk_122, W_blk_122)
    
    # Block 12: i=2; j=2,3; k=4 (1x2x1)
    add_block(2, 2, 4, 1, 2, 1, U_blk_121, V_blk_121, W_blk_121)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    # Try structured decomposition first
    try:
        U_str, V_str, W_str = _structured_3x4x5()
        if _verify(U_str, V_str, W_str):
            return {"U": U_str, "V": V_str, "W": W_str}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U_fb, V_fb, W_fb = _schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
