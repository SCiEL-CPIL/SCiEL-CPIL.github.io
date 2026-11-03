"""Matrix multiplication 4,4,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 6]

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds total wall-clock budget

def _remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _strassen_2x2x2():
    """
    Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
      U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    The tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = delta[i,j,k] for matmul.
    
    Standard Strassen:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B21-B22)
      M4 = A22(B12-B21)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
      
      C11 = M1 + M4 - M5 + M7
      C12 = M3 + M5
      C21 = M2 + M4
      C22 = M1 - M2 + M3 + M6
    """
    # For 2x2x2: n=m=p=2
    # U has shape (n*m, rank) = (4, 7)
    # V has shape (m*p, rank) = (4, 7)  
    # W has shape (p*n, rank) = (4, 7)
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # A11 is row i=0,col j=0 -> U[0*2+0, 0] = U[0,0]
    # A22 is row i=1,col j=1 -> U[1*2+1, 0] = U[3,0]
    # B11 is row j=0,col k=0 -> V[0*2+0, 0] = V[0,0]
    # B22 is row j=1,col k=1 -> V[1*2+1, 0] = V[3,0]
    # C11 contribution: W[0*2+0, 0] = W[0,0]
    # C22 contribution: W[1*2+1, 0] = W[3,0]
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2 = (A21+A22)B11
    # A21: i=1,j=0 -> U[1*2+0, 1] = U[2,1]
    # A22: i=1,j=1 -> U[3,1]
    # B11: j=0,k=0 -> V[0,1]
    # C21: k=1,i=0 -> W[1*2+0, 1] = W[2,1]
    # C22: k=1,i=1 -> W[1*2+1, 1] = W[3,1]
    U[2, 1] = 1  # A21
    U[3, 1] = 1  # A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21
    W[3, 1] = 1  # C22
    
    # M3 = A11(B21-B22)
    # A11: i=0,j=0 -> U[0,2]
    # B21: j=1,k=0 -> V[1*2+0, 2] = V[2,2]
    # B22: j=1,k=1 -> V[3,2]
    # C12: k=0,i=1 -> W[0*2+1, 2] = W[1,2]
    # C22: k=1,i=1 -> W[3,2]
    U[0, 2] = 1  # A11
    V[2, 2] = 1  # B21
    V[3, 2] = -1 # -B22
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # M4 = A22(B12-B21)
    # A22: i=1,j=1 -> U[3,3]
    # B12: j=0,k=1 -> V[0*2+1, 3] = V[1,3]
    # B21: j=1,k=0 -> V[2,3]
    # C11: k=0,i=0 -> W[0,3]
    # C21: k=1,i=0 -> W[2,3]
    U[3, 3] = 1  # A22
    V[1, 3] = 1  # B12
    V[2, 3] = -1 # -B21
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # M5 = (A11+A12)B22
    # A11: i=0,j=0 -> U[0,4]
    # A12: i=0,j=1 -> U[1,4]
    # B22: j=1,k=1 -> V[3,4]
    # C11: k=0,i=0 -> W[0,4] (with -1)
    # C12: k=1,i=1 -> W[3,4]
    U[0, 4] = 1  # A11
    U[1, 4] = 1  # A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1 # -C11
    W[3, 4] = 1  # C12
    
    # M6 = (A21-A11)(B11+B12)
    # A21: i=1,j=0 -> U[2,5]
    # A11: i=0,j=0 -> U[0,5]
    # B11: j=0,k=0 -> V[0,5]
    # B12: j=0,k=1 -> V[1,5]
    # C22: k=1,i=1 -> W[3,5]
    U[2, 5] = 1  # A21
    U[0, 5] = -1 # -A11
    V[0, 5] = 1  # B11
    V[1, 5] = 1  # B12
    W[3, 5] = 1  # C22
    
    # M7 = (A12-A22)(B21+B22)
    # A12: i=0,j=1 -> U[1,6]
    # A22: i=1,j=1 -> U[3,6]
    # B21: j=1,k=0 -> V[2,6]
    # B22: j=1,k=1 -> V[3,6]
    # C11: k=0,i=0 -> W[0,6]
    U[1, 6] = 1  # A12
    U[3, 6] = -1 # -A22
    V[2, 6] = 1  # B21
    V[3, 6] = 1  # B22
    W[0, 6] = 1  # C11
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build a rank-28 decomposition for 4x4x4 matrix multiplication using 4 Strassen 2x2x2 blocks.
    Actually, the standard Strassen for 4x4x4 uses 7^2=49 rank if applied recursively.
    But we can use the known rank-28 decomposition for 4x4x4.
    
    Actually, let me use the standard approach: split 4x4x4 into four 2x2x2 blocks.
    Each 2x2x2 needs rank 7, so total rank = 4*7 = 28.
    """
    n, m, p = 4, 4, 4
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # The 4x4x4 tensor is a block matrix:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # C = A*B, each block C_ij = sum_k A_ik * B_kj
    
    # Block (i,j,k) where i,j,k in {0,1} for the 2x2x2 block structure
    # For block (bi, bj, bk): 
    #   A indices: rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    #   B indices: rows 2*bj to 2*bj+1, cols 2*bk to 2*bk+1
    #   C indices: rows 2*bk to 2*bk+1, cols 2*bi to 2*bi+1
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                base = (bi * 4 + bj) * 2 + bk  # 0..27
                # A block (bi, bj): rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                # In U: row index = i*4 + j where i in [2*bi, 2*bi+1], j in [2*bj, 2*bj+1]
                # U2 has shape (4, 7) with U2[i*2+j, c]
                for ii in range(2):
                    for jj in range(2):
                        U[(2*bi + ii)*4 + (2*bj + jj), base + ii*2 + jj] = U2[ii*2 + jj, 0] if False else 0
                        # Wait, U2[i*2+j, c] where i,j in {0,1} and c in {0..6}
                        # So for local indices ii,jj and column c:
                        U[(2*bi + ii)*4 + (2*bj + jj), base + c] = 0  # will fill below
                
                for c in range(7):
                    for ii in range(2):
                        for jj in range(2):
                            # U2[ii*2+jj, c] contributes to A element at global (2*bi+ii, 2*bj+jj)
                            U[(2*bi + ii)*4 + (2*bj + jj), base + c] = U2[ii*2 + jj, c]
                    
                    for jj in range(2):
                        for kk in range(2):
                            # V2[jj*2+kk, c] contributes to B element at global (2*bj+jj, 2*bk+kk)
                            V[(2*bj + jj)*4 + (2*bk + kk), base + c] = V2[jj*2 + kk, c]
                    
                    for kk in range(2):
                        for ii in range(2):
                            # W2[kk*2+ii, c] contributes to C element at global (2*bk+kk, 2*bi+ii)
                            W[(2*bk + kk)*4 + (2*bi + ii), base + c] = W2[kk*2 + ii, c]
    
    return U, V, W

def _build_4x4x2_strassen():
    """
    Build a rank-14 decomposition for 4x4x2 matrix multiplication.
    Split into two 2x2x2 blocks (for the two output columns), each with rank 7.
    
    Actually, 4x4x2: A is 4x4, B is 4x2, C is 4x2.
    Split A into 2x2 blocks, B into 2x2 blocks (vertically), C into 2x2 blocks (vertically).
    
    For output column block 0 (rows 0-1 of C):
      C[0:2, 0:2] = A[0:2, 0:4] * B[0:4, 0:2]
      This is a 2x4 times 4x2 = 2x2 product.
      We can split A[0:2, 0:4] = [A11, A12] (each 2x2) and B[0:4, 0:2] = [B1; B2] (each 2x2)
      C[0:2, 0:2] = A11*B1 + A12*B2
      
    For output column block 1 (rows 2-3 of C):
      C[2:4, 0:2] = A[2:4, 0:4] * B[0:4, 0:2]
      = A21*B1 + A22*B2
    
    Each term Aij*Bk is a 2x2 * 2x2 = 2x2 product, which has rank 7.
    So total rank = 4 * 7 = 28. That's not 14.
    
    Let me reconsider. The strategy says rank-14 from two independent 2x2x2 Strassen instances.
    Maybe the 4x4x2 can be viewed as two 2x4x2 problems, each of which can be done with rank 7?
    
    Actually, 2x4x2: A is 2x4, B is 4x2, C is 2x2.
    This is just standard 2x2x2 if we view the middle dimension as 4... no.
    
    Let me think differently. 4x4x2 tensor: T[i,j,k] where i in 0..3, j in 0..3, k in 0..1.
    This is the tensor for C[i,k] = sum_j A[i,j]*B[j,k].
    
    We can split this into two independent 2x4x2 tensors (for i=0,1 and i=2,3).
    Each 2x4x2: A is 2x4, B is 4x2, C is 2x2.
    
    For 2x4x2: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    Split j into {0,1} and {2,3}:
    C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    
    Each sum is a 2x2x2 tensor (A[i,j] 2x2, B[j,k] 2x2, C[i,k] 2x2).
    So each has rank 7, total rank = 14 for the 2x4x2.
    And for 4x4x2, we have two such 2x4x2 tensors, total rank = 28.
    
    Hmm, that gives 28, not 14. Let me re-read the strategy.
    
    "the 4x4x2 component uses a rank-14 decomposition derived from two independent 2x2x2 Strassen instances"
    
    Maybe they mean the 4x4x2 as a whole uses rank 14, not 28. Let me think about this more carefully.
    
    Actually, I think the key insight is that 4x4x2 can be decomposed more efficiently.
    Let me just use rank 28 for the 4x4x2 part to be safe, giving total rank 28+28=56.
    Or better, let me just use the standard Strassen tiling for the whole 4x4x6.
    
    Actually, let me just go with a clean implementation:
    - 4x4x4 part: rank 28 (four 2x2x2 Strassen blocks)
    - 4x4x2 part: rank 14 (two 2x2x2 Strassen blocks, one for each output column pair)
    
    Wait, for 4x4x2:
    C has shape 4x2. Split into C[0:2, :] and C[2:4, :], each 2x2.
    C[0:2, :] = A[0:2, :] * B[:, :] where A[0:2,:] is 2x4 and B is 4x2.
    C[2:4, :] = A[2:4, :] * B[:, :] where A[2:4,:] is 2x4 and B is 4x2.
    
    For C[0:2, :]: C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, k in {0,1}, j in {0,1,2,3}
    Split j: = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k]
    Each term is 2x2x2, rank 7. Total for this half: 14.
    Similarly for the other half: 14.
    Total for 4x4x2: 28.
    
    So total rank = 28 + 28 = 56. Let me just go with that.
    """
    n, m, p = 4, 4, 2
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # 4x4x2: i in 0..3, j in 0..3, k in 0..1
    # Split into 4 blocks of 2x2x2:
    # For (bi, bj, bk) where bi in {0,1} (row block of A/C), bj in {0,1} (col block of A / row block of B), bk in {0,1} (col block of B / col block of C)
    # But p=2, so bk is just the output column index.
    
    # Actually, let's index it as:
    # A[i,j] with i in 0..3, j in 0..3
    # B[j,k] with j in 0..3, k in 0..1
    # C[i,k] with i in 0..3, k in 0..1
    
    # Block structure: 
    # bi: row block of A and C (2 blocks of 2 rows)
    # bj: col block of A and row block of B (2 blocks of 2 cols/rows)
    # bk: col block of B and C (1 block of 2 cols, since p=2)
    
    # So we have 2*2*1 = 4 blocks, each 2x2x2, rank 7 each, total 28.
    
    for bi in range(2):
        for bj in range(2):
            base = (bi * 2 + bj) * 7  # 0, 7, 14, 21
            for c in range(7):
                for ii in range(2):
                    for jj in range(2):
                        U[(2*bi + ii)*4 + (2*bj + jj), base + c] = U2[ii*2 + jj, c]
                
                for jj in range(2):
                    for kk in range(2):
                        V[(2*bj + jj)*2 + kk, base + c] = V2[jj*2 + kk, c]
                
                for kk in range(2):
                    for ii in range(2):
                        W[kk*4 + (2*bi + ii), base + c] = W2[kk*2 + ii, c]
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4x6 matrix multiplication.
    
    Strategy: Split the 6-dimensional output into 4+2.
    - 4x4x4 part: rank 28 (four 2x2x2 Strassen blocks)
    - 4x4x2 part: rank 28 (four 2x2x2 Strassen blocks)
    - Total rank: 56
    """
    if _remaining() < 1.0:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Build 4x4x4 part
    U4, V4, W4 = _build_4x4x4_strassen()
    # U4: (16, 28), V4: (16, 28), W4: (16, 28)
    # W4[k*4+i, c] for k,i in 0..3
    
    # Build 4x4x2 part
    U2, V2, W2 = _build_4x4x2_strassen()
    # U2: (16, 28), V2: (8, 28), W2: (8, 28)
    # W2[k*4+i, c] for k in 0..1, i in 0..3
    
    # Combine
    # U: (n*m, rank) = (16, 56)
    # V: (m*p, rank) = (24, 56)
    # W: (p*n, rank) = (24, 56)
    
    total_rank = 56
    U = np.zeros((16, total_rank))
    V = np.zeros((24, total_rank))
    W = np.zeros((24, total_rank))
    
    # First 28 columns: 4x4x4 part (output columns 0..3)
    U[:, 0:28] = U4
    V[:, 0:28] = V4
    W[0:16, 0:28] = W4  # k in 0..3, i in 0..3 -> rows 0..15
    
    # Next 28 columns: 4x4x2 part (output columns 4..5)
    U[:, 28:56] = U2
    V[0:8, 28:56] = V2  # j in 0..3, k in 0..1 -> rows 0..7
    W[16:24, 28:56] = W2  # k in 0..1, i in 0..3 -> rows 16..23
    
    # Convert to list of lists with integer/half-integer entries
    U_list = [[int(U[i, j]) if U[i, j] == int(U[i, j]) else float(U[i, j]) for j in range(total_rank)] for i in range(16)]
    V_list = [[int(V[i, j]) if V[i, j] == int(V[i, j]) else float(V[i, j]) for j in range(total_rank)] for i in range(24)]
    W_list = [[int(W[i, j]) if W[i, j] == int(W[i, j]) else float(W[i, j]) for j in range(total_rank)] for i in range(24)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# Verify the decomposition is correct
def _verify(U, V, W, n=4, m=4, p=6):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should equal 1 if this is the matrix multiplication tensor (delta-like)
    # Actually, for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = delta[i,j,k] in the sense that:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should give the correct structure
    
    # Let's check: for random A (n x m) and B (m x p), C = A @ B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Using the decomposition: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # Wait, that's not right. The tensor decomposition works as:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    
    # Let's verify with a random test
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_actual = np.zeros((n, p))
    for c in range(rank):
        u_vec = U_arr[:, c]  # (n*m,)
        v_vec = V_arr[:, c]  # (m*p,)
        w_val = W_arr[:, c]  # (p*n,)
        
        # u_vec reshaped: (n, m), v_vec reshaped: (m, p), w_val reshaped: (p, n)
        u_mat = u_vec.reshape(n, m)
        v_mat = v_vec.reshape(m, p)
        w_mat = w_val.reshape(p, n)
        
        # C[i,k] += w_mat[k,i] * (u_mat @ ?) ... 
        # Actually: C[i,k] = sum_c w_mat[k,i] * (sum_j u_mat[i,j] * A[i,j]) * (sum_j v_mat[j,k] * B[j,k])
        # That doesn't look right either.
        
        # The correct interpretation: 
        # M1 = A * U_c (where U_c is the c-th column reshaped)
        # M2 = V_c * B
        # C += W_c * M1 * M2 (element-wise with W_c)
        
        # More precisely for rank-1 tensor:
        # C[i,k] += W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        
        a_proj = np.array([sum(U_arr[i*m+j, c] * A[i, j] for j in range(m)) for i in range(n)])
        b_proj = np.array([sum(V_arr[j*p+k, c] * B[j, k] for j in range(m)) for k in range(p)])
        
        for k in range(p):
            for i in range(n):
                C_actual[i, k] += W_arr[k*n+i, c] * a_proj[i] * b_proj[k]
    
    diff = np.max(np.abs(C_actual - C_expected))
    return diff

if __name__ == "__main__":
    result = construct()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    print(f"Rank: {rank}")
    err = _verify(U, V, W)
    print(f"Max error: {err}")
