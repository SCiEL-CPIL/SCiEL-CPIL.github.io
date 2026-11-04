"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Check shapes
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Build the multiplication tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need to check this for all (i,j,k)
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        # u_col has shape (n*m,), v_col has shape (m*p,), w_col has shape (p*n,)
        # Reshape: u -> (n,m), v -> (m,p), w -> (p,n)
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        w_mat = w_col.reshape(p, n)
        # T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # This is an outer product structure
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
    
    # For matrix multiplication, T[i,j,k] should be 1 for all (i,j,k)
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=1e-9)

def _strassen_4x4_4x4():
    """
    Return Strassen decomposition for 4x4 * 4x4 = 4x4.
    Rank 7 per 2x2 block, but for 4x4 we can use rank 49 (7*7) via block Strassen,
    or better: use the known rank-49 decomposition.
    
    Actually, the known result: 4x4*4x4 can be done with rank 49 (schoolbook for 4x4 blocks
    where each 2x2 block multiplication uses Strassen rank 7).
    
    But actually, the minimal known rank for 4x4*4x4 is 49 (using Strassen on 2x2 sub-blocks).
    Let's construct it properly.
    
    For a 4x4 matrix A partitioned into four 2x2 blocks:
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    C = A*B = [[C11, C12], [C21, C22]]
    
    Using Strassen on each 2x2 block multiplication:
    C11 = A11*B11 - A12*(B21-B11)  -> needs M(A11,B11), M(A12, B21-B11)
    C12 = A12*B22
    C21 = A21*B11
    C22 = A22*B22 - A21*(B12-B22)  -> needs M(A22,B22), M(A21, B12-B22)
    
    Wait, standard Strassen for 2x2*2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For 4x4*4x4, we have 4 block multiplications of 2x2*2x2, each with rank 7.
    Total rank = 4*7 = 28? No, that's not right either.
    
    Actually, the 4x4*4x4 matrix multiplication can be viewed as multiplying two 2x2 block matrices
    where each block is 2x2. The tensor for 4x4*4x4 is the tensor product of two 2x2*2x2 tensors.
    The rank of a tensor product is at most the product of ranks, so rank <= 7*7 = 49.
    And it's known that rank(4x4*4x4) = 49 (or possibly less, but 49 is the standard construction).
    
    Let me just use the schoolbook approach but with Strassen structure for the 2x2 sub-blocks.
    Actually, let me think more carefully.
    
    The simplest correct approach: use the known Strassen-based rank-49 decomposition for 4x4*4x4.
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # We'll use the tensor product structure.
    # The 4x4*4x4 tensor = (2x2*2x2) ⊗ (2x2*2x2)
    # Strassen for 2x2*2x2 has rank 7 with specific U,V,W matrices.
    
    # Strassen 2x2*2x2:
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r] for i,j,k in {0,1}, r in {0,...,6}
    # The 7 products are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Let me define the Strassen matrices for 2x2*2x2.
    # U2 has shape (4, 7), V2 has shape (4, 7), W2 has shape (4, 7)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # A11 is index 0 (i=0,j=0), A22 is index 3 (i=1,j=1)
    # B11 is index 0 (j=0,k=0), B22 is index 3 (j=1,k=1)
    # C11 gets M1: C11 is (i=0,k=0) -> W index 0*2+0 = 0
    U2[0, 0] = 1  # A11
    U2[3, 0] = 1  # A22
    V2[0, 0] = 1  # B11
    V2[3, 0] = 1  # B22
    W2[0, 0] = 1  # C11
    
    # M2 = (A21+A22)B11
    # A21 is index 2 (i=1,j=0), A22 is index 3 (i=1,j=1)
    # B11 is index 0
    # C21 gets M2: C21 is (i=1,k=0) -> W index 1*2+0 = 2
    U2[2, 1] = 1  # A21
    U2[3, 1] = 1  # A22
    V2[0, 1] = 1  # B11
    W2[2, 1] = 1  # C21
    
    # M3 = A11(B12-B22)
    # A11 is index 0
    # B12 is index 1 (j=0,k=1), B22 is index 3
    # C12 gets M3: C12 is (i=0,k=1) -> W index 0*2+1 = 1
    U2[0, 2] = 1  # A11
    V2[1, 2] = 1  # B12
    V2[3, 2] = -1  # B22
    W2[1, 2] = 1  # C12
    
    # M4 = A22(B21-B11)
    # A22 is index 3
    # B21 is index 2 (j=1,k=0), B11 is index 0
    # C11 gets M4: C11 is (i=0,k=0) -> W index 0
    # C21 gets M4: C21 is (i=1,k=0) -> W index 2
    U2[3, 3] = 1  # A22
    V2[2, 3] = 1  # B21
    V2[0, 3] = -1  # B11
    W2[0, 3] = 1  # C11
    W2[2, 3] = 1  # C21
    
    # M5 = (A11+A12)B22
    # A11 is index 0, A12 is index 1 (i=0,j=1)
    # B22 is index 3
    # C11 gets M5: C11 is (i=0,k=0) -> W index 0
    # C12 gets M5: C12 is (i=0,k=1) -> W index 1
    U2[0, 4] = 1  # A11
    U2[1, 4] = 1  # A12
    V2[3, 4] = 1  # B22
    W2[0, 4] = -1  # C11 (negative because C11 = M1+M4-M5+M7)
    W2[1, 4] = 1  # C12
    
    # M6 = (A21-A11)(B11+B12)
    # A21 is index 2, A11 is index 0
    # B11 is index 0, B12 is index 1
    # C22 gets M6: C22 is (i=1,k=1) -> W index 1*2+1 = 3
    U2[2, 5] = 1  # A21
    U2[0, 5] = -1  # A11
    V2[0, 5] = 1  # B11
    V2[1, 5] = 1  # B12
    W2[3, 5] = 1  # C22
    
    # M7 = (A12-A22)(B21+B22)
    # A12 is index 1, A22 is index 3
    # B21 is index 2, B22 is index 3
    # C11 gets M7: C11 is (i=0,k=0) -> W index 0
    # C22 gets M7: C22 is (i=1,k=1) -> W index 3
    U2[1, 6] = 1  # A12
    U2[3, 6] = -1  # A22
    V2[2, 6] = 1  # B21
    V2[3, 6] = 1  # B22
    W2[0, 6] = 1  # C11
    W2[3, 6] = 1  # C22
    
    # Now build the 4x4*4x4 decomposition using tensor product.
    # The 4x4 matrix is indexed as (i1,i2) where i1 in {0,1} is the block row, i2 in {0,1} is the row within block.
    # Similarly for j and k.
    # A[i,j] where i = i1*2+i2, j = j1*2+j2
    # B[j,k] where j = j1*2+j2, k = k1*2+k2
    # C[i,k] where i = i1*2+i2, k = k1*2+k2
    
    # The tensor product: for each pair (r1, r2) where r1,r2 in {0,...,6},
    # we have a rank-1 term in the 4x4*4x4 tensor.
    # U4[i1*2+i2, j1*2+j2, r1*7+r2] = U2[i1*2+i2, r1] * U2[j1*2+j2, r2]
    # Wait, that's not quite right. Let me think again.
    
    # Actually, the 4x4*4x4 tensor T[i,j,k] where i,j,k in {0,1,2,3}
    # = T2[i1,j1,k1] * T2[i2,j2,k2] where i=i1*2+i2, etc.
    # And T2[a,b,c] = sum_r U2[a,b,r] * V2[b,c,r] * W2[c,a,r]
    
    # So T[i,j,k] = sum_{r1,r2} U2[i1,j1,r1]*V2[j1,k1,r1]*W2[k1,i1,r1] * U2[i2,j2,r2]*V2[j2,k2,r2]*W2[k2,i2,r2]
    # = sum_{r1,r2} [U2[i1,j1,r1]*U2[i2,j2,r2]] * [V2[j1,k1,r1]*V2[j2,k2,r2]] * [W2[k1,i1,r1]*W2[k2,i2,r2]]
    
    # So for the 4x4*4x4 decomposition:
    # U4[i,j, r1*7+r2] = U2[i1,j1,r1] * U2[i2,j2,r2] where i=i1*2+i2, j=j1*2+j2
    # V4[j,k, r1*7+r2] = V2[j1,k1,r1] * V2[j2,k2,r2] where j=j1*2+j2, k=k1*2+k2
    # W4[k,i, r1*7+r2] = W2[k1,i1,r1] * W2[k2,i2,r2] where k=k1*2+k2, i=i1*2+i2
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U4_val = U2[i1 * 2 + i2, r1] * U2[j1 * 2 + j2, r2]
                            U[i * 4 + j, r] = U4_val
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    V4_val = V2[j1 * 2 + j2, k1] * V2[j2 * 2 + k2, r2]
                                    W4_val = W2[k1 * 2 + i1, r1] * W2[k2 * 2 + i2, r2]
                                    V[j * 4 + k, r] = V4_val
                                    W[k * 4 + i, r] = W4_val
    
    return U, V, W

def _schoolbook_4x4_4x3():
    """Schoolbook decomposition for 4x4 * 4x3 = 4x3. Rank = 4*4*3 = 48."""
    n, m, p = 4, 4, 3
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _combine(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Combine two decompositions:
    First: n1 x m1 * m1 x p1 (for the first part)
    Second: n2 x m2 * m2 x p2 (for the second part)
    
    We want to compute (n1+n2) x m * m x (p1+p2) where m = m1 = m2.
    Actually, for our case: 4x4 * 4x7 = (4x4*4x4) + (4x4*4x3)
    The first part computes C1 = A * B1 where B1 is 4x4
    The second part computes C2 = A * B2 where B2 is 4x3
    
    U has shape (n*m, rank) = (4*4, rank) = (16, rank)
    V has shape (m*p, rank) = (4*7, rank) = (28, rank)
    W has shape (p*n, rank) = (7*4, rank) = (28, rank)
    
    For the first part (4x4*4x4):
    U1 has shape (16, 49), V1 has shape (16, 49), W1 has shape (16, 49)
    
    For the second part (4x4*4x3):
    U2 has shape (16, 48), V2 has shape (12, 48), W2 has shape (12, 48)
    
    Combined:
    U = [U1, U2] horizontally, shape (16, 49+48) = (16, 97)
    V = [V1, V2] horizontally, shape (28, 97)
       V1 occupies rows 0-15 (for k=0..3 in the first 4 columns of B)
       V2 occupies rows 16-27 (for k=4..6 in the last 3 columns of B)
       So V[0:16, 0:49] = V1, V[16:28, 49:97] = V2
    W = [W1, W2] horizontally, shape (28, 97)
       W1 occupies rows 0-15 (for k=0..3)
       W2 occupies rows 16-27 (for k=4..6)
       So W[0:16, 0:49] = W1, W[16:28, 49:97] = W2
    """
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 + rank2
    
    n = n1  # = n2 = 4
    m = m1  # = m2 = 4
    p = p1 + p2  # = 4 + 3 = 7
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Copy U1 and U2
    U[:, :rank1] = U1
    U[:, rank1:] = U2
    
    # Copy V1 (for first p1 columns) and V2 (for next p2 columns)
    V[:m * p1, :rank1] = V1
    V[m * p1:, rank1:] = V2
    
    # Copy W1 (for first p1 columns) and W2 (for next p2 columns)
    W[:p1 * n, :rank1] = W1
    W[p1 * n:, rank1:] = W2
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Use Strassen-based decomposition for 4x4*4x4 (rank 49)
    # and schoolbook for 4x4*4x3 (rank 48)
    # Total rank = 49 + 48 = 97
    
    U1, V1, W1 = _strassen_4x4_4x4()
    U2, V2, W2 = _schoolbook_4x4_4x3()
    
    U, V, W = _combine(U1, V1, W1, U2, V2, W2, 4, 4, 4, 4, 4, 3)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
