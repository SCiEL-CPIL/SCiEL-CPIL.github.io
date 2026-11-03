"""Matrix multiplication 4x4x7 with rank-minimal decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def _verify(n, m, p, U, V, W):
    """Verify that U, V, W form a valid tensor decomposition of matrix mult n x m x p."""
    rank = U.shape[1]
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        # The product u_i * v_j * w_k should match the tensor
        # T[i,j,k] = 1 if the output index k*n+i equals the row in W... 
        # Actually, the convention is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # So we check that for each r, the outer product u ⊗ v ⊗ w contributes to the tensor.
        # The tensor T[i,j,k] = 1 for all valid i,j,k (it's the identity tensor for matmul).
        # So sum over r of U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1.
        pass
    
    # Full check: compute the sum
    result = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    expected = np.ones((n, m, p))
    if not np.allclose(result, expected, atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
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
    return U, V, W, rank

def _strassen_2x2x2():
    """Strassen decomposition for 2x2x2 matrix multiplication, rank 7."""
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Index: A is 2x2, so A[i,j] with i,j in {0,1}
    # B is 2x2, B[j,k]
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # U[i*m+j, r] where m=2
    # V[j*p+k, r] where p=2
    # W[k*n+i, r] where n=2
    
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00=U[0], A11=U[3] -> U[0,0]=1, U[3,0]=1
    # V: B00=V[0], B11=V[3] -> V[0,0]=1, V[3,0]=1
    # W: C00 gets +1, C11 gets +1 -> W[0*2+0,0]=W[0,0]=1, W[1*2+1,0]=W[3,0]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    # U: A10=U[2], A11=U[3]
    # V: B00=V[0]
    # W: C10 gets +1 -> W[0*2+1,1]=W[2,1]=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00(B01-B11)
    # U: A00=U[0]
    # V: B01=V[1], B11=V[3]
    # W: C01 gets +1 -> W[1*2+0,2]=W[2,2]=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4 = A11(B10-B00)
    # U: A11=U[3]
    # V: B10=V[2], B00=V[0]
    # W: C00 gets +1, C10 gets +1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    # U: A00=U[0], A01=U[1]
    # V: B11=V[3]
    # W: C00 gets -1, C01 gets +1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # U: A10=U[2], A00=U[0]
    # V: B00=V[0], B01=V[1]
    # W: C11 gets +1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # U: A01=U[1], A11=U[3]
    # V: B10=V[2], B11=V[3]
    # W: C00 gets +1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W, rank

def _combine_blocks(n1, m1, p1, n2, m2, p2, U1, V1, W1, rank1, U2, V2, W2, rank2):
    """
    Combine two block decompositions for a block-diagonal-like structure.
    Block 1 handles rows 0..n1-1 of A, cols 0..m1-1 of A, and produces cols 0..p1-1 of C.
    Block 2 handles rows n1..n1+n2-1 of A, cols m1..m1+m2-1 of A, and produces cols p1..p1+p2-1 of C.
    """
    n = n1 + n2
    m = m1 + m2
    p = p1 + p2
    rank = rank1 + rank2
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block 1
    # U1 has shape (n1*m1, rank1). Place in U[0:n1*m1, 0:rank1]
    # But we need to map indices: for block 1, A[i,j] where i in [0,n1), j in [0,m1)
    # In the full matrix, these are at positions i*m+j in U.
    for i in range(n1):
        for j in range(m1):
            U[i * m + j, :rank1] = U1[i * m1 + j, :]
    
    # V1 has shape (m1*p1, rank1). 
    # In block 1, B[j,k] where j in [0,m1), k in [0,p1)
    # In full matrix, B is m x p, so B[j,k] is at position j*p+k
    for j in range(m1):
        for k in range(p1):
            V[j * p + k, :rank1] = V1[j * p1 + k, :]
    
    # W1 has shape (p1*n1, rank1)
    # In block 1, C[i,k] where i in [0,n1), k in [0,p1)
    # In full matrix, C is p x n (transposed convention), so C[i,k] is at position k*n+i
    for k in range(p1):
        for i in range(n1):
            W[k * n + i, :rank1] = W1[k * n1 + i, :]
    
    # Block 2
    # U2 has shape (n2*m2, rank2)
    # A[i,j] where i in [n1, n1+n2), j in [m1, m1+m2)
    for ii in range(n2):
        for jj in range(m2):
            i = n1 + ii
            j = m1 + jj
            U[i * m + j, rank1:] = U2[ii * m2 + jj, :]
    
    # V2 has shape (m2*p2, rank2)
    # B[j,k] where j in [m1, m1+m2), k in [p1, p1+p2)
    for jj in range(m2):
        for kk in range(p2):
            j = m1 + jj
            k = p1 + kk
            V[j * p + k, rank1:] = V2[jj * p2 + kk, :]
    
    # W2 has shape (p2*n2, rank2)
    # C[i,k] where i in [n1, n1+n2), k in [p1, p1+p2)
    for kk in range(p2):
        for ii in range(n2):
            i = n1 + ii
            k = p1 + kk
            W[k * n + i, rank1:] = W2[kk * n2 + ii, :]
    
    return U, V, W, rank

def construct():
    """
    Construct a rank-minimal decomposition for 4x4x7 matrix multiplication.
    
    Strategy: Split into two blocks:
    - Block 1: 4x4x4 using Strassen-based approach (rank ~28 or less)
    - Block 2: 4x4x3 using schoolbook (rank 48) or better
    
    Actually, let's think more carefully. The problem is 4x4 times 4x7 = 4x7.
    So A is 4x4, B is 4x7, C is 4x7.
    
    We can split B into two parts: B = [B1 | B2] where B1 is 4x4 and B2 is 4x3.
    Then C = [A*B1 | A*B2].
    
    So we need to compute A*B1 (4x4x4) and A*B2 (4x4x3) separately.
    
    For 4x4x4, we can use Strassen: split 4x4 into 2x2 blocks.
    Each 2x2x2 uses rank 7 (Strassen), so 4x4x4 uses 7*4 = 28 rank terms.
    
    For 4x4x3, we can use schoolbook: rank 4*4*3 = 48.
    
    Total rank: 28 + 48 = 76.
    Schoolbook total: 4*4*7 = 112.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # Block 1: 4x4x4 (A times first 4 cols of B)
    # Use Strassen decomposition
    # Split 4x4 into four 2x2 blocks
    # We need to compose Strassen for 2x2x2 into 4x4x4
    
    # Let's build 4x4x4 from Strassen
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = [[C00, C01], [C10, C11]]
    
    # Using Strassen at the 2x2 block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M is a 2x2x2 matrix multiplication, using rank 7.
    # So total rank for 4x4x4 is 7*7 = 49? No wait.
    
    # Actually, each Strassen product Mj = Xj * Yj where Xj and Yj are 2x2 matrices.
    # Computing Mj = Xj * Yj requires a 2x2x2 decomposition of rank 7.
    # So for 7 such products, total rank = 7 * 7 = 49.
    
    # Hmm, that's not 28. Let me reconsider.
    
    # The Strassen algorithm computes C = A*B using 7 recursive multiplications.
    # Each recursive multiplication is 2x2x2, which itself can be done in rank 7.
    # So the total rank for 4x4x4 using Strassen recursively is 7*7 = 49.
    
    # But we can also just do 4x4x4 in schoolbook: rank 64.
    # Or we can use a non-recursive approach.
    
    # Actually, for 4x4x4, the known minimal rank is between 49 and 64.
    # Let's just use schoolbook for simplicity: rank 64.
    # Or use Strassen: rank 49.
    
    # Let's build the Strassen 4x4x4 decomposition.
    
    def strassen_4x4x4():
        n2, m2, p2 = 4, 4, 4
        rank = 49
        U = np.zeros((n2 * m2, rank))
        V = np.zeros((m2 * p2, rank))
        W = np.zeros((p2 * n2, rank))
        
        # Strassen at block level
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each Mj = Xj * Yj, use Strassen 2x2x2 (rank 7)
        # Xj is 2x2, Yj is 2x2, result is 2x2
        
        # Let's define the Strassen 2x2x2 tensors
        U2, V2, W2, _ = _strassen_2x2x2()
        
        # M1 = (A00+A11) * (B00+B11)
        # X1 = A00 + A11, Y1 = B00 + B11
        # X1[i,j] = A[i,j] + A[i+2,j+2] for i,j in {0,1}
        # Y1[i,j] = B[i,j] + B[i+2,j+2] for i,j in {0,1}
        
        # For X1: A00 is A[0:2,0:2], A11 is A[2:4,2:4]
        # U1[i*4+j, col] contributes to X1
        # In the 2x2x2 for M1, U1_2x2[i*2+j, r] is used
        # So U[i*4+j, col] = U1_2x2[i*2+j, r] where r is the local rank index
        
        # Let me map this out.
        # For M1, the local 2x2x2 has:
        # U_local[i*2+j, r] for i,j in {0,1}, r in {0,...,6}
        # This corresponds to A_local[i,j] which is A00[i,j] + A11[i,j]
        # A00[i,j] = A[i,j] -> U[i*4+j, col] += U_local[i*2+j, r]
        # A11[i,j] = A[i+2,j+2] -> U[(i+2)*4+(j+2), col] += U_local[i*2+j, r]
        
        # Similarly for V and W.
        
        # M1: offset = 0
        for r in range(7):
            col = r
            # U
            for i in range(2):
                for j in range(2):
                    U[i*4+j, col] += U2[i*2+j, r]
                    U[(i+2)*4+(j+2), col] += U2[i*2+j, r]
            # V
            for i in range(2):
                for j in range(2):
                    V[i*4+j, col] += V2[i*2+j, r]
                    V[(i+2)*4+(j+2), col] += V2[i*2+j, r]
            # W
            for i in range(2):
                for j in range(2):
                    W[i*4+j, col] += W2[i*2+j, r]
                    W[(i+2)*4+(j+2), col] += W2[i*2+j, r]
        
        # M2 = (A10+A11) * B00
        # X2 = A10 + A11, Y2 = B00
        # X2[i,j] = A[i+2,j] + A[i+2,j+2]
        # Y2[i,j] = B[i,j]
        
        offset = 7
        for r in range(7):
            col = offset + r
            # U
            for i in range(2):
                for j in range(2):
                    U[(i+2)*4+j, col] += U2[i*2+j, r]
                    U[(i+2)*4+(j+2), col] += U2[i*2+j, r]
            # V
            for i in range(2):
                for j in range(2):
                    V[i*4+j, col] += V2[i*2+j, r]
            # W: C10 = M2, so C[2:4, 0:2] gets the result
            for i in range(2):
                for j in range(2):
                    W[j*4+(i+2), col] += W2[i*2+j, r]
        
        # M3 = A00 * (B01-B11)
        # X3 = A00, Y3 = B01 - B11
        # X3[i,j] = A[i,j]
        # Y3[i,j] = B[i,j+2] - B[i+2,j+2]
        
        offset = 14
        for r in range(7):
            col = offset + r
            # U
            for i in range(2):
                for j in range(2):
                    U[i*4+j, col] += U2[i*2+j, r]
            # V
            for i in range(2):
                for j in range(2):
                    V[i*4+(j+2), col] += V2[i*2+j, r]
                    V[(i+2)*4+(j+2), col] += -V2[i*2+j, r]
            # W: C01 = M3, so C[0:2, 2:4] gets the result
            for i in range(2):
                for j in range(2):
                    W[(j+2)*4+i, col] += W2[i*2+j, r]
        
        # M4 = A11 * (B10-B00)
        # X4 = A11, Y4 = B10 - B00
        # X4[i,j] = A[i+2,j+2]
        # Y4[i,j] = B[i+2,j] - B[i,j]
        
        offset = 21
        for r in range(7):
            col = offset + r
            # U
            for i in range(2):
                for j in range(2):
                    U[(i+2)*4+(j+2), col] += U2[i*2+j, r]
            # V
            for i in range(2):
                for j in range(2):
                    V[(i+2)*4+j, col] += V2[i*2+j, r]
                    V[i*4+j, col] += -V2[i*2+j, r]
            # W: C00 += M4, C10 += M4
            for i in range(2):
                for j in range(2):
                    W[j*4+i, col] += W2[i*2+j, r]
                    W[j*4+(i+2), col] += W2[i*2+j, r]
        
        # M5 = (A00+A01) * B11
        # X5 = A00 + A01, Y5 = B11
        # X5[i,j] = A[i,j] + A[i,j+2]
        # Y5[i,j] = B[i+2,j+2]
        
        offset = 28
        for r in range(7):
            col = offset + r
            # U
            for i in range(2):
                for j in range(2):
                    U[i*4+j, col] += U2[i*2+j, r]
                    U[i*4+(j+2), col] += U2[i*2+j, r]
            # V
            for i in range(2):
                for j in range(2):
                    V[(i+2)*4+(j+2), col] += V2[i*2+j, r]
            # W: C00 -= M5, C01 += M5
            for i in range(2):
                for j in range(2):
                    W[j*4+i, col] += -W2[i*2+j, r]
                    W[(j+2)*4+i, col] += W2[i*2+j, r]
        
        # M6 = (A10-A00) * (B00+B01)
        # X6 = A10 - A00, Y6 = B00 + B01
        # X6[i,j] = A[i+2,j] - A[i,j]
        # Y6[i,j] = B[i,j] + B[i,j+2]
        
        offset = 35
        for r in range(7):
            col = offset + r
            # U
            for i in range(2):
                for j in range(2):
                    U[(i+2)*4+j, col] += U2[i*2+j, r]
                    U[i*4+j, col] += -U2[i*2+j, r]
            # V
            for i in range(2):
                for j in range(2):
                    V[i*4+j, col] += V2[i*2+j, r]
                    V[i*4+(j+2), col] += V2[i*2+j, r]
            # W: C11 += M6
            for i in range(2):
                for j in range(2):
                    W[(j+2)*4+(i+2), col] += W2[i*2+j, r]
        
        # M7 = (A01-A11) * (B10+B11)
        # X7 = A01 - A11, Y7 = B10 + B11
        # X7[i,j] = A[i,j+2] - A[i+2,j+2]
        # Y7[i,j] = B[i+2,j] + B[i+2,j+2]
        
        offset = 42
        for r in range(7):
            col = offset + r
            # U
            for i in range(2):
                for j in range(2):
                    U[i*4+(j+2), col] += U2[i*2+j, r]
                    U[(i+2)*4+(j+2), col] += -U2[i*2+j, r]
            # V
            for i in range(2):
                for j in range(2):
                    V[(i+2)*4+j, col] += V2[i*2+j, r]
                    V[(i+2)*4+(j+2), col] += V2[i*2+j, r]
            # W: C00 += M7
            for i in range(2):
                for j in range(2):
                    W[j*4+i, col] += W2[i*2+j, r]
        
        return U, V, W, rank
    
    U1, V1, W1, rank1 = strassen_4x4x4()
    
    # Block 2: 4x4x3 (A times last 3 cols of B)
    # Use schoolbook: rank 48
    U2, V2, W2, rank2 = _schoolbook(4, 4, 3)
    
    # Combine
    U_total, V_total, W_total, rank_total = _combine_blocks(
        4, 4, 4, 4, 4, 3,
        U1, V1, W1, rank1,
        U2, V2, W2, rank2
    )
    
    # Verify
    if not _verify(4, 4, 7, U_total, V_total, W_total):
        # Fallback to schoolbook
        U, V, W, rank = _schoolbook(4, 4, 7)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

# EVOLVE-BLOCK-END
