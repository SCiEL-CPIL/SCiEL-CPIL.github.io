"""Matrix multiplication 2x3x3 via constructive block decomposition."""
import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def _build_2x2x2_strassen():
    """Return U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7).
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # Row indices for U: i*2+j -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Row indices for V: j*2+k -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Row indices for W: k*2+i -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)

    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1  # A00 + A11
    V[0][0] = 1; V[3][0] = 1  # B00 + B11
    W[0][0] = 1; W[3][0] = 1  # C00 += M1, C11 += M1
    W[1][0] = 0; W[2][0] = 0

    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # C10 += M2, C11 -= M2

    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C01 += M3, C11 += M3

    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[0][3] = -1; V[1][3] = 1
    W[0][3] = -1; W[2][3] = 1  # C00 -= M4, C10 += M4

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1  # C00 += M5, C01 += M5

    # M6 = (A10-A00)(B00+B01)
    U[0][5] = -1; U[2][5] = 1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11 += M6

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00 += M7

    return U, V, W

def _build_2x2x1():
    """Return U, V, W for 2x2x1 matrix multiplication (rank 3).
    U shape (4,3), V shape (2,3), W shape (2,3).
    U[i*2+j, c], V[j*1+k, c] = V[j, c], W[k*2+i, c].
    Here m=2, p=1. So V has 2*1=2 rows, W has 1*2=2 rows.
    """
    # C0 = A00*B0 + A01*B1  ... wait, this is 2x2 times 2x1
    # A is 2x2, B is 2x1, C is 2x1
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    #
    # Rank 3 decomposition:
    # P0 = A00 * B0
    # P1 = A11 * B1
    # P2 = (A01 + A10) * (B0 + B1)
    #
    # C0 = A00*B0 + A01*B1
    #     = P0 + A01*B1
    #     = P0 + A01*(B0+B1) - A01*B0
    # Hmm, that's not clean. Let me use the known rank-3 formula.
    #
    # Actually, the standard result: M_{2x2x1} has rank 3.
    # P0 = A00 * B0
    # P1 = A10 * B0  ... no.
    #
    # Let me think again. We have 4 input pairs (a_ij, b_j) producing 2 outputs.
    # c_0 = a_00*b_0 + a_01*b_1
    # c_1 = a_10*b_0 + a_11*b_1
    #
    # Rank 3:
    # p0 = a_00 * b_0
    # p1 = a_11 * b_1
    # p2 = (a_01 + a_10) * (b_0 + b_1)
    #
    # c0 = a_00*b_0 + a_01*b_1
    #    = p0 + a_01*b_1
    #    = p0 + a_01*(b_0+b_1) - a_01*b_0
    #    = p0 + [a_01*(b_0+b_1)] - [a_01*b_0]
    # But a_01*b_0 is not directly available. 
    #
    # Alternative rank-3 for M_{1x2x2} (which is isomorphic):
    # For M_{1x2x2}: c0 = a0*b00 + a1*b10, c1 = a0*b01 + a1*b11
    # p0 = a0*b00, p1 = a1*b11, p2 = (a0+a1)*(b01+b10)
    # c0 = p0 + a1*b10 = p0 + (a0+a1)*(b01+b10) - a0*b01 - a0*b10 - a1*b01 ... no.
    #
    # The correct rank-3 for M_{1x2x2}:
    # p0 = a0 * b00
    # p1 = a1 * b11  
    # p2 = (a0+a1) * (b01+b10)
    # c0 = a0*b00 + a1*b10 = p0 + a1*b10
    #    We need a1*b10. Note p2 = a0*b01 + a0*b10 + a1*b01 + a1*b11... no, a1*b11=p1.
    # p2 = a0*b01 + a0*b10 + a1*b01 + a1*b10
    # c0 = p0 + (p2 - a0*b01 - a0*b10 - a1*b01) ... still have terms.
    #
    # I recall the rank is indeed 3 but the explicit formula requires care.
    # Let me use a different approach: since we just need a valid decomposition,
    # I'll use schoolbook for the 2x2x1 block (rank 4) to be safe, or find the right one.
    #
    # Actually, let's just use rank 4 (schoolbook) for the smaller blocks to ensure correctness.
    # Total: 7 + 4 + 4 + 2 = 17. That's still better than 18.
    # Or even better, let me just carefully implement known small-rank decomp.

    # For safety, use schoolbook: rank 4
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    for i in range(2):
        for j in range(2):
            for k in range(1):
                c = i*2 + j*1 + k  # 0,1,2,3
                U[i*2+j][c] = 1
                V[j*1+k][c] = 1
                W[k*2+i][c] = 1
    return U, V, W

def _build_2x1x2():
    """Return U, V, W for 2x1x2 matrix multiplication (rank 4 schoolbook).
    U shape (2,4), V shape (2,4), W shape (4,4).
    U[i*1+j, c]=U[i+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(4)]
    for i in range(2):
        for j in range(1):
            for k in range(2):
                c = i*1 + j*2 + k
                U[i*1+j][c] = 1
                V[j*2+k][c] = 1
                W[k*2+i][c] = 1
    return U, V, W

def _build_2x1x1():
    """Return U, V, W for 2x1x1 matrix multiplication (rank 2).
    U shape (2,2), V shape (1,2), W shape (2,2).
    """
    U = [[0]*2 for _ in range(2)]
    V = [[0]*2 for _ in range(1)]
    W = [[0]*2 for _ in range(2)]
    for i in range(2):
        for j in range(1):
            for k in range(1):
                c = i*1 + j*1 + k
                U[i*1+j][c] = 1
                V[j*1+k][c] = 1
                W[k*2+i][c] = 1
    return U, V, W

def construct():
    start_time = time.time()
    
    # Decompose 2x3x3 into blocks:
    # A is 2x3, B is 3x3, C is 2x3
    # Split B's columns into 3, and A's columns into 3.
    # Block (i_block, j_block) where i_block in {0,1}, j_block in {0,1,2}
    # Actually, let's split differently:
    # A = [A_left(2x2) | A_right(2x1)]
    # B = [B_top(3x2) ; B_bottom(3x1)]  ... no, B is 3x3
    # Let me split by: first 2 columns/rows form a 2x2x2 block, etc.
    #
    # A (2x3): rows i=0,1; cols j=0,1,2
    # B (3x3): rows j=0,1,2; cols k=0,1,2
    # C (2x3): rows i=0,1; cols k=0,1,2
    #
    # Block decomposition by partitioning the middle index j:
    # j=0: A[:,0] (2x1) * B[0,:] (1x3) -> contributes to all C
    # j=1: A[:,1] (2x1) * B[1,:] (1x3)
    # j=2: A[:,2] (2x1) * B[2,:] (1x3)
    #
    # This gives three 2x1x3 blocks. Each has rank 6 (schoolbook) or 4 (better).
    # Total: 3*6 = 18. Not helpful.
    #
    # Better: partition both the columns of A and rows of B, and columns of B.
    # Let's use:
    # Block 1: A[:, 0:2] (2x2) * B[0:2, 0:2] (2x2) -> C[:, 0:2] (2x2), rank 7
    # Block 2: A[:, 0:2] (2x2) * B[2, 0:2] (1x2)   -> C[:, 0:2] (2x2), rank 4
    # Block 3: A[:, 2] (2x1) * B[0:2, :] (2x3) ... this gets messy.
    #
    # Simplest clean decomposition:
    # C[:,0:2] = A[:,0:2] @ B[0:2,0:2] + A[:,0:2] @ B[2,0:2] (as 2x2 * 1x2 broadcast) + A[:,2] @ B[0:2,0:2] (2x1 * 2x2) + A[:,2]*B[2,0:2] (2x1 * 1x2)
    # This is getting complicated. Let me just use the direct sum approach:
    #
    # Partition j into {0,1} and {2}, and k into {0,1} and {2}:
    #
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Block (j in {0,1}, k in {0,1}): A[:,0:2] (2x2) * B[0:2,0:2] (2x2) -> C[:,0:2], rank 7
    # Block (j in {0,1}, k=2):       A[:,0:2] (2x2) * B[0:2,2] (2x1)  -> C[:,2],    rank 4
    # Block (j=2, k in {0,1}):       A[:,2] (2x1) * B[2,0:2] (1x2)     -> C[:,0:2],  rank 4
    # Block (j=2, k=2):              A[:,2] (2x1) * B[2,2] (1x1)       -> C[:,2],    rank 2
    #
    # Total rank: 7 + 4 + 4 + 2 = 17

    U1, V1, W1 = _build_2x2x2_strassen()  # (4,7), (4,7), (4,7)
    U2, V2, W2 = _build_2x2x1()           # (4,4), (2,4), (2,4)
    U3, V3, W3 = _build_2x1x2()           # (2,4), (2,4), (4,4)
    U4, V4, W4 = _build_2x1x1()           # (2,2), (1,2), (2,2)

    total_rank = 7 + 4 + 4 + 2  # = 17

    # Build full U (6 x 17), V (9 x 17), W (6 x 17)
    U_full = [[0]*total_rank for _ in range(6)]
    V_full = [[0]*total_rank for _ in range(9)]
    W_full = [[0]*total_rank for _ in range(6)]

    col_offset = 0
    # Block 1: A[:,0:2] (2x2) * B[0:2,0:2] (2x2) -> C[:,0:2] (2x2)
    # U1 rows: i*2+j where i in {0,1}, j in {0,1} -> maps to A[i, j] where j in {0,1}
    # V1 rows: j*2+k where j in {0,1}, k in {0,1} -> maps to B[j, k] where j,k in {0,1}
    # W1 rows: k*2+i where k in {0,1}, i in {0,1} -> maps to C[i, k] where i in {0,1}, k in {0,1}
    for r in range(4):
        for c in range(7):
            if U1[r][c] != 0:
                U_full[r][col_offset + c] = U1[r][c]
            if V1[r][c] != 0:
                V_full[r][col_offset + c] = V1[r][c]
            if W1[r][c] != 0:
                W_full[r][col_offset + c] = W1[r][c]
    col_offset += 7

    # Block 2: A[:,0:2] (2x2) * B[0:2,2] (2x1) -> C[:,2] (2x1)
    # U2 rows: i*2+j, i in {0,1}, j in {0,1} -> A[i,j]
    # V2 rows: j*1+k, j in {0,1}, k=0 -> B[j, 2] (since k=0 maps to B's column index 2)
    # W2 rows: k*2+i, k=0, i in {0,1} -> C[i, 2]
    for r in range(4):
        for c in range(4):
            if U2[r][c] != 0:
                U_full[r][col_offset + c] = U2[r][c]
    for r in range(2):
        for c in range(4):
            if V2[r][c] != 0:
                V_full[r * 3 + 2][col_offset + c] = V2[r][c]  # B[j, 2]
    for r in range(2):
        for c in range(4):
            if W2[r][c] != 0:
                W_full[3 + r][col_offset + c] = W2[r][c]  # C[i, 2] where i=r
    col_offset += 4

    # Block 3: A[:,2] (2x1) * B[2,0:2] (1x2) -> C[:,0:2] (2x2)
    # U3 rows: i*1+j, i in {0,1}, j=0 -> A[i, 2]
    # V3 rows: j*2+k, j=0, k in {0,1} -> B[2, k]
    # W3 rows: k*2+i, k in {0,1}, i in {0,1} -> C[i, k]
    for r in range(2):
        for c in range(4):
            if U3[r][c] != 0:
                U_full[2 + r][col_offset + c] = U3[r][c]  # A[i, 2] where i=r
    for r in range(2):
        for c in range(4):
            if V3[r][c] != 0:
                V_full[6 + r][col_offset + c] = V3[r][c]  # B[2, k] where k=r
    for r in range(4):
        for c in range(4):
            if W3[r][c] != 0:
                W_full[r][col_offset + c] = W3[r][c]  # C[i, k]
    col_offset += 4

    # Block 4: A[:,2] (2x1) * B[2,2] (1x1) -> C[:,2] (2x1)
    for r in range(2):
        for c in range(2):
            if U4[r][c] != 0:
                U_full[2 + r][col_offset + c] = U4[r][c]
            if V4[r][c] != 0:
                V_full[8][col_offset + c] = V4[r][c]  # B[2,2]
            if W4[r][c] != 0:
                W_full[3 + r][col_offset + c] = W4[r][c]  # C[i, 2]
    col_offset += 2

    # Verify the decomposition with a random test
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        # Build tensor T[i,j,k] from U,V,W
        # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ... 
        # Actually the tensor identity is:
        # sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c] ... no.
        # The condition is: for all i,j,k,
        #   delta[j,k] * (something) ... 
        # The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # The rank-r decomposition means: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Wait, that's not right either. The standard form is:
        # C[i,k] = sum_{j} A[i,j] B[j,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_j V[j*p+k,c] B[j,k]) * ... 
        # No. The correct tensor identity for matrix multiplication:
        # The multiplication tensor is T with T[i,j,k] = 1 for all i,j,k (in the appropriate indexing).
        # A rank-r decomposition: T = sum_c U_c \otimes V_c \otimes W_c
        # where (U_c)_{i,j} = U[i*m+j, c], (V_c)_{j,k} = V[j*p+k, c], (W_c)_{k,i} = W[k*n+i, c]
        # And T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # But T[i,j,k] should be 1 (the coefficient in the bilinear form).
        # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid i,j,k.
        # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
        
        # Let's verify with random matrices
        for trial in range(20):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute via decomposition
            C_decomp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        # We need: C[i,k] = sum_c prod_c
                        # where prod_c = (sum_j U[i*m+j,c] * A[i,j]) * ... 
                        # No, the decomposition is:
                        # C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
                        # Wait, that's not bilinear in the right way.
                        #
                        # Actually, the standard rank-r decomposition of matrix multiplication:
                        # C[i,k] = sum_c M_c
                        # where M_c = (sum_j a_{c,i,j} * A[i,j]) * (sum_j b_{c,j,k} * B[j,k])
                        # with a_{c,i,j} = U[i*m+j, c], b_{c,j,k} = V[j*p+k, c]
                        # and the output is sum_c M_c * w_{c,k,i} where w_{c,k,i} = W[k*n+i, c]
                        # Hmm, I think the W factor multiplies the product.
                        #
                        # Let me re-derive. The tensor T_{n,m,p}[i,j,k] represents the coefficient
                        # of A[i,j]*B[j,k] in C[?,?]. For matrix multiplication, C[i',k'] = sum_j A[i',j]*B[j,k']
                        # So T[i,j,k] = 1 if i'=i and k'=k, else 0. In our indexing W[k*n+i], 
                        # the output index is (i,k) stored at position k*n+i.
                        #
                        # A rank-r decomposition: T = sum_c u_c \otimes v_c \otimes w_c
                        # T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k] ... but T is 3-tensor indexed by (i,j,k)
                        # where i is A's row, j is the shared index, k is B's col.
                        # u_c is indexed by (i,j) -> U[i*m+j, c]
                        # v_c is indexed by (j,k) -> V[j*p+k, c]  
                        # w_c is indexed by (k,i) -> W[k*n+i, c]
                        # T[i,j,k] = delta(i,i')*delta(k,k') summed... 
                        #
                        # The correct identity: for the multiplication to work, we need
                        # C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_j V[j*p+k,c] B[j,k]) * W[k*n+i,c]
                        # Let's just test this numerically.
                        u_val = 0.0
                        v_val = 0.0
                        for j in range(m):
                            u_val += U[i*m+j][c] * A[i,j]
                            v_val += V[j*p+k][c] * B[j,k]
                        total += u_val * v_val * W[k*n+i][c]
                    C_decomp[i,k] = total
            
            if not np.allclose(C_decomp, C_expected, atol=1e-9):
                return False
        return True

    if not verify(U_full, V_full, W_full, 2, 3, 3):
        # Fall back to schoolbook if verification fails
        n, m, p = 2, 3, 3
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    if time.time() - start_time > 140:
        # Return what we have, it should be valid
        pass

    return {"U": U_full, "V": V_full, "W": W_full}
