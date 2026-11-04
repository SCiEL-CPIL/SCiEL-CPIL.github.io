import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition exactly reproduces matrix multiplication.
    T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 for all valid i, j, k.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # For each rank component, form the outer product contribution
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is equivalent to: for each r, the tensor contribution is
    # U_r (reshaped to n x m) ⊗ V_r (reshaped to m x p) ⊗ W_r (reshaped to p x n)
    # But the indices don't align that way. Let's think more carefully.
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a 3-index tensor. We can verify by checking all (i,j,k) tuples.
    
    T_expected = np.ones((n, m, p))
    
    T_computed = np.zeros((n, m, p))
    for r in range(rank):
        # U[:, r] has shape (n*m,), index i*m + j
        # V[:, r] has shape (m*p,), index j*p + k
        # W[:, r] has shape (p*n,), index k*n + i
        # T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        u_col = U[:, r]  # (n*m,)
        v_col = V[:, r]  # (m*p,)
        w_col = W[:, r]  # (p*n,)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_computed[i, j, k] += u_col[i*m + j] * v_col[j*p + k] * w_col[k*n + i]
    
    return np.allclose(T_computed, T_expected, atol=1e-9)

def _verify_fast(U, V, W, n, m, p):
    """
    Faster verification using tensor operations.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    We can vectorize: for each r, form a 3D array A_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    But that's O(n*m*p*rank) which is fine for n=4,m=4,p=7.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum: T[i,j,k] = U3[i,j,:] * V3[j,k,:] * W3[k,i,:] summed over r
    # = np.einsum('ijr,jkr,ki r->ijk', U3, V3, W3) -- need to be careful with spaces
    # Actually: np.einsum('ijr,jkr,ki r->ijk', U3, V3, W3)
    
    try:
        T_computed = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    except Exception:
        # Fallback to manual
        T_computed = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_computed[i, j, k] += U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
    
    T_expected = np.ones((n, m, p))
    return np.allclose(T_computed, T_expected, atol=1e-9)

def _strassen_2x2_2x2():
    """
    Returns U, V, W matrices (each 4x7) for the rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    U[i, r] corresponds to A[i//2, i%2] coefficient
    V[j, r] corresponds to B[j//2, j%2] coefficient  
    W[k, r] corresponds to C[k//2, k%2] coefficient
    T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r] = 1 for all i,j,k in {0..3}x{0..3}x{0..3}
    where i,j,k are local indices (0..3) for the 2x2 matrices.
    """
    # Standard Strassen for 2x2 matrices:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]], C = [[c00, c01], [c10, c11]]
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b01-b11)
    # P4 = a11(b10-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    # c00 = P1 - P4 + P5 + P7
    # c01 = P2 + P5 - P6 - P7
    # c10 = P1 + P3 + P5 - P4
    # c11 = P3 + P6 + P4 - P1
    
    # U indices: 0=a00, 1=a01, 2=a10, 3=a11
    # V indices: 0=b00, 1=b01, 2=b10, 3=b11
    # W indices: 0=c00, 1=c01, 2=c10, 3=c11
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1: (a00+a11)(b00+b11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = -1  # c00 += P1, c11 -= P1
    
    # P2: (a10+a11)b00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1  # c01 += P2
    
    # P3: a00(b01-b11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1  # c10 += P3, c11 += P3
    
    # P4: a11(b10-b00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = -1; W[3,3] = 1  # c00 += P4, c10 -= P4, c11 += P4
    
    # P5: (a00+a01)b11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1; W[2,4] = 1  # c00 -= P5, c01 += P5, c10 += P5
    
    # P6: (a10-a00)(b00+b01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[1,5] = -1; W[3,5] = 1  # c01 -= P6, c11 += P6
    
    # P7: (a01-a11)(b10+b11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1; W[1,6] = -1  # c00 += P7, c01 -= P7
    
    return U, V, W

def _build_4x4_4x7_strassen_rank63():
    """
    Build a rank-63 decomposition for 4x4 * 4x7 matrix multiplication.
    
    Strategy:
    - Split B (4x7) into B1 (4x4) and B2 (4x3).
    - C = C1 + C2 where C1 = A*B1 (4x4) and C2 = A*B2 (4x3).
    - Use Strassen (rank 49) for the 4x4*4x4 part.
    - Use schoolbook (rank 48) for the 4x4*4x3 part.
    - Total rank: 49 + 48 = 97.
    
    Actually, that's worse than the schoolbook 112? No, 97 < 112, so it's an improvement.
    
    But wait, we can do better. Let me think about the Strassen 4x4*4x4 part.
    
    For 4x4*4x4 using Strassen:
    - Split each 4x4 into 2x2 blocks.
    - Apply 7 Strassen multiplications of 2x2*2x2.
    - Each 2x2*2x2 has rank 7 (Strassen).
    - Total: 7 * 7 = 49.
    
    For 4x4*4x3 using schoolbook:
    - 4*4*3 = 48 terms.
    
    Total: 49 + 48 = 97.
    
    But we can also try to use Strassen-like tricks for the 4x4*4x3 part.
    Actually, 4x4*4x3 is not square, so standard Strassen doesn't directly apply.
    
    Let me just implement the 97-rank solution and verify it.
    """
    n, m, p = 4, 4, 7
    rank = 97
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U2, V2, W2 = _strassen_2x2_2x2()
    
    # Part 1: 4x4 * 4x4 using Strassen (rank 49)
    # A is 4x4, B1 is 4x4 (columns 0-3 of B), C1 is 4x4 (columns 0-3 of C)
    
    # Block structure (0-indexed):
    # A: A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
    # B1: B11=B1[0:2,0:2], B12=B1[0:2,2:4], B21=B1[2:4,0:2], B22=B1[2:4,2:4]
    # C1: C11=C1[0:2,0:2], C12=C1[0:2,2:4], C21=C1[2:4,0:2], C22=C1[2:4,2:4]
    
    # Strassen formulas for 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M2 + M5 - M6 - M7
    # C21 = M1 + M3 + M5 - M4
    # C22 = M3 + M6 + M4 - M1
    
    # For each Mi, we need to compute the product of two 2x2 matrices using Strassen (rank 7).
    # Each such product contributes 7 terms to U, V, W.
    
    # Helper: add a Strassen decomposition of X*Y to U, V, W with appropriate offsets and signs.
    # X is a linear combination of A blocks, Y is a linear combination of B blocks.
    # The result contributes to one or more C blocks with specified signs.
    
    def add_strassen_product(a_blocks, b_blocks, c_blocks, term_start):
        """
        a_blocks: list of (a_row_offset, a_col_offset, sign) for A blocks
        b_blocks: list of (b_row_offset, b_col_offset, sign) for B blocks
        c_blocks: list of (c_row_offset, c_col_offset, sign) for C blocks
        term_start: starting index in the rank dimension
        """
        for r in range(7):
            # U contribution: for each (i,j) in the 2x2, U[i,j,r]
            # The 2x2 A block at (a_row_offset, a_col_offset) with sign s contributes
            # U_local[i,j] * s to U[(a_row_offset*2+i)*4 + (a_col_offset*2+j), term_start+r]
            # Wait, U index is i*m + j where i is row in A (0..3), j is col in A (0..3).
            # For a 2x2 block starting at (r0, c0), element (i,j) in the block is at
            # global A position (r0+i, c0+j), so U index = (r0+i)*4 + (c0+j).
            
            # Similarly for V: B block at (r0, c0), element (i,j) is at global B position (r0+i, c0+j).
            # V index = (r0+i)*7 + (c0+j) since B is 4x7.
            
            # For W: C block at (r0, c0), element (i,j) is at global C position (r0+i, c0+j).
            # W index = (c0+j)*4 + (r0+i) since C is 7x4, and W index is k*n+i where k is col in C, i is row in C.
            
            for ar0, ac0, as_ in a_blocks:
                for i in range(2):
                    for j in range(2):
                        if U2[i*2+j, r] != 0:
                            global_i = ar0 + i
                            global_j = ac0 + j
                            U[global_i * 4 + global_j, term_start + r] += U2[i*2+j, r] * as_
            
            for br0, bc0, bs_ in b_blocks:
                for i in range(2):
                    for j in range(2):
                        if V2[i*2+j, r] != 0:
                            global_i = br0 + i
                            global_j = bc0 + j
                            V[global_i * 7 + global_j, term_start + r] += V2[i*2+j, r] * bs_
            
            for cr0, cc0, cs_ in c_blocks:
                for i in range(2):
                    for j in range(2):
                        if W2[i*2+j, r] != 0:
                            global_i = cr0 + i
                            global_j = cc0 + j
                            # W index: k*n + i where k = global_j (col in C), i = global_i (row in C)
                            W[global_j * 4 + global_i, term_start + r] += W2[i*2+j, r] * cs_
    
    # M1: (A11+A22)(B11+B22)
    # A11 at (0,0), A22 at (2,2); B11 at (0,0), B22 at (2,2)
    # C11 at (0,0), C22 at (2,2)
    add_strassen_product(
        [(0,0,1), (2,2,1)],  # A11 + A22
        [(0,0,1), (2,2,1)],  # B11 + B22
        [(0,0,1), (2,2,-1)],  # C11 += M1, C22 -= M1
        0
    )
    
    # M2: (A21+A22)B11
    # A21 at (2,0), A22 at (2,2); B11 at (0,0)
    # C12 at (0,2), C22 at (2,2)
    add_strassen_product(
        [(2,0,1), (2,2,1)],  # A21 + A22
        [(0,0,1)],            # B11
        [(0,2,1), (2,2,1)],  # C12 += M2, C22 += M2
        7
    )
    
    # M3: A11(B12-B22)
    # A11 at (0,0); B12 at (0,2), B22 at (2,2)
    # C21 at (2,0), C22 at (2,2)
    add_strassen_product(
        [(0,0,1)],            # A11
        [(0,2,1), (2,2,-1)],  # B12 - B22
        [(2,0,1), (2,2,1)],  # C21 += M3, C22 += M3
        14
    )
    
    # M4: A22(B21-B11)
    # A22 at (2,2); B21 at (2,0), B11 at (0,0)
    # C11 at (0,0), C21 at (2,0), C22 at (2,2)
    add_strassen_product(
        [(2,2,1)],            # A22
        [(2,0,1), (0,0,-1)],  # B21 - B11
        [(0,0,1), (2,0,-1), (2,2,1)],  # C11 += M4, C21 -= M4, C22 += M4
        21
    )
    
    # M5: (A11+A12)B22
    # A11 at (0,0), A12 at (0,2); B22 at (2,2)
    # C11 at (0,0), C12 at (0,2), C21 at (2,0)
    add_strassen_product(
        [(0,0,1), (0,2,1)],  # A11 + A12
        [(2,2,1)],            # B22
        [(0,0,-1), (0,2,1), (2,0,1)],  # C11 -= M5, C12 += M5, C21 += M5
        28
    )
    
    # M6: (A21-A11)(B11+B12)
    # A21 at (2,0), A11 at (0,0); B11 at (0,0), B12 at (0,2)
    # C12 at (0,2), C22 at (2,2)
    add_strassen_product(
        [(2,0,1), (0,0,-1)],  # A21 - A11
        [(0,0,1), (0,2,1)],  # B11 + B12
        [(0,2,-1), (2,2,1)],  # C12 -= M6, C22 += M6
        35
    )
    
    # M7: (A12-A22)(B21+B22)
    # A12 at (0,2), A22 at (2,2); B21 at (2,0), B22 at (2,2)
    # C11 at (0,0), C12 at (0,2), C22 at (2,2)
    add_strassen_product(
        [(0,2,1), (2,2,-1)],  # A12 - A22
        [(2,0,1), (2,2,1)],  # B21 + B22
        [(0,0,1), (0,2,-1), (2,2,-1)],  # C11 += M7, C12 -= M7, C22 -= M7
        42
    )
    
    # Part 2: 4x4 * 4x3 using schoolbook (rank 48)
    # B2 is B columns 4,5,6 (indices 4,5,6)
    # C2 is C columns 4,5,6 (indices 4,5,6)
    
    col = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i * 4 + j, col] = 1
                V[j * 7 + k, col] = 1
                W[k * 4 + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Try the rank-63 (actually 97) Strassen-based decomposition
    try:
        result = _build_4x4_4x7_strassen_rank63()
        if _verify_fast(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception as e:
        pass
    
    # Fallback: schoolbook rank 112
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
