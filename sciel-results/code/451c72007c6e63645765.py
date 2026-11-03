"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _time_left(deadline):
    return deadline - time.time()

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = delta for all (i,j,k)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Reshape: U is (n*m, rank) -> (n, m, rank)
    # V is (m*p, rank) -> (m, p, rank)
    # W is (p*n, rank) -> (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # For each (i,j,k), check sum_c U3[i,j,c]*V3[j,k,c]*W3[k,i,c] == 1
    # Build tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # We can compute this by: for each c, outer product, but more efficiently:
    # T[i,j,k] = sum_c (U3[i,j,c] * V3[j,k,c]) * W3[k,i,c]
    # Let's build it term by term or use einsum
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # This is a contraction over c
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    # T[i,j,k] should be 1 for all i,j,k
    return np.allclose(T, 1.0, atol=tol)

def _schoolbook(n, m, p):
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
    return U, V, W

def _strassen_2x2_block(n, m, p):
    """
    Build a rank-7 Strassen-like decomposition for a 2x2 * 2x2 multiplication block.
    Returns (U_block, V_block, W_block) where:
    U_block has shape (2*2, 7) = (4, 7)
    V_block has shape (2*2, 7) = (4, 7)
    W_block has shape (2*2, 7) = (4, 7)
    The Strassen formulas for 2x2 matrix multiplication:
    M1 = (A11 + A22)(B21 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B11 - B21)
    M4 = A22(B22 - B12)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U indices: 0=A11, 1=A12, 2=A21, 3=A22
    # V indices: 0=B11, 1=B12, 2=B21, 3=B22
    # W indices: 0=C11, 1=C12, 2=C21, 3=C22
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11 + A22)(B21 + B22)
    U[0][0] = 1; U[3][0] = 1
    V[2][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C11 += M1, C22 += M1
    
    # M2 = (A21 + A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # C21 += M2, C22 -= M2
    
    # M3 = A11(B11 - B21)
    U[0][2] = 1
    V[0][2] = 1; V[2][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C12 += M3, C22 += M3
    
    # M4 = A22(B22 - B12)
    U[3][3] = 1
    V[3][3] = 1; V[1][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C11 += M4, C21 += M4
    
    # M5 = (A11 + A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C11 -= M5, C12 += M5
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C22 += M6
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C11 += M7
    
    return U, V, W

def _build_strassen_embedding(n, m, p):
    """
    Embed 4 disjoint 2x2 Strassen blocks into the 3x3 * 3x8 problem.
    We partition:
    - A rows: 0,1 (block 1), 2 (single)
    - A cols: 0,1 (block 1), 2 (single)
    - B rows: 0,1 (block 1), 2 (single)
    - B cols: 0,1 (block 1), 2,3,4,5,6,7
    
    Actually, let's use 4 blocks of 2x2 * 2x2:
    Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    Block 2: A[0:2, 2:3] * B[0:2, 2:4] -> C[0:2, 2:4]  -- wait, this doesn't work cleanly
    
    Let me think more carefully. The standard approach for 3x3 * 3x8:
    We can use 2 Strassen 2x2 blocks covering rows 0,1 of A and cols 0,1 of B,
    then handle the remaining parts with schoolbook.
    
    Alternative: Use the known rank-68 construction.
    For 3x3 * 3x8, we can partition into:
    - 2 blocks of size 2x2 * 2x2 (rank 7 each) covering A[0:2,:], B[:,0:2]
    - Remaining: A[2,:] (1x3) * B[:,0:2] (3x2) -> rank 6 schoolbook
    - A[0:2,2] (2x1) * B[:,2:8] (3x6) -> rank 12 schoolbook
    - A[2,2] (1x1) * B[:,2:8] (3x6) -> rank 6 schoolbook
    Total: 7 + 7 + 6 + 12 + 6 = 38? That seems too low. Let me reconsider.
    
    Actually, the known result for 3x3 * 3x8 is rank 68.
    Let me use a simpler approach: 4 Strassen blocks.
    
    Partition A (3x3) into:
    - A11 = A[0:2, 0:2] (2x2)
    - A12 = A[0:2, 2] (2x1)
    - A21 = A[2, 0:2] (1x2)
    - A22 = A[2, 2] (1x1)
    
    Partition B (3x8) into:
    - B11 = B[0:2, 0:2] (2x2)
    - B12 = B[0:2, 2:8] (2x6)
    - B21 = B[2, 0:2] (1x2)
    - B22 = B[2, 2:8] (1x6)
    
    C = A*B (3x8):
    C[0:2, 0:2] = A11*B11 + A12*B21
    C[0:2, 2:8] = A11*B12 + A12*B22
    C[2, 0:2] = A21*B11 + A22*B21
    C[2, 2:8] = A21*B12 + A22*B22
    
    Use Strassen for A11*B11 (rank 7)
    Use Strassen for A11*B12? No, B12 is 2x6, not 2x2.
    
    Let me just use the verified rank-68 construction from the literature.
    For now, let's use 4 independent 2x2 Strassen blocks if possible.
    """
    # For 3x3 * 3x8, we can do:
    # 2 Strassen blocks for the 2x2 * 2x2 corner
    # and schoolbook for the rest
    # But let me try a different partition that allows 4 Strassen blocks.
    
    # Actually, let's just implement a clean solution:
    # Use 1 Strassen block for A[0:2,0:2] * B[0:2,0:2]
    # Schoolbook for everything else
    # Rank = 7 + (remaining schoolbook)
    # Remaining: 
    #   A[0:2,2] * B[:,2:8] : 2*1*6 = 12
    #   A[2,:] * B[:,0:2] : 1*3*2 = 6
    #   A[2,2] * B[:,2:8] : 1*1*6 = 6
    #   A[0:2,0:2] * B[:,2:8] : 2*2*6 = 24
    #   A[2,0:2] * B[0:2,0:2] : 1*2*2 = 4
    # Wait, I'm double counting. Let me be more careful.
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For i in 0,1 and k in 0,1: sum over j=0,1,2
    #   = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # The Strassen block handles sum over j=0,1 for A[i,j]*B[j,k] when i in 0,1 and k in 0,1
    # So we need to add A[i,2]*B[2,k] for i in 0,1, k in 0,1: rank 4
    # For i in 0,1 and k in 2..7: sum over j=0,1,2
    #   = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #   Schoolbook: 2*3*6 = 36
    # For i=2 and k in 0,1: sum over j=0,1,2
    #   = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    #   Schoolbook: 1*3*2 = 6
    # For i=2 and k in 2..7: sum over j=0,1,2
    #   = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    #   Schoolbook: 1*3*6 = 18
    # Total: 7 + 4 + 36 + 6 + 18 = 71. That's worse than 72? No, 71 < 72.
    # But we want 68. Let me try to use more Strassen.
    
    # Try: 2 Strassen blocks
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2], rank 7
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4], rank 7
    # Then schoolbook for the rest:
    #   C[0:2, 0:2] also needs A[0:2, 2] * B[2, 0:2]: 2*1*2 = 4
    #   C[0:2, 2:4] also needs A[0:2, 2] * B[2, 2:4]: 2*1*2 = 4
    #   C[0:2, 4:8] = A[0:2, :] * B[:, 4:8]: 2*3*4 = 24
    #   C[2, 0:2] = A[2, :] * B[:, 0:2]: 1*3*2 = 6
    #   C[2, 2:4] = A[2, :] * B[:, 2:4]: 1*3*2 = 6
    #   C[2, 4:8] = A[2, :] * B[:, 4:8]: 1*3*4 = 12
    # Total: 7 + 7 + 4 + 4 + 24 + 6 + 6 + 12 = 70
    # Still not 68.
    
    # Let me try 3 Strassen blocks:
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2], rank 7
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4], rank 7
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6], rank 7
    # Then schoolbook:
    #   C[0:2, 0:2] += A[0:2, 2] * B[2, 0:2]: 4
    #   C[0:2, 2:4] += A[0:2, 2] * B[2, 2:4]: 4
    #   C[0:2, 4:6] += A[0:2, 2] * B[2, 4:6]: 4
    #   C[0:2, 6:8] = A[0:2, :] * B[:, 6:8]: 2*3*2 = 12
    #   C[2, 0:2] = A[2, :] * B[:, 0:2]: 6
    #   C[2, 2:4] = A[2, :] * B[:, 2:4]: 6
    #   C[2, 4:6] = A[2, :] * B[:, 4:6]: 6
    #   C[2, 6:8] = A[2, :] * B[:, 6:8]: 12
    # Total: 7*3 + 4*3 + 12 + 6*3 + 12 = 21 + 12 + 12 + 18 + 12 = 75. Worse!
    
    # The issue is that splitting B columns into more blocks doesn't help because
    # we still need schoolbook for the A[2,:] and A[:,2] parts.
    
    # Let me just go with the rank-68 construction that's known to work.
    # I'll use a direct construction with 68 terms.
    
    # Actually, let me just use the 1-Strassen + schoolbook approach which gives 71,
    # and then try to optimize further.
    
    # For now, let's implement the 1-Strassen approach (rank 71) and verify it.
    n, m, p = 3, 3, 8
    
    # Strassen block for A[0:2, 0:2] * B[0:2, 0:2]
    U_s, V_s, W_s = _strassen_2x2_block(n, m, p)
    
    rank = 7  # Strassen
    # Add schoolbook terms
    # C[0:2, 0:2] += A[0:2, 2] * B[2, 0:2]: 2*1*2 = 4 terms
    # C[0:2, 2:8] = A[0:2, :] * B[:, 2:8]: 2*3*6 = 36 terms
    # C[2, 0:2] = A[2, :] * B[:, 0:2]: 1*3*2 = 6 terms
    # C[2, 2:8] = A[2, :] * B[:, 2:8]: 1*3*6 = 18 terms
    rank += 4 + 36 + 6 + 18
    # rank = 7 + 64 = 71
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Add Strassen block
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i * m + j][col] = U_s[i * 2 + j][c]
                V[j * p + (i if i < 2 else i)]  # wait, B indices
                # B[0:2, 0:2]: row j (0,1), col k (0,1)
                # V index: j * p + k where k is 0,1
                # But in the Strassen block, V index is for B[0:2, 0:2]
                # V_s has 4 rows: B11, B12, B21, B22 -> indices 0,1,2,3
                # These correspond to B[0,0], B[0,1], B[1,0], B[1,1]
                # In full V, these are at positions: 0*p+0, 0*p+1, 1*p+0, 1*p+1
                pass
    
    # This is getting complicated. Let me use a cleaner approach.
    # I'll directly build the rank-71 decomposition.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen block: A[0:2, 0:2] * B[0:2, 0:2]
    for c in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j][c] != 0:
                    U[i*m+j][col] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k][c] != 0:
                    V[j*p+k][col] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i][c] != 0:
                    W[k*n+i][col] = W_s[k*2+i][c]
        col += 1
    
    # Schoolbook: A[0:2, 2] * B[2, 0:2]
    for i in range(2):
        for k in range(2):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
    
    # Schoolbook: A[0:2, :] * B[:, 2:8]
    for i in range(2):
        for j in range(3):
            for k in range(2, 8):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # Schoolbook: A[2, :] * B[:, 0:2]
    for j in range(3):
        for k in range(2):
            U[2*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+2][col] = 1
            col += 1
    
    # Schoolbook: A[2, :] * B[:, 2:8]
    for j in range(3):
        for k in range(2, 8):
            U[2*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+2][col] = 1
            col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 120
    try:
        return _build_strassen_embedding(3, 3, 8)
    except Exception:
        n, m, p = DIMENSIONS
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
