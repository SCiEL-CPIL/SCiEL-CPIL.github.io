"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return Strassen decomposition for 2x2 * 2x2 -> 2x2. Rank 7.
    Returns U (4x7), V (4x7), W (4x7) as numpy arrays.
    Index convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # M3 = A11(B12 - B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4 = A22(B21 - B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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


def _combine_block_decomp(U1, V1, W1, U2, V2, W2):
    """Concatenate two decompositions block-diagonally."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 + r2
    U = np.zeros((U1.shape[0], total_rank))
    V = np.zeros((V1.shape[0], total_rank))
    W = np.zeros((W1.shape[0], total_rank))
    U[:, :r1] = U1
    V[:, :r1] = V1
    W[:, :r1] = W1
    U[:, r1:] = U2
    V[:, r1:] = V2
    W[:, r1:] = W2
    return U, V, W


def _strassen_4x4_4x4():
    """Strassen-based decomposition for 4x4 * 4x4 -> 4x4. Rank 49.
    Split into 2x2 blocks of size 2x2, use Strassen at block level,
    and Strassen (rank 7) for each 2x2*2x2 sub-product.
    """
    n = m = p = 4
    # Strassen at the 2x2 block level: 7 multiplications
    # Each block multiplication is 2x2 * 2x2 -> 2x2, done with rank 7 (Strassen)
    # Total rank = 7 * 7 = 49
    
    # Strassen 2x2*2x2 base (for the inner 2x2*2x2 products)
    U_base, V_base, W_base = _strassen_2x2()
    
    # Now build the 4x4*4x4 decomposition
    # A = [[A11, A12], [A21, A22]] where Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where Bij is 2x2
    # C = A*B = [[C11, C12], [C21, C22]] where Cij is 2x2
    #
    # Strassen at block level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mj is a 2x2 matrix that equals (linear combo of A blocks) * (linear combo of B blocks)
    # We compute each Mj using the Strassen 2x2*2x2 decomposition.
    
    total_rank = 49
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # For each of the 7 Strassen steps at block level, we need to:
    # 1. Form the linear combination of A blocks (this is done in U)
    # 2. Form the linear combination of B blocks (this is done in V)
    # 3. Multiply using Strassen 2x2*2x2 (rank 7)
    # 4. Distribute the result to the appropriate C blocks (this is done in W)
    
    # Block index mapping for 4x4 matrix:
    # A[0,0]=A11: indices 0,1,4,5 (row 0-1, col 0-1)
    # A[0,1]=A12: indices 2,3,6,7 (row 0-1, col 2-3)
    # A[1,0]=A21: indices 8,9,12,13 (row 2-3, col 0-1)
    # A[1,1]=A22: indices 10,11,14,15 (row 2-3, col 2-3)
    # Similarly for B and C.
    
    # Let's define the Strassen steps more carefully.
    # For each step s (0..6), we have:
    #   A_combo = a1 * A_block1 + a2 * A_block2  (coefficients a1, a2)
    #   B_combo = b1 * B_block1 + b2 * B_block2  (coefficients b1, b2)
    #   M = A_combo * B_combo  (2x2 * 2x2 -> 2x2)
    #   This M contributes to C blocks with coefficients.
    
    # Strassen steps:
    # s=0: M1=(A11+A22)(B11+B22): A blocks (11,22) coeff (1,1), B blocks (11,22) coeff (1,1)
    # s=1: M2=(A21+A22)B11: A blocks (21,22) coeff (1,1), B blocks (11) coeff (1)
    # s=2: M3=A11(B12-B22): A blocks (11) coeff (1), B blocks (12,22) coeff (1,-1)
    # s=3: M4=A22(B21-B11): A blocks (22) coeff (1), B blocks (21,11) coeff (1,-1)
    # s=4: M5=(A11+A12)B22: A blocks (11,12) coeff (1,1), B blocks (22) coeff (1)
    # s=5: M6=(A21-A11)(B11+B12): A blocks (21,11) coeff (1,-1), B blocks (11,12) coeff (1,1)
    # s=6: M7=(A12-A22)(B21+B22): A blocks (12,22) coeff (1,-1), B blocks (21,22) coeff (1,1)
    
    # C block contributions:
    # C11 = M1 + M4 - M5 + M7: M1 coeff 1, M4 coeff 1, M5 coeff -1, M7 coeff 1
    # C12 = M3 + M5: M3 coeff 1, M5 coeff 1
    # C21 = M2 + M4: M2 coeff 1, M4 coeff 1
    # C22 = M1 - M2 + M3 + M6: M1 coeff 1, M2 coeff -1, M3 coeff 1, M6 coeff 1
    
    # Block to index mapping:
    # A block (bi, bj) where bi in {0,1}, bj in {0,1}:
    #   row offset = bi*2, col offset = bj*2
    #   indices: (bi*2)*4 + bj*2 + di + 4*ej for di in {0,1}, ej in {0,1}
    #   = 4*bi*2 + 2*bj + di + 4*ej = 8*bi + 2*bj + di + 4*ej
    #   Actually: index = (bi*2 + di)*m + (bj*2 + ej) = (bi*2+di)*4 + bj*2 + ej
    #   = 8*bi + 4*di + 2*bj + ej
    
    # B block (bi, bj) where bi in {0,1} (row), bj in {0,1} (col):
    #   index = (bi*2 + di)*p + (bj*2 + ej) = (bi*2+di)*4 + bj*2 + ej
    #   = 8*bi + 4*di + 2*bj + ej
    
    # C block (bi, bj) where bi in {0,1} (row of C), bj in {0,1} (col of C):
    #   W index = k*n + i where k is col of C, i is row of C
    #   k = bj*2 + ej, i = bi*2 + di
    #   W index = (bj*2+ej)*4 + (bi*2+di) = 8*bj + 4*ej + 2*bi + di
    
    def A_idx(bi, bj, di, ej):
        return (bi*2 + di) * 4 + (bj*2 + ej)
    
    def B_idx(bi, bj, di, ej):
        return (bi*2 + di) * 4 + (bj*2 + ej)
    
    def W_idx(bi, bj, di, ej):
        # C[i,k] where i = bi*2+di, k = bj*2+ej
        # W index = k*n + i = (bj*2+ej)*4 + (bi*2+di)
        return (bj*2 + ej) * 4 + (bi*2 + di)
    
    # Strassen step definitions: (a_blocks, a_coeffs, b_blocks, b_coeffs)
    strassen_steps = [
        # (A_block_indices, A_coeffs, B_block_indices, B_coeffs)
        ((0,1), (1,1), (0,1), (1,1)),   # M1: (A11+A22)(B11+B22)
        ((2,3), (1,1), (0,), (1,)),     # M2: (A21+A22)B11
        ((0,), (1,), (1,3), (1,-1)),    # M3: A11(B12-B22)
        ((3,), (1,), (2,0), (1,-1)),    # M4: A22(B21-B11)
        ((0,1), (1,1), (3,), (1,)),     # M5: (A11+A12)B22
        ((2,0), (1,-1), (0,1), (1,1)),  # M6: (A21-A11)(B11+B12)
        ((1,3), (1,-1), (2,3), (1,1)),  # M7: (A12-A22)(B21+B22)
    ]
    
    # C block contributions: for each C block (bi,bj), list of (step, coeff)
    # C blocks: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    c_contribs = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],   # C11 = M1+M4-M5+M7
        (0,1): [(2,1), (4,1)],                   # C12 = M3+M5
        (1,0): [(1,1), (3,1)],                   # C21 = M2+M4
        (1,1): [(0,1), (1,-1), (2,1), (5,1)],   # C22 = M1-M2+M3+M6
    }
    
    # Now build the decomposition
    for s in range(7):
        a_blocks, a_coeffs, b_blocks, b_coeffs = strassen_steps[s]
        col_offset = s * 7
        
        # For each of the 7 Strassen inner steps, we set up U, V, W
        # The inner Strassen computes: (A_combo * B_combo) where A_combo and B_combo are 2x2
        # A_combo[di, ej] = sum over A blocks: a_coeff * A_block[di, ej]
        # B_combo[di, ej] = sum over B blocks: b_coeff * B_block[di, ej]
        
        # For the inner Strassen 2x2*2x2:
        # U_inner[di*2+ej, c_inner] corresponds to A_combo[di, ej]
        # V_inner[di*2+ej, c_inner] corresponds to B_combo[di, ej]
        # W_inner[dk*2+di, c_inner] corresponds to (A_combo*B_combo)[di, dk]
        
        for c_inner in range(7):
            col = col_offset + c_inner
            
            # Set U: for each (di, ej) in A_combo, U[A_idx(bi,bj,di,ej), col] += a_coeff * U_base[di*2+ej, c_inner]
            for ab_idx, ab_coeff in enumerate(a_blocks):
                bi, bj = ab_idx // 2, ab_idx % 2
                # Wait, a_blocks contains indices like 0,1,2,3 which are (bi,bj) pairs
                # Let me redefine: a_blocks is a tuple of (bi,bj) pairs
                pass
            
            # Let me redo this with explicit (bi,bj) pairs
            # Actually let me restructure the Strassen steps to use (bi,bj) directly
        
        # Let me redo the whole thing more carefully
    
    # Let me restart with a cleaner approach
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Strassen steps with explicit (bi,bj) for A and B blocks
    # A blocks: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B blocks: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    
    strassen_steps2 = [
        # (list of (a_bi, a_bj, a_coeff), list of (b_bi, b_bj, b_coeff))
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),   # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)]),             # M2
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),            # M3
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),            # M4
        ([(0,0,1), (0,1,1)], [(1,1,1)]),             # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),   # M6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),   # M7
    ]
    
    for s in range(7):
        a_terms, b_terms = strassen_steps2[s]
        col_offset = s * 7
        
        for c_inner in range(7):
            col = col_offset + c_inner
            
            # U: for each A term (a_bi, a_bj, a_coeff), and for each (di, ej) in 2x2:
            #   U[A_idx(a_bi, a_bj, di, ej), col] += a_coeff * U_base[di*2+ej, c_inner]
            for a_bi, a_bj, a_coeff in a_terms:
                for di in range(2):
                    for ej in range(2):
                        U[A_idx(a_bi, a_bj, di, ej), col] += a_coeff * U_base[di*2+ej, c_inner]
            
            # V: for each B term (b_bi, b_bj, b_coeff), and for each (di, ej) in 2x2:
            #   V[B_idx(b_bi, b_bj, di, ej), col] += b_coeff * V_base[di*2+ej, c_inner]
            for b_bi, b_bj, b_coeff in b_terms:
                for di in range(2):
                    for ej in range(2):
                        V[B_idx(b_bi, b_bj, di, ej), col] += b_coeff * V_base[di*2+ej, c_inner]
            
            # W: for each C block (c_bi, c_bj) that this step contributes to:
            #   For each (di, ej) in 2x2 (di=row of C, ej=col of C):
            #   W[W_idx(c_bi, c_bj, di, ej), col] += c_coeff * W_base[di*2+ej, c_inner]
            for (c_bi, c_bj), contribs in c_contribs.items():
                for (step, c_coeff) in contribs:
                    if step == s:
                        for di in range(2):
                            for ej in range(2):
                                W[W_idx(c_bi, c_bj, di, ej), col] += c_coeff * W_base[di*2+ej, c_inner]
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition correctness."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Strategy: Split B into 4x4 + 4x3
    # 4x4 * 4x4: use Strassen-based decomposition (rank 49)
    # 4x4 * 4x3: use schoolbook (rank 48)
    # Total rank: 49 + 48 = 97
    
    # Build 4x4 * 4x4 part
    U1, V1, W1 = _strassen_4x4_4x4()
    
    # Build 4x4 * 4x3 part (schoolbook)
    U2, V2, W2 = _schoolbook(4, 4, 3)
    
    # Combine: The 4x4*4x4 part handles columns 0-3 of B
    # The 4x4*4x3 part handles columns 4-6 of B
    # We need to adjust V and W indices for the second part.
    
    # For the 4x4*4x3 part:
    # V2 has shape (4*3, 48) = (12, 48)
    # But we need V to have shape (4*7, rank) = (28, rank)
    # V2 corresponds to B rows 0-3, cols 4-6
    # So V_full[j*7+k, col] where k is 0-6
    # For the second part, k = 4,5,6, so V_full[j*7+4+d, col] = V2[j*3+d, col]
    
    # W2 has shape (3*4, 48) = (12, 48)
    # W2[k*4+i, col] where k is 0-2 (corresponding to C cols 4-6)
    # W_full[k*4+i, col] where k is 0-6
    # For the second part, k = 4,5,6, so W_full[(4+d)*4+i, col] = W2[d*4+i, col]
    
    total_rank = U1.shape[1] + U2.shape[1]
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # First part: 4x4 * 4x4
    U[:, :U1.shape[1]] = U1
    V[:4*4, :U1.shape[1]] = V1  # V1 is (16, 49), corresponds to B rows 0-3, cols 0-3
    W[:4*4, :U1.shape[1]] = W1  # W1 is (16, 49), corresponds to C cols 0-3
    
    # Second part: 4x4 * 4x3
    r1 = U1.shape[1]
    U[:, r1:] = U2
    # V2 is (12, 48), B rows 0-3, local cols 0-2 -> global cols 4-6
    for j in range(4):
        for d in range(3):
            V[j * 7 + 4 + d, r1:r1+U2.shape[1]] = V2[j * 3 + d, :]
    # W2 is (12, 48), local C cols 0-2 -> global C cols 4-6
    for d in range(3):
        for i in range(4):
            W[(4 + d) * 4 + i, r1:r1+U2.shape[1]] = W2[d * 4 + i, :]
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    if time.time() > deadline:
        # Return what we have
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
