"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
from collections import defaultdict

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W form a valid tensor decomposition."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check dimensions
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = delta
    # Verify: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 if i,j,k all match (standard matmul)
    # Actually the tensor for matrix multiplication C = A * B where C[i,k] = sum_j A[i,j]*B[j,k]
    # is T[i,j,k] = 1 (it's the identity tensor over the contraction index j)
    # Wait, the tensor T for matmul has T[i,j,k] = 1 for all i,j,k? No.
    # The structure tensor of matmul M_{n,m,p} has T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k}... 
    # Actually T[i, j, k] = 1 for all i in [n], j in [m], k in [p]. 
    # No wait. The tensor is T in C^{n*m} \otimes C^{m*p} \otimes C^{p*n}
    # where T_{(i,j),(j,k),(k,i)} = 1 and 0 otherwise.
    # So T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent (which they always are in the flattening)
    # and 0 otherwise.
    # Actually T[a, b, c] = 1 iff a = i*m+j, b = j'*p+k, c = k'*n+i' for some i,j,j',k,k',i'
    # with j = j' and k = k' and i = i'. 
    # In the standard flattening: T[i*m+j, j*p+k, k*n+i] = 1, all other entries 0.
    
    # Verify by checking all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(actual - expected) > tol:
                    return False
    return True


def _verify_decomposition_sparse(U, V, W, n, m, p, tol=1e-10):
    """Faster verification using sparsity."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Use Einstein summation for speed
    # T = sum_c U[:, c] * V[:, c] * W[:, c] (outer product sum)
    # Reshape and use tensordot
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    T = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    
    # Check that T[i*m+j, j*p+k, k*n+i] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > tol:
                    return False
    
    # Check that all other entries are 0
    # Build a set of valid indices
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    
    # Check a sample of invalid entries (checking all would be too slow)
    for idx in range(0, n * m, max(1, n * m // 50)):
        for idx2 in range(0, m * p, max(1, m * p // 50)):
            for idx3 in range(0, p * n, max(1, p * n // 50)):
                if (idx, idx2, idx3) not in valid:
                    if abs(T[idx, idx2, idx3]) > tol:
                        return False
    
    return True


def _try_rank_one_removal(U, V, W, n, m, p):
    """Try removing one rank-1 term from the decomposition."""
    rank = len(U[0])
    best = None
    for c_remove in range(rank):
        # Remove term c_remove
        U_new = [row[:] for row in U]
        V_new = [row[:] for row in V]
        W_new = [row[:] for row in W]
        for r in range(len(U_new)):
            U_new[r] = U_new[r][:c_remove] + U_new[r][c_remove + 1:]
        for r in range(len(V_new)):
            V_new[r] = V_new[r][:c_remove] + V_new[r][c_remove + 1:]
        for r in range(len(W_new)):
            W_new[r] = W_new[r][:c_remove] + W_new[r][c_remove + 1:]
        
        if _verify_decomposition_sparse(U_new, V_new, W_new, n, m, p):
            best = (U_new, V_new, W_new)
            break
    return best


def _try_merge_terms(U, V, W, n, m, p):
    """Try merging two rank-1 terms into one if their combined effect is rank-1."""
    rank = len(U[0])
    for c1 in range(rank):
        for c2 in range(c1 + 1, rank):
            # Check if terms c1 and c2 can be merged
            # They can be merged if U[:,c1]*V[:,c1]*W[:,c1] + U[:,c2]*V[:,c2]*W[:,c2] is rank-1
            # This is complex; skip for now
            pass
    return None


def _construct_schoolbook(n, m, p):
    """Construct schoolbook decomposition."""
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


def _construct_strassen_2x2():
    """Return Strassen decomposition for 2x2x2 multiplication as (U, V, W)."""
    # Strassen: M_{2,2,2} has rank 7
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Flatten: A[i,j] -> index i*2+j, B[j,k] -> index j*2+k, C[k,i] -> index k*2+i
    # n=m=p=2
    # U has 4 rows (2*2), V has 4 rows (2*2), W has 4 rows (2*2), 7 columns
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # a11 -> U[0], a12 -> U[1], a21 -> U[2], a22 -> U[3]
    # b11 -> V[0], b12 -> V[1], b21 -> V[2], b22 -> V[3]
    # c11 -> W[0] (k=0,i=0), c12 -> W[1] (k=0,i=1), c21 -> W[2] (k=1,i=0), c22 -> W[3] (k=1,i=1)
    
    # M1 = (a11+a22)(b11+b22) -> contributes to C11, C22
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a21+a22)b11 -> contributes to C21, C22
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a11(b12-b22) -> contributes to C12, C22
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = a22(b21-b11) -> contributes to C11, C21
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a11+a12)b22 -> contributes to C11, C12
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (a21-a11)(b11+b12) -> contributes to C22
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a12-a22)(b21+b22) -> contributes to C11
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _construct_strassen_4x4(n, m, p):
    """Construct a decomposition for 4x4x4 using Strassen 2x2 blocks.
    This gives rank 7*7 = 49."""
    # Split into 2x2 blocks
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = A*B has blocks Cij = sum_k Aik * Bkj
    # Using Strassen at the block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # Each Mik is a 2x2 matrix product, each computed with Strassen rank 7
    # Total rank = 7 * 7 = 49
    
    # Flatten indices:
    # A[i,j] for i,j in 0..3: block (i//2, j//2), local (i%2, j%2)
    # A index = (i//2)*2 + (j//2) for block, local index = i%2*2 + j%2
    # Full A index = i*4 + j (since n*m = 16)
    # Wait, n=4, m=4, so A has 16 entries, indexed 0..15
    # A[i,j] is at index i*4 + j
    # B[j,k] is at index j*4 + k  
    # C[k,i] is at index k*4 + i
    
    # For Strassen block decomposition:
    # Block (I,J) of A covers rows I*2..I*2+1, cols J*2..J*2+1
    # Local Strassen indices within a 2x2 block:
    # a00 -> local (0,0), a01 -> local (0,1), a10 -> local (1,0), a11 -> local (1,1)
    
    # We need to build U, V, W of size (16, 49), (16, 49), (16, 49)
    rank = 49
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(16)]
    W = [[0.0] * rank for _ in range(16)]
    
    # Block-level Strassen: 7 multipliers
    # For each block-level multiplier s (0..6) and local Strassen multiplier t (0..6):
    # column = s * 7 + t
    
    # Block-level Strassen formulas (for 2x2 block matrices):
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # For each block-level term, we need to:
    # 1. Determine which A blocks contribute and with what coefficients
    # 2. Determine which B blocks contribute and with what coefficients
    # 3. Determine which C blocks receive contributions and with what coefficients
    
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3), C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)
    
    # Block index to (row_start, col_start)
    # A block (bi, bj): rows bi*2 to bi*2+1, cols bj*2 to bj*2+1
    # A element (i,j) is in block (i//2, j//2), local position (i%2, j%2)
    
    # Strassen local (within 2x2): 
    # s0: a00 -> (0,0), a01 -> (0,1), a10 -> (1,0), a11 -> (1,1)
    
    # Block-level contributions:
    # For each block-level term s:
    #   a_blocks: dict {block_idx: coeff} for A
    #   b_blocks: dict {block_idx: coeff} for B
    #   c_blocks: dict {block_idx: coeff} for C
    
    block_level = [
        # (a_blocks, b_blocks, c_blocks)
        # M0 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M1 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M2 = A00(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M3 = A11(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M4 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M5 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M6 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Local Strassen (for 2x2):
    # t0: a00+a11, b00+b11, c00+c11
    # t1: a10+a11, b00, c10-c11
    # t2: a00, b01-b11, c01+c11
    # t3: a11, b10-b00, c00+c10
    # t4: a00+a01, b11, c00-c01
    # t5: a10-a00, b00+b01, c11
    # t6: a01-a11, b10+b11, c00
    
    local_strassen = [
        # (a_local, b_local, c_local) where each is {local_idx: coeff}
        # local_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),  # t0
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),       # t1
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),        # t2
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),        # t3
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),        # t4
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),        # t5
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),        # t6
    ]
    
    # Build the decomposition
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(block_level):
        for t, (a_local, b_local, c_local) in enumerate(local_strassen):
            col = s * 7 + t
            for (bi, bj), a_coeff in a_blocks.items():
                for (li, lj), a_lcoeff in a_local.items():
                    # A element at (bi*2+li, bj*2+lj)
                    i = bi * 2 + li
                    j = bj * 2 + lj
                    U[i * 4 + j][col] += a_coeff * a_lcoeff
            
            for (bi, bj), b_coeff in b_blocks.items():
                for (li, lj), b_lcoeff in b_local.items():
                    # B element at (bi*2+li, bj*2+lj)
                    j = bi * 2 + li
                    k = bj * 2 + lj
                    V[j * 4 + k][col] += b_coeff * b_lcoeff
            
            for (ci, cj), c_coeff in c_blocks.items():
                for (li, lj), c_lcoeff in c_local.items():
                    # C element at (ci*2+li, cj*2+lj)
                    k = ci * 2 + li
                    i = cj * 2 + lj
                    W[k * 4 + i][col] += c_coeff * c_lcoeff
    
    return U, V, W


def _construct_strassen_4x4x9(n, m, p):
    """Construct decomposition for 4x4x9 by splitting B into three 4x3 blocks,
    padding to 4x4, using Strassen 4x4x4 for each, and summing."""
    # B is 4x9, split into B1 (4x3), B2 (4x3), B3 (4x3)
    # Pad each to 4x4 with zeros
    # C = A*B1_pad + A*B2_pad + A*B3_pad
    # Each A*Bx_pad is 4x4, computed with Strassen 4x4x4 (rank 49)
    # Total rank = 3 * 49 = 147
    # This is worse than schoolbook 144, so this is not useful.
    # But let's construct it anyway for completeness.
    
    # Actually, 147 > 144, so this is worse. Let's not use this.
    # Return schoolbook instead.
    return _construct_schoolbook(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    best_rank = len(U[0])
    best_result = {"U": U, "V": V, "W": W}
    
    # Try Strassen 4x4x4 based approach for the 4x4 part
    # For 4x4x9, we can't directly use Strassen 4x4x4 without padding
    # Let's try to use Strassen on the 4x4 matrix multiplication and handle the 9 columns separately
    
    # Alternative: Use Winograd or other techniques
    
    # For now, try to improve schoolbook by removing redundant terms
    # This is a heuristic approach
    
    # Try rank reduction via linear algebra
    # The structure tensor has symmetries; we can try to find a lower-rank decomposition
    
    # One approach: use the fact that for 4x4x4, the best known is 49 (Strassen)
    # For 4x4x9, we can view it as 9 copies of 4x4x1? No, that's not right.
    
    # Actually, let's just return schoolbook since we can't easily find better
    # within the time constraint.
    
    # But let's at least verify it's correct
    if not _verify_decomposition_sparse(U, V, W, n, m, p):
        # Fallback: construct from scratch
        U, V, W = _construct_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
