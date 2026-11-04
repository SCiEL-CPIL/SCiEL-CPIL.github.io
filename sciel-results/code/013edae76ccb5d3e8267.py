"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [4, 4, 6]

import time
import sys

# EVOLVE-BLOCK-START
def _to_list(mat):
    """Convert a list of lists / numpy array to a clean list of lists of ints (or floats)."""
    if hasattr(mat, 'tolist'):
        mat = mat.tolist()
    out = []
    for row in mat:
        out_row = []
        for v in row:
            if hasattr(v, 'item'):
                v = v.item()
            out_row.append(v)
        out.append(out_row)
    return out

def _verify(U, V, W, n, m, p):
    """Check tensor decomposition. Returns True if valid."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def _build_strassen_2x2():
    """Strassen 2x2*2x2 factors. Returns U(4,7), V(4,7), W(4,7)."""
    import numpy as np
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # m1 = (a00+a11)(b00+b11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # m2 = (a10+a11)b00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1
    # m3 = a00(b01-b11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1
    # m4 = a11(b10-b00)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1
    # m5 = (a00+a01)b11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[3,4]=1
    # m6 = (a10-a00)(b00+b01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[0,5]=1
    # m7 = (a01-a11)(b10+b11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[2,6]=1
    return U, V, W

def _build_kronecker_decomposition():
    """
    Build a structured decomposition for 4x4 * 4x6 using Kronecker product structure.
    
    Key insight: 4x4*4x6 can be seen as a 2x2 block matrix multiplication where:
    - A is 2x2 blocks of 2x2 (total 4x4)
    - B is 2x3 blocks of 2x2 (total 4x6)
    - C is 2x3 blocks of 2x2 (total 4x6)
    
    We use Strassen (rank 7) for the 2x2 block-level multiplication combined with
    the internal 2x2*2x2 structure.
    
    Alternative approach: Use the fact that 4x4*4x6 = (2x2 ⊗ 2x2) * (2x2 ⊗ 2x3)
    The Kronecker structure allows us to share computation across blocks.
    
    Strategy: 
    1. First, use Strassen at the block level (2x2 block matrices) to get rank 7 for block mult.
    2. Each block multiplication is 2x2 * 2x2 = 2x2, which itself has rank 4 (schoolbook) or 7 (Strassen).
    3. Total: 7 (block Strassen) * 4 (inner schoolbook) = 28? No, that's not right.
    
    Actually, let me think differently. The correct approach:
    
    For 4x4 * 4x6:
    - Split into 2x2 blocks: A has 2x2 block structure (4x4), B has 2x3 block structure (4x6)
    - Block multiplication: C_blocks = A_blocks * B_blocks (2x2 * 2x3 = 2x3)
    - Each block product: (2x2) * (2x2) = (2x2)
    
    The tensor T[i,j,k] for i in 0..3, j in 0..3, k in 0..5.
    Write i = 2*ia + ii, j = 2*ib + jj, k = 2*ic + kk where ia,ib in {0,1}, ic in {0,1,2}.
    
    T[2*ia+ii, 2*ib+jj, 2*ic+kk] = 1 for all valid indices.
    
    Factorization: sum_r U[2*ia+ii, 2*ib+jj, r] * V[2*ib+jj, 2*ic+kk, r] * W[2*ic+kk, 2*ia+ii, r] = 1
    
    Using Kronecker: Let U = U_block ⊗ U_inner where U_block is (2*2, R_block) and U_inner is (2*2, R_inner).
    But this only works if the tensor itself is a Kronecker product, which it's not exactly
    because of the rectangular shape.
    
    Better approach: Use the structured decomposition where we treat the problem as:
    - 2x2 block Strassen gives 7 "block products"
    - Each block product needs to compute 2x2 * 2x2 = 2x2
    - For each of the 7 Strassen terms, we need to handle the internal 2x2*2x2
    
    Total rank = 7 * (rank of 2x2*2x2) = 7 * 4 = 28 (using schoolbook inner)
    Or 7 * 7 = 49 (using Strassen inner)
    
    Wait, that's not right either. Let me be more careful.
    
    Actually the correct structured approach:
    The full tensor is NOT a Kronecker product. But we can use a "semi-structured" approach:
    
    For each of the 7 Strassen block-level products, we have a contribution that involves
    specific linear combinations of A-blocks times specific linear combinations of B-blocks,
    contributing to specific linear combinations of C-blocks.
    
    Within each such term, the internal 2x2*2x2 multiplication has rank 4 (schoolbook).
    So total rank = 7 * 4 = 28.
    
    But wait - the Strassen block-level factors involve the BLOCK indices (ia, ib, ic),
    while the internal factors involve the WITHIN-BLOCK indices (ii, jj, kk).
    These are independent, so we can use a Kronecker-like product:
    
    U[2*ia+ii, 2*ib+jj, r_block*R_inner + r_inner] = U_block[ia*2+ib, r_block] * U_inner[ii*2+jj, r_inner]
    V[2*ib+jj, 2*ic+kk, r_block*R_inner + r_inner] = V_block[ib*3+ic, r_block] * V_inner[jj*2+kk, r_inner]  
    W[2*ic+kk, 2*ia+ii, r_block*R_inner + r_inner] = W_block[ic*2+ia, r_block] * W_inner[kk*2+ii, r_inner]
    
    Hmm, but V_block has shape (2*3, 7) = (6, 7) and W_block has shape (3*2, 7) = (6, 7).
    And U_block has shape (2*2, 7) = (4, 7).
    
    U_inner: (4, 4) for 2x2*2x2 schoolbook
    V_inner: (4, 4) for 2x2*2x2 schoolbook  
    W_inner: (4, 4) for 2x2*2x2 schoolbook
    
    Total rank = 7 * 4 = 28.
    
    Let me verify this construction works.
    """
    import numpy as np
    n, m, p = 4, 4, 6
    
    # Block-level Strassen: 2x2 block matrices
    # A_blocks: 2x2 (each block is 2x2)
    # B_blocks: 2x3 (each block is 2x2)
    # C_blocks: 2x3 (each block is 2x2)
    # Block multiplication: C_blocks = A_blocks * B_blocks
    # This is (2x2) * (2x3) = (2x3)
    # We need a decomposition for 2x2 * 2x3.
    
    # For 2x2 * 2x3: schoolbook rank = 2*2*3 = 12
    # Can we do better? 2x2*2x3 = 2x2*(2x2 + 2x1) ... not obviously helpful.
    # Actually, 2x2 * 2x3: treat as two 2x2*2x1 multiplications plus... no.
    # Let's just use schoolbook for the block level: rank 12.
    # Total: 12 * 4 = 48.
    
    # Alternatively, use Strassen-like for 2x2*2x3:
    # 2x2 * 2x3: We can split B into two 2x2 and one 2x1, or...
    # Actually, let's try: 2x2 * 2x3 can be done by computing:
    # C[:, 0:2] = A * B[:, 0:2]  (2x2 * 2x2, rank 7 with Strassen)
    # C[:, 2:3] = A * B[:, 2:3]  (2x2 * 2x1, rank 2)
    # Total: 7 + 2 = 9 for block level.
    # But these share the A factor, so we can't just add ranks.
    # Actually in tensor decomposition, if the (i,j) indices for A are shared,
    # the factors DO overlap, and we need to be more careful.
    
    # Let me just use the straightforward block approach with schoolbook inner:
    # Block level: 2x2 * 2x3 schoolbook = rank 12
    # Inner level: 2x2 * 2x2 schoolbook = rank 4
    # Total: 12 * 4 = 48
    
    # Or use Strassen for 2x2*2x2 at block level where possible:
    # Split 2x2*2x3 into: 2x2*2x2 (first 2 cols) + 2x2*2x1 (last col)
    # 2x2*2x2 with Strassen: rank 7
    # 2x2*2x1: C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0], rank 2
    # Total block rank: 7 + 2 = 9
    # Inner: 2x2*2x2 schoolbook rank 4 for the first part, 2x2*2x1 rank 2 for second
    # Total: 7*4 + 2*2 = 28 + 4 = 32
    
    # Hmm, let me just go with the clean structured approach:
    # Use schoolbook at both levels: 12 * 4 = 48
    # Then try to reduce.
    
    # Actually, let me try a different decomposition:
    # 4x4 * 4x6:
    # Split A into rows: top 2 rows (2x4) and bottom 2 rows (2x4)
    # Split B into cols: left 4 cols (4x4) and right 2 cols (4x2)
    # C_top_left = A_top * B_left (2x4 * 4x4 = 2x4)
    # C_top_right = A_top * B_right (2x4 * 4x2 = 2x2)
    # C_bot_left = A_bot * B_left (2x4 * 4x4 = 2x4)
    # C_bot_right = A_bot * B_right (2x4 * 4x2 = 2x2)
    
    # 2x4 * 4x4: rank 2*4*4 = 32 schoolbook. Can we do better?
    # 2x4 * 4x4: treat as 2x(4x4) * (4x4)... 
    # Actually 2x4 * 4x4 = (2x2 ⊗ I_2) * ... no, not clean.
    # 2x4 * 4x4: split 4 into 2+2: 
    #   = 2x4 * (4x2 + 4x2) ... no, that's not how it works.
    #   = (2x2 + 2x2) * (4x4) ... no.
    #   Think of it as: C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    #   Split j: C[i,k] = sum_{j=0,1} A[i,j]B[j,k] + sum_{j=2,3} A[i,j]B[j,k]
    #   = A_left * B_top + A_right * B_bot where A_left is 2x2, B_top is 2x4, etc.
    #   This gives 2 sub-problems of 2x2 * 2x4.
    #   2x2 * 2x4: rank 2*2*4 = 16 schoolbook.
    #   Total: 2 * 16 = 32. Same as before.
    
    # I think the cleanest approach that gives a definite improvement is:
    # 2x2 block Strassen (rank 7) for the block level, with 2x2*2x2 schoolbook (rank 4) inside.
    # But we need to handle the rectangular 2x3 B block structure.
    
    # Let me just construct the 48-rank solution and then attempt rank reduction.
    
    # Build 4x4*4x6 with 2x2 blocks, schoolbook everywhere:
    # 12 block positions * 4 inner positions = 48
    U = np.zeros((16, 48))
    V = np.zeros((24, 48))
    W = np.zeros((24, 48))
    
    col = 0
    for ia in range(2):  # row block of A
        for ib in range(2):  # col block of A = row block of B
            for ic in range(3):  # col block of B
                for ii in range(2):  # within block row
                    for jj in range(2):  # within block col
                        i = 2*ia + ii
                        j = 2*ib + jj
                        U[i*4 + j, col] = 1
                        for kk in range(2):
                            k = 2*ic + kk
                            V[j*6 + k, col] = 1
                            W[k*4 + i, col] = 1
                        col += 1
    # Wait, that's not right. Let me redo.
    # For each (ia, ib, ic) block triple and each (ii, jj, kk) inner triple:
    # U[2*ia+ii, 2*ib+jj] = 1
    # V[2*ib+jj, 2*ic+kk] = 1
    # W[2*ic+kk, 2*ia+ii] = 1
    
    U = np.zeros((16, 48))
    V = np.zeros((24, 48))
    W = np.zeros((24, 48))
    col = 0
    for ia in range(2):
        for ib in range(2):
            for ic in range(3):
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            i = 2*ia + ii
                            j = 2*ib + jj
                            k = 2*ic + kk
                            U[i*4 + j, col] = 1
                            V[j*6 + k, col] = 1
                            W[k*4 + i, col] = 1
                            col += 1
    # col should be 2*2*3*2*2*2 = 96... wait no.
    # 2*2*3 * 2*2*2 = 12 * 8 = 96. That's just schoolbook!
    
    # I made an error. The block decomposition doesn't reduce rank if we use schoolbook at both levels.
    # The rank IS n*m*p = 96 for schoolbook.
    
    # The key insight: Strassen reduces the number of MULTIPLICATIONS but in tensor decomposition,
    # the rank is the number of rank-1 terms. Schoolbook already has rank n*m*p.
    # Strassen has rank 7 for 2x2*2x2 vs schoolbook rank 4... wait, 7 > 4!
    # 
    # OH WAIT. I've been confused. For 2x2*2x2:
    # Schoolbook: 4 multiplications, rank 4.
    # Strassen: 7 multiplications, rank 7.
    # Strassen has HIGHER rank! It's faster in practice due to fewer additions,
    # but the tensor rank is higher!
    #
    # So the minimum known rank for 2x2*2x2 is 7 (Strassen is optimal for 2x2).
    # Schoolbook rank 4 is NOT a valid decomposition of the matrix multiplication tensor!
    #
    # Wait, let me reconsider. The tensor T[i,j,k] = 1 for all i,j,k in {0,1}.
    # Schoolbook: T[i,j,k] = sum_{j'} delta_{j,j'} = ... no.
    # 
    # Actually: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is: T[i,j,k] such that (AB)[i,k] = sum_j A[i,j] B[j,k] * T[i,j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k).
    # 
    # Factorization: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # 
    # Schoolbook: For each j, define U[i,j,r_j] = delta_{i,i} (always 1 for matching i,j), 
    # V[j,k,r_j] = delta_{j,j} (always 1 for matching j,k), W[k,i,r_j] = delta_{k,k} (always 1).
    # Wait, that gives T[i,j,k] = sum_j 1*1*1 = m, not 1.
    #
    # I think I've been misunderstanding the tensor decomposition. Let me re-read the task.
    #
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for (n x m) * (m x p) is:
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # Which is always 1 for valid indices.
    #
    # And the decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    #
    # For schoolbook with n=m=p=2:
    # r ranges over j (the summation index), so rank = m = 2? No, that doesn't work.
    #
    # Actually, I think the correct interpretation is:
    # The "multiplication tensor" encodes which products A[i,j]*B[j,k] contribute to C[i,k].
    # It's a 3-tensor where T[i,j,k] = 1 means "A[i,j] * B[j,k] contributes to C[i,k]".
    # For standard matrix multiplication, ALL terms contribute, so T[i,j,k] = 1 for all valid (i,j,k).
    #
    # The rank-1 decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    #
    # For 2x2*2x2, T is 2x2x2 with all entries 1.
    # Rank of this tensor: it's the matrix multiplication tensor of format (2,2,2).
    # The rank of the (2,2,2) matrix multiplication tensor is 7 (Strassen's result).
    #
    # So schoolbook rank 4 is WRONG. The correct schoolbook gives rank 4 only if we're
    # decomposing differently. Let me re-examine.
    #
    # Actually wait - I think the issue is that the "schoolbook" decomposition of the
    # multiplication tensor has rank m (the inner dimension), not n*m*p.
    # For 2x2*2x2: rank 2? That can't be right since Strassen says 7 is minimal.
    #
    # I'm clearly confused. Let me just verify with the code's _verify function.
    # The verify function checks: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k.
    #
    # For 2x2*2x2 with the Strassen factors I built earlier, let me check if _verify passes.
    # If it does, then rank 7 is correct for 2x2*2x2.
    #
    # For 4x4*4x6: the tensor T[i,j,k] = 1 for all i in 0..3, j in 0..3, k in 0..5.
    # We need: sum_r U[i*4+j,r] * V[j*6+k,r] * W[k*4+i,r] = 1 for all i,j,k.
    #
    # The rank of the (4,4,6) matrix multiplication tensor is what we're trying to minimize.
    # Known bounds: between 4*4*6/2 = 48 (trivial lower bound from dimension) and 96 (schoolbook).
    # Actually the trivial lower bound is max(n*m, m*p, p*n) / something... 
    # The true lower bound is more complex. For (4,4,6), best known is around 77-84.
    
    # Let me just build the Strassen-tiled version (rank 84) and verify it works,
    # then attempt rank reduction.
    
    U, V, W = _build_strassen_tiled()
    if _verify(U, V, W, 4, 4, 6):
        return U, V, W
    else:
        # Fallback: build schoolbook
        rank = 96
        U = np.zeros((16, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, col] = 1
                    V[j*6+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return U, V, W

def _build_strassen_tiled():
    """
    Build Strassen-tiled decomposition for 4x4 * 4x6.
    Uses 2x2 blocks: A is 2x2 blocks of 2x2, B is 2x3 blocks of 2x2.
    For each of the 12 block triples (ia,ib,ic), use Strassen 2x2*2x2 (rank 7).
    Total rank: 12 * 7 = 84.
    """
    import numpy as np
    n, m, p = 4, 4, 6
    U_s, V_s, W_s = _build_strassen_2x2()  # (4,7), (4,7), (4,7)
    
    total_rank = 12 * 7
    U = np.zeros((16, total_rank))
    V = np.zeros((24, total_rank))
    W = np.zeros((24, total_rank))
    
    col = 0
    for ia in range(2):
        for ib in range(2):
            for ic in range(3):
                for r in range(7):
                    c = col + r
                    # U: index (i*m+j) = (2*ia+ii)*4 + (2*ib+jj)
                    for ii in range(2):
                        for jj in range(2):
                            i = 2*ia + ii
                            j = 2*ib + jj
                            U[i*4 + j, c] = U_s[ii*2+jj, r]
                    # V: index (j*p+k) = (2*ib+jj)*6 + (2*ic+kk)
                    for jj in range(2):
                        for kk in range(2):
                            j = 2*ib + jj
                            k = 2*ic + kk
                            V[j*6 + k, c] = V_s[jj*2+kk, r]
                    # W: index (k*n+i) = (2*ic+kk)*4 + (2*ia+ii)
                    for kk in range(2):
                        for ii in range(2):
                            k = 2*ic + kk
                            i = 2*ia + ii
                            W[k*4 + i, c] = W_s[kk*2+ii, r]
                col += 7
    return U, V, W

def _rank_reduce(U, V, W, n, m, p, time_budget=100):
    """
    Attempt to reduce the rank by identifying redundant columns.
    A column r is redundant if removing it and adjusting others still satisfies the constraint.
    Simple approach: try removing each column and check if the remaining can be 
    linearly combined to still satisfy the constraint.
    
    More practical: try to find a subset of columns that still works.
    """
    import numpy as np
    start_time = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check which columns are actually needed.
    # A column r contributes U[:,r] ⊗ V[:,r] ⊗ W[:,r] to the tensor.
    # If this contribution is zero, the column is unnecessary.
    # More generally, if the column's contribution can be expressed as a linear combination
    # of other columns' contributions, it's redundant.
    
    # Simple check: for each column, see if removing it breaks the decomposition.
    # If the tensor value at each (i,j,k) depends on multiple columns, removing one
    # might still work if others compensate. But for exact decomposition, each column
    # contributes a specific rank-1 tensor, and we need their sum to equal the target.
    
    # Try: for each column, zero it out and check if the result is still valid.
    # This only works if the column was truly redundant (contribution = 0).
    
    # Better: try to find linear dependencies among the rank-1 tensors.
    # The rank-1 tensor for column r is T_r[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r].
    # If sum_{r in S} alpha_r * T_r = 0 for some non-trivial alpha, we can remove columns.
    
    # This is expensive. Let's just try removing columns one at a time.
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try removing each column
    for r in range(rank):
        if time.time() - start_time > time_budget:
            break
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        U_test[:, r] = 0
        V_test[:, r] = 0
        W_test[:, r] = 0
        if _verify(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p):
            # Column r is redundant
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            rank -= 1
            r = -1  # restart from beginning
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return best_U, best_V, best_W

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    start_time = time.time()
    deadline = 120.0
    
    # Build Strassen-tiled decomposition (rank 84)
    U, V, W = _build_strassen_tiled()
    
    # Verify
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    # Attempt rank reduction
    remaining_time = deadline - (time.time() - start_time) - 10.0  # 10s safety margin
    if remaining_time > 5:
        U, V, W = _rank_reduce(U, V, W, n, m, p, time_budget=remaining_time)
    
    # Final verification
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Rebuild from scratch
        U, V, W = _build_strassen_tiled()
    
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
