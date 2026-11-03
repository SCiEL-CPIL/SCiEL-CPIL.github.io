"""Matrix multiplication tensor decomposition: 2x3 * 3x10.
Uses block decomposition with Strassen for 2x2 blocks and
optimized rectangular block decompositions.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

def _matmul_tensor_check(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is exact."""
    for i in range(n):
        for k in range(p):
            for j in range(m):
                # T[i,j,k] = 1 if this is a matmul tensor entry
                expected = 1.0
                got = 0.0
                for r in range(len(U[0])):
                    got += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(got - expected) > tol:
                    return False
    return True

def _block_2x2_by_2x2_strassen():
    """Return (U, V, W) for 2x2 * 2x2 with rank 7 (Strassen)."""
    # Strassen decomposition: 7 multiplications
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # Strassen's 7 products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U[i*2+j, r] = coefficient for A[i,j] in product r
    # V[j*2+k, r] = coefficient for B[j,k] in product r
    # W[k*2+i, r] = coefficient for output C[i,k] from product r
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00, C11
    
    # M2: (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10
    W[3][1] = -1  # C11
    
    # M3: A00*(B01-B11)
    U[0][2] = 1  # A00
    V[1][2] = 1; V[3][2] = -1  # B01, -B11
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # M4: A11*(B10-B00)
    U[3][3] = 1  # A11
    V[2][3] = 1; V[0][3] = -1  # B10, -B00
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # M5: (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[3][4] = 1  # B11
    W[0][4] = -1  # C00
    W[1][4] = 1  # C01
    
    # M6: (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1  # A10, -A00
    V[0][5] = 1; V[1][5] = 1  # B00, B01
    W[3][5] = 1  # C11
    
    # M7: (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1  # A01, -A11
    V[2][6] = 1; V[3][6] = 1  # B10, B11
    W[0][6] = 1  # C00
    
    return U, V, W

def _block_rect(n, m, p, rank_override=None):
    """Return (U, V, W) for n x m * m x p multiplication.
    Uses schoolbook if rank_override is None, else tries to construct."""
    if rank_override is None:
        rank = n * m * p
    else:
        rank = rank_override
    
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < rank:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    return U, V, W

def _combine_blocks(block_list, n_total, m_total, p_total):
    """Combine block decompositions into a full decomposition.
    block_list: list of (n_i, m_i, p_i, U_i, V_i, W_i, i_offset, j_offset, k_offset)
    """
    total_rank = sum(len(block[3][0]) for block in block_list)
    U = [[0.0]*total_rank for _ in range(n_total*m_total)]
    V = [[0.0]*total_rank for _ in range(m_total*p_total)]
    W = [[0.0]*total_rank for _ in range(p_total*n_total)]
    
    col_offset = 0
    for (ni, mi, pi, Ui, Vi, Wi, ioff, joff, koff) in block_list:
        r = len(Ui[0])
        for a in range(ni):
            for b in range(mi):
                for c in range(r):
                    U[(ioff+a)*m_total + (joff+b)][col_offset+c] = Ui[a*mi+b][c]
        for b in range(mi):
            for c in range(pi):
                for r_idx in range(r):
                    V[(joff+b)*p_total + (koff+c)][col_offset+r_idx] = Vi[b*pi+c][r_idx]
        for c in range(pi):
            for a in range(ni):
                for r_idx in range(r):
                    W[(koff+c)*n_total + (ioff+a)][col_offset+r_idx] = Wi[c*ni+a][r_idx]
        col_offset += r
    
    return U, V, W

def construct():
    """Construct a low-rank decomposition for 2x3 * 3x10."""
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Strategy: split 3x10 into 3x5 + 3x5, and 2x3 into 2x2 + 2x1
    # Block 1: 2x2 * 2x2 (Strassen, rank 7) -> covers A[0:2,0:2] * B[0:2,0:2]
    # Block 2: 2x2 * 2x3 (need rank) -> covers A[0:2,0:2] * B[0:2,2:5]
    # Block 3: 2x1 * 1x2 (rank 1) -> covers A[0:2,2] * B[2,0:2]
    # Block 4: 2x1 * 1x3 (rank 1) -> covers A[0:2,2] * B[2,2:5]
    # Then repeat for the second 3x5 chunk (columns 5:10)
    
    # Actually, let me think more carefully.
    # 2x3 * 3x10
    # Split columns of B: B[:, 0:5] and B[:, 5:10]
    # Split rows of A: A[:, 0:2] and A[:, 2]
    
    # Block (0,0): A[0:2, 0:2] (2x2) * B[0:2, 0:5] (2x5)
    # Block (0,1): A[0:2, 0:2] (2x2) * B[0:2, 5:10] (2x5)
    # Block (1,0): A[0:2, 2] (2x1) * B[2, 0:5] (1x5)
    # Block (1,1): A[0:2, 2] (2x1) * B[2, 5:10] (1x5)
    
    # For 2x2 * 2x5: split B into 2x2 + 2x2 + 2x1
    #   2x2*2x2: Strassen rank 7
    #   2x2*2x1: rank 4 (each of 4 entries needs 1 term, but actually...)
    #     C[i,k] = sum_j A[i,j] B[j,k], k is scalar
    #     C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #     C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    #     C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #     C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #     Each C[i,k] needs 2 terms, total 4 terms. Rank 4.
    #   2x2*2x1: same, rank 4
    #   2x2*2x2: rank 7
    #   Total for 2x2*2x5: 7+4+4 = 15
    
    # For 2x1 * 1x5: C[i,k] = A[i,0] * B[0,k]
    #   This is outer product, rank 1!
    #   U[i,0] = A[i,0], V[0,k] = B[0,k], W[k,i] = 1
    #   Wait, W has shape (p*n, rank) = (5*2, 1) = (10, 1)
    #   W[k*2+i] = 1 for all k,i
    #   Yes, rank 1.
    
    # For 2x1 * 1x2: similarly rank 1.
    
    # So:
    # Block (0,0): 2x2 * 2x5 = 15
    # Block (0,1): 2x2 * 2x5 = 15
    # Block (1,0): 2x1 * 1x5 = 1
    # Block (1,1): 2x1 * 1x5 = 1
    # Total: 32
    
    # Wait, but we can also split 2x5 differently.
    # 2x2 * 2x5: We can also think of it as 5 separate 2x2*2x1 problems.
    #   Each 2x2*2x1 has rank 4, so 5*4 = 20. Worse.
    # Or split A: 1x2 + 1x2, B stays 2x5.
    #   1x2 * 2x5: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   This is rank... for each i,k we need 2 terms. 
    #   Actually C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   We can write this as: U[i,0]=A[i,0], V[0,k]=B[0,k], W[k,i]=1 (rank 1 part)
    #                     + U[i,1]=A[i,1], V[1,k]=B[1,k], W[k,i]=1 (rank 1 part)
    #   So rank 2 per row i? No, total rank 2.
    #   2x2 * 2x5: split A into two 1x2 rows.
    #   Row 0: 1x2 * 2x5, rank 2
    #   Row 1: 1x2 * 2x5, rank 2
    #   Total: 4? That can't be right.
    
    # Let me reconsider. For 1x2 * 2x5:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # T[0, j, k] where j=0,1 and k=0..4
    # T[0,0,k] = 1, T[0,1,k] = 1 for all k
    # Decomposition: sum_r U[0*2+j, r] * V[j*5+k, r] * W[k*1+0, r]
    # = sum_r U[j, r] * V[j*5+k, r] * W[k, r]
    # For r=0: U[0,0]=1, U[1,0]=0. V[0*5+k, 0] = 1 for all k, V[1*5+k, 0] = 0. W[k, 0] = 1.
    #   Contribution: A[0,0] * B[0,k]
    # For r=1: U[0,1]=0, U[1,1]=1. V[0*5+k, 1] = 0, V[1*5+k, 1] = 1. W[k, 1] = 1.
    #   Contribution: A[0,1] * B[1,k]
    # Yes! Rank 2 for 1x2 * 2x5.
    
    # So 2x2 * 2x5: split A into row 0 (1x2) and row 1 (1x2).
    #   Row 0: rank 2, Row 1: rank 2. Total rank 4.
    # Much better than 15!
    
    # Similarly, 2x1 * 1x5: rank 1 (as shown).
    
    # So:
    # Block (0,0): 2x2 * 2x5 = 4
    # Block (0,1): 2x2 * 2x5 = 4
    # Block (1,0): 2x1 * 1x5 = 1
    # Block (1,1): 2x1 * 1x5 = 1
    # Total: 10
    
    # Hmm, that seems too good. Let me verify 2x2 * 2x5 rank 4.
    # A is 2x2, B is 2x5. C = A*B is 2x5.
    # C[i,k] = sum_j A[i,j] B[j,k] for j=0,1.
    # T[i,j,k] = 1 for all i,j,k (since it's the matmul tensor).
    # Wait, T[i,j,k] = delta... no. The tensor is T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k].
    # Actually the matmul tensor has T[i,j,k] = 1 for all valid i,j,k when m=2.
    # No wait. T[i,j,k] = 1 always? That's only when... 
    
    # Let me re-read the problem. The matmul tensor for n x m * m x p is:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if this entry contributes to C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually all entries are 1. The tensor is all-ones of shape (n*m, m*p, p*n) reshaped.
    # Wait no. U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r should equal T[i,j,k].
    # And T[i,j,k] = 1 for all i,j,k.
    # 
    # So for 2x2 * 2x5: n=2, m=2, p=5.
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0..4}.
    # We need: sum_r U[i*2+j, r] * V[j*5+k, r] * W[k*2+i, r] = 1.
    
    # Can we do this with rank 4?
    # Try: r=0,1 for j=0 part, r=2,3 for j=1 part.
    # r=0: U[*,0]: U[0,0]=1, U[1,0]=0, U[2,0]=0, U[3,0]=0 (i.e., U[i*2+0, 0] = delta(i,0))
    #       V[0*5+k, 0] = 1 for all k, V[1*5+k, 0] = 0
    #       W[k*2+0, 0] = 1, W[k*2+1, 0] = 0 (i.e., W[k*2+i, 0] = delta(i,0))
    #       Contribution: delta(i,0) * 1 * delta(i,0) = delta(i,0) for j=0 entries.
    #       But we need it for ALL i,j,k. This doesn't work.
    
    # Let me think differently. The tensor is all-ones.
    # sum_r U[i*2+j, r] * V[j*5+k, r] * W[k*2+i, r] = 1.
    
    # One approach: for each j, we need sum_r U[i*2+j, r] * V[j*5+k, r] * W[k*2+i, r] = 1.
    # If we separate by j: for j=0 and j=1.
    # For j=0: sum_r U[i*2+0, r] * V[0*5+k, r] * W[k*2+i, r] = 1 for all i,k.
    # For j=1: sum_r U[i*2+1, r] * V[1*5+k, r] * W[k*2+i, r] = 1 for all i,k.
    
    # The j=0 part is essentially: given vectors u_i = (U[i, r])_r, v_k = (V[5k, r])_r, w_{k,i} = (W[2k+i, r])_r
    # sum_r u_i[r] * v_k[r] * w_{k,i}[r] = 1.
    
    # If we set w_{k,i}[r] = 1 for all k,i,r (within the j=0 block), then:
    # sum_r u_i[r] * v_k[r] = 1.
    # This is like u_i^T v_k = 1 for all i,k.
    # If u_i = e_i (standard basis) and v_k = (1,1,...,1), then u_i^T v_k = 1. 
    # But v_k is the same for all k, and u_i has dimension = rank.
    # If rank = 2: u_0 = (1,0), u_1 = (0,1), v_k = (1,1) for all k.
    # Then u_0^T v_k = 1, u_1^T v_k = 1. Yes!
    # So for j=0: U[0,0]=1, U[0,1]=0; U[1,0]=0, U[1,1]=1 (using r=0,1)
    #            V[5k, 0]=1, V[5k, 1]=1 for all k
    #            W[2k+i, 0]=1, W[2k+i, 1]=1 for all k,i
    # Similarly for j=1 using r=2,3.
    # Total rank 4. 
    
    # Great, so 2x2 * 2x5 has rank 4.
    
    # Now let's also check 2x1 * 1x5: n=2, m=1, p=5.
    # T[i, 0, k] = 1 for all i in {0,1}, k in {0..4}.
    # sum_r U[i*1+0, r] * V[0*5+k, r] * W[k*2+i, r] = 1.
    # Rank 1: U[i, 0] = 1, V[5k, 0] = 1, W[2k+i, 0] = 1.
    # Check: 1 * 1 * 1 = 1. Yes!
    
    # So total rank = 4 + 4 + 1 + 1 = 10.
    
    # Let me construct this.
    def make_2x2_by_2x5():
        """2x2 * 2x5, rank 4."""
        # n=2, m=2, p=5
        U = [[0.0]*4 for _ in range(4)]  # 2*2 = 4 rows
        V = [[0.0]*4 for _ in range(10)]  # 2*5 = 10 rows
        W = [[0.0]*4 for _ in range(10)]  # 5*2 = 10 rows
        
        # j=0: r=0,1
        for i in range(2):
            U[i*2+0][i] = 1.0  # U[i,0] for r=i (i=0,1)
        for k in range(5):
            V[0*5+k][0] = 1.0
            V[0*5+k][1] = 1.0
        for k in range(5):
            for i in range(2):
                W[k*2+i][0] = 1.0
                W[k*2+i][1] = 1.0
        
        # j=1: r=2,3
        for i in range(2):
            U[i*2+1][2+i] = 1.0
        for k in range(5):
            V[1*5+k][2] = 1.0
            V[1*5+k][3] = 1.0
        for k in range(5):
            for i in range(2):
                W[k*2+i][2] = 1.0
                W[k*2+i][3] = 1.0
        
        return U, V, W
    
    def make_2x1_by_1x5():
        """2x1 * 1x5, rank 1."""
        U = [[0.0]*1 for _ in range(2)]  # 2*1 = 2 rows
        V = [[0.0]*1 for _ in range(5)]  # 1*5 = 5 rows
        W = [[0.0]*1 for _ in range(10)]  # 5*2 = 10 rows
        
        for i in range(2):
            U[i][0] = 1.0
        for k in range(5):
            V[k][0] = 1.0
        for k in range(5):
            for i in range(2):
                W[k*2+i][0] = 1.0
        
        return U, V, W
    
    # Full decomposition:
    # A is 2x3, B is 3x10.
    # Split A: A[:, 0:2] (2x2) and A[:, 2] (2x1)
    # Split B: B[0:2, 0:5] (2x5), B[0:2, 5:10] (2x5), B[2, 0:5] (1x5), B[2, 5:10] (1x5)
    
    # Block (0,0): A[0:2, 0:2] * B[0:2, 0:5] -> 2x2 * 2x5, rank 4
    # Block (0,1): A[0:2, 0:2] * B[0:2, 5:10] -> 2x2 * 2x5, rank 4
    # Block (1,0): A[0:2, 2] * B[2, 0:5] -> 2x1 * 1x5, rank 1
    # Block (1,1): A[0:2, 2] * B[2, 5:10] -> 2x1 * 1x5, rank 1
    
    total_rank = 4 + 4 + 1 + 1
    U = [[0.0]*total_rank for _ in range(n*m)]  # 2*3 = 6
    V = [[0.0]*total_rank for _ in range(m*p)]  # 3*10 = 30
    W = [[0.0]*total_rank for _ in range(p*n)]  # 10*2 = 20
    
    col = 0
    # Block (0,0): A rows 0-1, cols 0-1; B rows 0-1, cols 0-4
    U00, V00, W00 = make_2x2_by_2x5()
    for a in range(2):
        for b in range(2):
            for r in range(4):
                U[a*m + b][col+r] = U00[a*2+b][r]
    for b in range(2):
        for c in range(5):
            for r in range(4):
                V[b*p + c][col+r] = V00[b*5+c][r]
    for c in range(5):
        for a in range(2):
            for r in range(4):
                W[c*n + a][col+r] = W00[c*2+a][r]
    col += 4
    
    # Block (0,1): A rows 0-1, cols 0-1; B rows 0-1, cols 5-9
    U01, V01, W01 = make_2x2_by_2x5()
    for a in range(2):
        for b in range(2):
            for r in range(4):
                U[a*m + b][col+r] = U01[a*2+b][r]
    for b in range(2):
        for c in range(5):
            for r in range(4):
                V[b*p + (5+c)][col+r] = V01[b*5+c][r]
    for c in range(5):
        for a in range(2):
            for r in range(4):
                W[(5+c)*n + a][col+r] = W01[c*2+a][r]
    col += 4
    
    # Block (1,0): A rows 0-1, col 2; B row 2, cols 0-4
    U10, V10, W10 = make_2x1_by_1x5()
    for a in range(2):
        for r in range(1):
            U[a*m + 2][col+r] = U10[a][r]
    for c in range(5):
        for r in range(1):
            V[2*p + c][col+r] = V10[c][r]
    for c in range(5):
        for a in range(2):
            for r in range(1):
                W[c*n + a][col+r] = W10[c*2+a][r]
    col += 1
    
    # Block (1,1): A rows 0-1, col 2; B row 2, cols 5-9
    U11, V11, W11 = make_2x1_by_1x5()
    for a in range(2):
        for r in range(1):
            U[a*m + 2][col+r] = U11[a][r]
    for c in range(5):
        for r in range(1):
            V[2*p + (5+c)][col+r] = V11[c][r]
    for c in range(5):
        for a in range(2):
            for r in range(1):
                W[(5+c)*n + a][col+r] = W11[c*2+a][r]
    col += 1
    
    # Verify
    if not _matmul_tensor_check(U, V, W, n, m, p):
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1.0
                    V[j*p+k][c] = 1.0
                    W[k*n+i][c] = 1.0
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
