"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import time

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Check exact decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Actually the tensor is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank) with W indexed as W[k*n+i, r].
    The multiplication tensor T[i,j,k] = delta(i,j,k) in the sense that
    C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i,j,r]) (sum_k ... )
    
    More precisely: C[i,k] = sum_j A[i,j] B[j,k].
    We write C[i,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r]) ... no.
    
    Standard: C[i,k] = sum_r M_r * N_r where M_r = sum_j A[i,j] U[i*m+j,r] ... 
    
    Actually the correct decomposition is:
    C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r])
    This doesn't work directly. Let me think again.
    
    The matrix multiplication tensor T has T[i,j,k] = 1 if these indices form a valid product term.
    The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    So for the schoolbook: U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k) triple.
    Then sum over r gives 1 for each (i,j,k) which is correct since T[i,j,k]=1 for all valid triples.
    
    So verification: for all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # U_arr shape: (n*m, rank), V_arr: (m*p, rank), W_arr: (p*n, rank)
    # For each (i,j,k): U_arr[i*m+j] dot V_arr[j*p+k] dot W_arr[k*n+i]
    # This is a 3-tensor contraction. Let's do it efficiently.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    # Contract over r: result[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    # Use einsum
    result = np.einsum('ijr,jkr,kr->ijk', U_reshaped, V_reshaped, W_reshaped)
    # Check if all entries are 1 (within tolerance for float, but we want exact for int/half-int)
    if not np.allclose(result, 1.0, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook rank 50
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # Strategy: Use Strassen-like block decomposition
    # Partition the 5x5 B matrix into blocks. Since 5 = 2+2+1, we can partition
    # B into three 2x2 blocks and one 1x1 block, but that's asymmetric.
    # Actually, let's partition rows and columns of B (m x p = 5x5) into blocks.
    # And A is 2x5, C is 2x5.
    
    # A is 2x5, B is 5x5, C is 2x5.
    # Let's partition the 5 columns of A (and rows of B) into blocks of sizes 2,2,1.
    # Let's partition the 5 columns of C (and columns of B) into blocks of sizes 2,2,1.
    
    # So A = [A1 | A2 | A3] where A1 is 2x2, A2 is 2x2, A3 is 2x1
    # B = [B11 B12 B13; B21 B22 B23; B31 B32 B33] where B11, B22 are 2x2, B13, B23, B31, B32 are 2x1 or 1x2, B33 is 1x1
    # C = [C1 | C2 | C3] where C1 is 2x2, C2 is 2x2, C3 is 2x1
    
    # C1 = A1*B11 + A2*B21 + A3*B31
    # C2 = A1*B12 + A2*B22 + A3*B32
    # C3 = A1*B13 + A2*B23 + A3*B33
    
    # For 2x2 * 2x2, Strassen uses 7 multiplications instead of 8.
    # We have two 2x2*2x2 products: (A1*B11) and (A2*B22).
    # The other products involve 2x2 * 2x1, 2x1 * 2x2, 2x1 * 1x1, etc.
    
    # Let's count the schoolbook rank for this partition:
    # A1*B11: 2x2 * 2x2 = 4 mults
    # A2*B21: 2x2 * 2x1 = 4 mults
    # A3*B31: 2x1 * 1x2 = 2 mults
    # A1*B12: 2x2 * 2x2 = 4 mults
    # A2*B22: 2x2 * 2x2 = 4 mults
    # A3*B32: 2x1 * 1x2 = 2 mults
    # A1*B13: 2x2 * 1x1 = 2 mults
    # A2*B23: 2x2 * 1x1 = 2 mults
    # A3*B33: 2x1 * 1x1 = 2 mults
    # Total: 4+4+2+4+4+2+2+2+2 = 26 mults... wait that's less than 50?
    
    # No, I'm confusing myself. The rank is the number of scalar multiplications.
    # For A1*B11 where A1 is 2x2 and B11 is 2x2, schoolbook is 4 mults.
    # But the total schoolbook is 2*5*5 = 50.
    
    # Let me recount: A is 2x5, B is 5x5. Each entry of C[i,k] = sum_j A[i,j]*B[j,k].
    # There are 2*5 = 10 entries in C, each requiring 5 mults, total 50.
    
    # With block partition 2,2,1 on both the 5-dim of A/B-rows and B-cols/C:
    # The 5x5 B is partitioned into 3x3 blocks of sizes (2,2,1)x(2,2,1).
    # A is partitioned into 3 blocks of sizes (2,2,1) along the 5-dim.
    # C is partitioned into 3 blocks of sizes (2,2,1) along the 5-dim.
    
    # C[a,b] = sum_c A[a,c] * B[c,d] where a,b,c are block indices.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # Each term is a matrix product of the appropriate sizes.
    
    # Rank contributions:
    # A[0,0]*B[0,0]: 2x2 * 2x2 -> 4
    # A[0,1]*B[1,0]: 2x2 * 2x2 -> 4
    # A[0,2]*B[2,0]: 2x1 * 1x2 -> 2
    # A[1,0]*B[0,1]: 2x2 * 2x2 -> 4
    # A[1,1]*B[1,1]: 2x2 * 2x2 -> 4
    # A[1,2]*B[2,1]: 2x1 * 1x2 -> 2
    # A[2,0]*B[0,2]: 2x1 * 2x1 -> 2
    # A[2,1]*B[1,2]: 2x1 * 2x1 -> 2
    # A[2,2]*B[2,2]: 2x1 * 1x1 -> 2
    # Total: 4+4+2+4+4+2+2+2+2 = 26
    
    # Wait, that's 26, but schoolbook is 50. That can't be right.
    # Oh I see, I'm miscounting. The block A[0,0] is 2x2, B[0,0] is 2x2.
    # Their product is 2x2, requiring 4 scalar mults in schoolbook.
    # But the total number of scalar mults should still be 50.
    
    # Let me recount: sum over all block pairs of (rows_A_block * cols_A_block * cols_B_block)
    # = sum_a sum_b sum_c (size_A[a,c] * size_B[c,b])
    # where size_A[a,c] = row_size[a] * col_size[c] and size_B[c,b] = row_size[c] * col_size[b]
    # with row_size = [2,2,1] and col_size = [2,2,1].
    
    # Total = sum_c (sum_a row_size[a]) * (sum_b col_size[b]) * ... no.
    # Total = sum_a sum_b sum_c row_size[a] * col_size[c] * row_size[c] * col_size[b]
    # = (sum_a row_size[a]) * (sum_c col_size[c] * row_size[c]) * (sum_b col_size[b])
    # = 5 * (2*2 + 2*2 + 1*1) * 5 = 5 * 9 * 5 = 225? No that's wrong too.
    
    # Let me just directly count: for each (i,j,k) triple, there's one mult.
    # i in [0,2), j in [0,5), k in [0,5). Total = 2*5*5 = 50.
    
    # The block decomposition doesn't change the total number of scalar mults in schoolbook.
    # It just groups them. The rank is still 50 for schoolbook.
    
    # The key insight is that Strassen reduces the rank for 2x2*2x2 from 4 to 7? No, Strassen is 7 vs 8 for 2x2*2x2... wait.
    # Strassen: 2x2 * 2x2 uses 7 mults instead of 8. But schoolbook for 2x2*2x2 is 4 mults, not 8.
    
    # I'm confusing myself. Let me restart the analysis.
    
    # For multiplying an n x m matrix by an m x p matrix, schoolbook uses n*m*p scalar mults.
    # For 2x2 * 2x2, that's 8 mults. Strassen uses 7.
    
    # So if we have a 2x2 * 2x2 subproblem, Strassen saves 1 mult.
    
    # In our 2x5 * 5x5 problem, if we can identify disjoint 2x2*2x2 subproblems, we can save.
    
    # Partition the 5-dim into 2+2+1. We have two 2x2 blocks in the 5-dim.
    # The 2x5 * 5x5 can be seen as having subproducts:
    # - (2x2) * (2x2) for the first 2 rows of B and first 2 cols of B: 8 mults -> 7 with Strassen
    # - (2x2) * (2x2) for the first 2 rows of B and next 2 cols of B: 8 mults -> 7 with Strassen
    # - etc.
    
    # Let me be more careful. The 5x5 B matrix has blocks:
    # B = [B11 B12 B13]
    #     [B21 B22 B23]
    #     [B31 B32 B33]
    # where B11, B22 are 2x2, B12, B21, B13, B23, B31, B32 are 2x1 or 1x2, B33 is 1x1.
    
    # A = [A1 A2 A3] where A1, A2 are 2x2, A3 is 2x1.
    # C = [C1 C2 C3] where C1, C2 are 2x2, C3 is 2x1.
    
    # C1 = A1*B11 + A2*B21 + A3*B31
    # C2 = A1*B12 + A2*B22 + A3*B32
    # C3 = A1*B13 + A2*B23 + A3*B33
    
    # Schoolbook mults:
    # A1*B11: 2x2*2x2 = 8
    # A2*B21: 2x2*2x2 = 8
    # A3*B31: 2x1*1x2 = 2
    # A1*B12: 2x2*2x2 = 8
    # A2*B22: 2x2*2x2 = 8
    # A3*B32: 2x1*1x2 = 2
    # A1*B13: 2x2*1x1 = 2
    # A2*B23: 2x2*1x1 = 2
    # A3*B33: 2x1*1x1 = 2
    # Total: 8+8+2+8+8+2+2+2+2 = 42
    
    # Wait, that's 42, not 50. Let me recount.
    # A1*B11: A1 is 2x2, B11 is 2x2. Product is 2x2. Schoolbook: 4 mults per output entry, 4 entries = 16? No.
    # For 2x2 * 2x2, each of the 4 output entries requires 2 mults, total 8 mults.
    
    # But the total should be 50. Let me recount all:
    # A1*B11: 8
    # A2*B21: 8
    # A3*B31: 2 (A3 is 2x1, B31 is 1x2, product is 2x2, each entry 1 mult, 4 entries = 4? No, 2x1 * 1x2: each of 4 entries is 1 mult, total 4)
    
    # I keep making errors. Let me just use the formula: for a (r1 x c1) * (r2 x c2) product,
    # the number of scalar mults is r1 * c1 * c2 = r1 * r2 * c2 (since c1 = r2).
    
    # A1*B11: 2x2 * 2x2: 2*2*2 = 8
    # A2*B21: 2x2 * 2x2: 2*2*2 = 8
    # A3*B31: 2x1 * 1x2: 2*1*2 = 4
    # A1*B12: 2x2 * 2x2: 8
    # A2*B22: 2x2 * 2x2: 8
    # A3*B32: 2x1 * 1x2: 4
    # A1*B13: 2x2 * 1x1: 2*2*1 = 4
    # A2*B23: 2x2 * 1x1: 4
    # A3*B33: 2x1 * 1x1: 2*1*1 = 2
    # Total: 8+8+4+8+8+4+4+4+2 = 50. Good.
    
    # Now, Strassen applies to 2x2*2x2, reducing 8 to 7.
    # We have four 2x2*2x2 subproducts: A1*B11, A2*B21, A1*B12, A2*B22.
    # If we can apply Strassen to each, we save 4 mults, giving rank 46.
    
    # But can we? Strassen requires the subproblems to be independent.
    # The four 2x2*2x2 products share A1, A2, B11, B12, B21, B22.
    # They are not independent in the sense that they share factors.
    # However, for tensor decomposition, we can still use Strassen on each block separately
    # if the decomposition is block-diagonal in some sense.
    
    # Actually, the rank of a direct sum of tensors is the sum of ranks.
    # But these are not a direct sum; they share the A and B factors.
    
    # Let me think differently. The multiplication tensor for 2x5*5x5 can be decomposed
    # by considering the 5-dim as a direct sum of 2+2+1.
    # The tensor T = sum_{a,b,c} T[a,b,c] where T[a,b,c] is the tensor for the (a,b,c) block.
    # But T is not a direct sum; it's a sum of tensors that share indices.
    
    # I think the correct approach is:
    # The rank of the full tensor is at most the sum of ranks of the block tensors
    # when the blocks are "independent" in a certain sense.
    
    # Let me just try to construct a rank-46 decomposition using Strassen on the 2x2*2x2 blocks.
    
    # For the 2x2*2x2 Strassen decomposition, the 7 mults are:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For our problem, let's apply this to the A1*B11 product.
    # A1 = [[a11, a12], [a21, a22]], B11 = [[b11, b12], [b21, b22]]
    # The 7 mults involve linear combinations of entries of A1 and B11.
    
    # Similarly for A2*B21, A1*B12, A2*B22.
    
    # The non-Strassen parts:
    # A3*B31: 4 mults (schoolbook)
    # A3*B32: 4 mults
    # A1*B13: 4 mults
    # A2*B23: 4 mults
    # A3*B33: 2 mults
    # Total non-Strassen: 4+4+4+4+2 = 18
    
    # Strassen parts: 4 blocks * 7 = 28
    # Total: 28 + 18 = 46
    
    # Let's construct this decomposition.
    
    # Indexing:
    # A[i,j] where i in {0,1}, j in {0,...,4}
    # B[j,k] where j in {0,...,4}, k in {0,...,4}
    # C[i,k] where i in {0,1}, k in {0,...,4}
    
    # Block partition of the 5-dim: indices 0,1 -> block 0; indices 2,3 -> block 1; index 4 -> block 2.
    
    # A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2), A3 = A[:, 4] (2x1)
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B13 = B[0:2, 4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4], B23 = B[2:4, 4]
    # B31 = B[4, 0:2], B32 = B[4, 2:4], B33 = B[4, 4]
    
    # C1 = C[:, 0:2], C2 = C[:, 2:4], C3 = C[:, 4]
    
    # C1 = A1*B11 + A2*B21 + A3*B31
    # C2 = A1*B12 + A2*B22 + A3*B32
    # C3 = A1*B13 + A2*B23 + A3*B33
    
    # Strassen for A1*B11:
    # Let A1 = [[a00, a01], [a10, a11]] (using A[0,0], A[0,1], A[1,0], A[1,1])
    # B11 = [[b00, b01], [b10, b11]] (using B[0,0], B[0,1], B[1,0], B[1,1])
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # C1[0,0] = M1 + M4 - M5 + M7
    # C1[0,1] = M3 + M5
    # C1[1,0] = M2 + M4
    # C1[1,1] = M1 - M2 + M3 + M6
    
    # Similarly for A2*B21, A1*B12, A2*B22.
    
    # This is getting complex. Let me just hardcode the rank-46 decomposition.
    
    # Actually, let me try a simpler approach: just return the schoolbook rank-50 for now,
    # and add a time guard. The task says the current program scores 1.0, and we need to improve.
    
    # Let me try to implement the rank-46 decomposition.
    
    # I'll use numpy to build the U, V, W matrices.
    
    # Total rank: 4 Strassen blocks * 7 + 18 schoolbook = 28 + 18 = 46
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # 10 x 46
    V = [[0] * rank for _ in range(m * p)]  # 25 x 46
    W = [[0] * rank for _ in range(p * n)]  # 10 x 46
    
    col = 0
    
    # Strassen block 1: A1*B11
    # A1 entries: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices: 0, 1, 5, 6
    # B11 entries: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices: 0, 1, 5, 6
    # C1 entries: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices: 0, 1, 5, 6
    
    # M1 = (a00+a11)*(b00+b11)
    # U: A[0,0]=1, A[1,1]=1 -> U[0, col]=1, U[6, col]=1
    # V: B[0,0]=1, B[1,1]=1 -> V[0, col]=1, V[6, col]=1
    # W: C[0,0]=1, C[0,1]=0, C[1,0]=0, C[1,1]=1 -> W[0, col]=1, W[6, col]=1
    U[0][col] = 1; U[6][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[6][col] = 1
    col += 1
    
    # M2 = (a10+a11)*b00
    U[5][col] = 1; U[6][col] = 1
    V[0][col] = 1
    W[5][col] = 1
    col += 1
    
    # M3 = a00*(b01-b11)
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[1][col] = 1
    col += 1
    
    # M4 = a11*(b10-b00)
    U[6][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[5][col] = 1
    col += 1
    
    # M5 = (a00+a01)*b11
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6 = (a10-a00)*(b00+b01)
    U[5][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[6][col] = 1
    col += 1
    
    # M7 = (a01-a11)*(b10+b11)
    U[1][col] = 1; U[6][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[6][col] = -1
    col += 1
    
    # Strassen block 2: A2*B21
    # A2 entries: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices: 2, 3, 7, 8
    # B21 entries: B[2,0], B[2,1], B[3,0], B[3,1] -> V indices: 10, 11, 15, 16
    # C1 entries: same as before
    
    # M1 = (a00+a11)*(b00+b11) where a00=A[0,2], a11=A[1,3], b00=B[2,0], b11=B[3,1]
    U[2][col] = 1; U[8][col] = 1
    V[10][col] = 1; V[16][col] = 1
    W[0][col] = 1; W[6][col] = 1
    col += 1
    
    # M2 = (a10+a11)*b00 where a10=A[1,2], a11=A[1,3], b00=B[2,0]
    U[7][col] = 1; U[8][col] = 1
    V[10][col] = 1
    W[5][col] = 1
    col += 1
    
    # M3 = a00*(b01-b11) where a00=A[0,2], b01=B[2,1], b11=B[3,1]
    U[2][col] = 1
    V[11][col] = 1; V[16][col] = -1
    W[1][col] = 1
    col += 1
    
    # M4 = a11*(b10-b00) where a11=A[1,3], b10=B[3,0], b00=B[2,0]
    U[8][col] = 1
    V[15][col] = 1; V[10][col] = -1
    W[0][col] = 1; W[5][col] = 1
    col += 1
    
    # M5 = (a00+a01)*b11 where a00=A[0,2], a01=A[0,3], b11=B[3,1]
    U[2][col] = 1; U[3][col] = 1
    V[16][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6 = (a10-a00)*(b00+b01) where a10=A[1,2], a00=A[0,2], b00=B[2,0], b01=B[2,1]
    U[7][col] = 1; U[2][col] = -1
    V[10][col] = 1; V[11][col] = 1
    W[6][col] = 1
    col += 1
    
    # M7 = (a01-a11)*(b10+b11) where a01=A[0,3], a11=A[1,3], b10=B[3,0], b11=B[3,1]
    U[3][col] = 1; U[8][col] = -1
    V[15][col] = 1; V[16][col] = 1
    W[0][col] = 1; W[6][col] = -1
    col += 1
    
    # Strassen block 3: A1*B12
    # A1 entries: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices: 0, 1, 5, 6
    # B12 entries: B[0,2], B[0,3], B[1,2], B[1,3] -> V indices: 2, 3, 7, 8
    # C2 entries: C[0,2], C[0,3], C[1,2], C[1,3] -> W indices: 2, 3, 7, 8
    
    # M1 = (a00+a11)*(b00+b11) where b00=B[0,2], b11=B[1,3]
    U[0][col] = 1; U[6][col] = 1
    V[2][col] = 1; V[8][col] = 1
    W[2][col] = 1; W[8][col] = 1
    col += 1
    
    # M2 = (a10+a11)*b00 where b00=B[0,2]
    U[5][col] = 1; U[6][col] = 1
    V[2][col] = 1
    W[7][col] = 1
    col += 1
    
    # M3 = a00*(b01-b11) where b01=B[0,3], b11=B[1,3]
    U[0][col] = 1
    V[3][col] = 1; V[8][col] = -1
    W[3][col] = 1
    col += 1
    
    # M4 = a11*(b10-b00) where b10=B[1,2], b00=B[0,2]
    U[6][col] = 1
    V[7][col] = 1; V[2][col] = -1
    W[2][col] = 1; W[7][col] = 1
    col += 1
    
    # M5 = (a00+a01)*b11 where b11=B[1,3]
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1
    W[2][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (a10-a00)*(b00+b01) where b00=B[0,2], b01=B[0,3]
    U[5][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[8][col] = 1
    col += 1
    
    # M7 = (a01-a11)*(b10+b11) where b10=B[1,2], b11=B[1,3]
    U[1][col] = 1; U[6][col] = -1
    V[7][col] = 1; V[8][col] = 1
    W[2][col] = 1; W[8][col] = -1
    col += 1
    
    # Strassen block 4: A2*B22
    # A2 entries: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices: 2, 3, 7, 8
    # B22 entries: B[2,2], B[2,3], B[3,2], B[3,3] -> V indices: 12, 13, 17, 18
    # C2 entries: C[0,2], C[0,3], C[1,2], C[1,3] -> W indices: 2, 3, 7, 8
    
    # M1 = (a00+a11)*(b00+b11) where a00=A[0,2], a11=A[1,3], b00=B[2,2], b11=B[3,3]
    U[2][col] = 1; U[8][col] = 1
    V[12][col] = 1; V[18][col] = 1
    W[2][col] = 1; W[8][col] = 1
    col += 1
    
    # M2 = (a10+a11)*b00 where a10=A[1,2], a11=A[1,3], b00=B[2,2]
    U[7][col] = 1; U[8][col] = 1
    V[12][col] = 1
    W[7][col] = 1
    col += 1
    
    # M3 = a00*(b01-b11) where a00=A[0,2], b01=B[2,3], b11=B[3,3]
    U[2][col] = 1
    V[13][col] = 1; V[18][col] = -1
    W[3][col] = 1
    col += 1
    
    # M4 = a11*(b10-b00) where a11=A[1,3], b10=B[3,2], b00=B[2,2]
    U[8][col] = 1
    V[17][col] = 1; V[12][col] = -1
    W[2][col] = 1; W[7][col] = 1
    col += 1
    
    # M5 = (a00+a01)*b11 where a00=A[0,2], a01=A[0,3], b11=B[3,3]
    U[2][col] = 1; U[3][col] = 1
    V[18][col] = 1
    W[2][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (a10-a00)*(b00+b01) where a10=A[1,2], a00=A[0,2], b00=B[2,2], b01=B[2,3]
    U[7][col] = 1; U[2][col] = -1
    V[12][col] = 1; V[13][col] = 1
    W[8][col] = 1
    col += 1
    
    # M7 = (a01-a11)*(b10+b11) where a01=A[0,3], a11=A[1,3], b10=B[3,2], b11=B[3,3]
    U[3][col] = 1; U[8][col] = -1
    V[17][col] = 1; V[18][col] = 1
    W[2][col] = 1; W[8][col] = -1
    col += 1
    
    # Now the schoolbook parts:
    # A3*B31: A3 = A[:, 4] (2x1), B31 = B[4, 0:2] (1x2)
    # C31 = C[:, 0:2] (2x2)
    # 4 mults: A[0,4]*B[4,0], A[0,4]*B[4,1], A[1,4]*B[4,0], A[1,4]*B[4,1]
    
    # A[0,4] is U index 4, B[4,0] is V index 20, C[0,0] is W index 0
    U[4][col] = 1; V[20][col] = 1; W[0][col] = 1
    col += 1
    U[4][col] = 1; V[21][col] = 1; W[1][col] = 1
    col += 1
    U[9][col] = 1; V[20][col] = 1; W[5][col] = 1
    col += 1
    U[9][col] = 1; V[21][col] = 1; W[6][col] = 1
    col += 1
    
    # A3*B32: A3 = A[:, 4] (2x1), B32 = B[4, 2:4] (1x2)
    # C32 = C[:, 2:4] (2x2)
    # 4 mults
    
    U[4][col] = 1; V[22][col] = 1; W[2][col] = 1
    col += 1
    U[4][col] = 1; V[23][col] = 1; W[3][col] = 1
    col += 1
    U[9][col] = 1; V[22][col] = 1; W[7][col] = 1
    col += 1
    U[9][col] = 1; V[23][col] = 1; W[8][col] = 1
    col += 1
    
    # A1*B13: A1 = A[:, 0:2] (2x2), B13 = B[0:2, 4] (2x1)
    # C13 = C[:, 4] (2x1)
    # 4 mults: A[0,0]*B[0,4], A[0,1]*B[1,4], A[1,0]*B[0,4], A[1,1]*B[1,4]
    
    U[0][col] = 1; V[4][col] = 1; W[4][col] = 1
    col += 1
    U[1][col] = 1; V[9][col] = 1; W[4][col] = 1
    col += 1
    U[5][col] = 1; V[4][col] = 1; W[9][col] = 1
    col += 1
    U[6][col] = 1; V[9][col] = 1; W[9][col] = 1
    col += 1
    
    # A2*B23: A2 = A[:, 2:4] (2x2), B23 = B[2:4, 4] (2x1)
    # C23 = C[:, 4] (2x1)
    # 4 mults
    
    U[2][col] = 1; V[14][col] = 1; W[4][col] = 1
    col += 1
    U[3][col] = 1; V[19][col] = 1; W[4][col] = 1
    col += 1
    U[7][col] = 1; V[14][col] = 1; W[9][col] = 1
    col += 1
    U[8][col] = 1; V[19][col] = 1; W[9][col] = 1
    col += 1
    
    # A3*B33: A3 = A[:, 4] (2x1), B33 = B[4, 4] (1x1)
    # C33 = C[:, 4] (2x1)
    # 2 mults
    
    U[4][col] = 1; V[24][col] = 1; W[4][col] = 1
    col += 1
    U[9][col] = 1; V[24][col] = 1; W[9][col] = 1
    col += 1
    
    # Check that col == rank
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify the decomposition
    if _verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook
        return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
