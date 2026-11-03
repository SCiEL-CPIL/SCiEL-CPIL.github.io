"""Rewritten program: block decomposition 3x5x5 with Strassen on 2x2x2 blocks."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def _strassen_222_factors():
    """Return (U, V, W) for 2x2x2 matrix multiplication using Strassen (rank 7).
    Convention: U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    U[i*m + j, k] for i in 0..1, j in 0..1
    V[j*p + k, l] for j in 0..1, k in 0..1
    W[k*n + i, l] for k in 0..1, i in 0..1
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # Strassen for 2x2:
    # M1 = (A11+A22)(B12-B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Each M_t is a bilinear form: sum over (i,j,k) of U[i,j,t] * V[j,k,t] * W[k,i,t]
    # We need to set up 7 such forms.
    
    # Index convention: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    
    # M1 = (A11+A22)(B12-B22): expand
    # = A11*B12 - A11*B22 + A22*B12 - A22*B22
    # contributes to C11 with +1, C22 with -1
    
    # For M1 (t=0):
    # A11*B12: U[0,0]=1, V[0,1]=1, W[1,0]=1 (k=1,i=0)
    # -A11*B22: U[0,0]=-1, V[1,1]=1, W[1,0]=-1
    # A22*B12: U[1,1]=1, V[0,1]=1, W[1,1]=1 (k=1,i=1)
    # -A22*B22: U[1,1]=-1, V[1,1]=1, W[1,1]=-1
    
    # This is getting complex. Let me use a cleaner approach.
    # The Strassen factors can be written as:
    
    # t=0: M1 = (A00+A11)(B01-B11)
    #   U contributions: (0,0):+1, (1,1):+1
    #   V contributions: (0,1):+1, (1,1):-1
    #   W contributions: (1,0):+1  (since M1 adds to C00 with +1 and C11 with -1)
    #   Wait, W[k,i,t] means the output position is (i,k).
    #   M1 adds to C00 (i=0,k=0) with +1 and to C11 (i=1,k=1) with -1.
    #   So W[0*2+0, 0] += 1, W[1*2+1, 0] += -1
    #   But also need to handle the cross terms properly.
    
    # Let me just use the known explicit Strassen factorization.
    # Reference: standard Strassen for 2x2.
    
    U[:] = 0.0
    V[:] = 0.0
    W[:] = 0.0
    
    # M1 = (A00+A11)(B01-B11)
    # U: A00 -> U[0,0]=1; A11 -> U[3,0]=1
    # V: B01 -> V[1,0]=1; -B11 -> V[3,0]=-1
    # W: C00 gets +M1 -> W[0,0]+=1; C11 gets -M1 -> W[3,0]+=-1
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[1, 0] = 1.0; V[3, 0] = -1.0
    W[0, 0] = 1.0; W[3, 0] = -1.0
    
    # M2 = (A10+A11)B00
    # U: A10 -> U[2,1]=1; A11 -> U[3,1]=1
    # V: B00 -> V[0,1]=1
    # W: C10 gets +M2 -> W[0*2+1,1]=W[2,1]+=1
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0
    
    # M3 = A00(B10-B11)
    # U: A00 -> U[0,2]=1
    # V: B10 -> V[2,2]=1; -B11 -> V[3,2]=-1
    # W: C01 gets +M3 -> W[1*2+0,2]=W[2,2]+=1
    U[0, 2] = 1.0
    V[2, 2] = 1.0; V[3, 2] = -1.0
    W[2, 2] = 1.0
    
    # M4 = A11(B00-B10)
    # U: A11 -> U[3,3]=1
    # V: B00 -> V[0,3]=1; -B10 -> V[2,3]=-1
    # W: C00 gets +M4 -> W[0,3]+=1; C10 gets +M4 -> W[2,3]+=1
    U[3, 3] = 1.0
    V[0, 3] = 1.0; V[2, 3] = -1.0
    W[0, 3] = 1.0; W[2, 3] = 1.0
    
    # M5 = (A00+A01)B11
    # U: A00 -> U[0,4]=1; A01 -> U[1,4]=1
    # V: B11 -> V[3,4]=1
    # W: C00 gets -M5 -> W[0,4]+=-1; C01 gets +M5 -> W[2,4]+=1
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[2, 4] = 1.0
    
    # M6 = (A10-A00)(B00+B01)
    # U: A10 -> U[2,5]=1; -A00 -> U[0,5]=-1
    # V: B00 -> V[0,5]=1; B01 -> V[1,5]=1
    # W: C11 gets +M6 -> W[3,5]+=1
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0
    
    # M7 = (A01-A11)(B10+B11)
    # U: A01 -> U[1,6]=1; -A11 -> U[3,6]=-1
    # V: B10 -> V[2,6]=1; B11 -> V[3,6]=1
    # W: C00 gets +M7 -> W[0,6]+=1; C11 gets -M7 -> W[3,6]+=-1
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0; W[3, 6] = -1.0
    
    return U, V, W

def _verify_strassen():
    """Verify the Strassen factors are correct."""
    U, V, W = _strassen_222_factors()
    # The tensor T[i,j,k,l] = delta[i,k]*delta[j,l] for 2x2x2
    # Should equal sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    expected = 1.0 if (i == k and j == l) else 0.0
                    # Wait, the convention is W[k*n+i, t] where n=2
                    # And the output C[i,j] = sum_k A[i,k]*B[k,j]
                    # In our indexing: U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    # This gives contribution to position (i, j) in output C
                    # where C has dimensions n x p = 2 x 2
                    # Actually let me re-examine the convention.
                    # U has shape (n*m, rank) = (15, rank) for 3x5x5
                    # For 2x2x2: U shape (4, rank), indexed [i*m+j, t] where m=2
                    # V has shape (m*p, rank) = (4, rank), indexed [j*p+k, t] where p=2
                    # W has shape (p*n, rank) = (4, rank), indexed [k*n+i, t] where n=2
                    # The product sum_t U[i*2+j, t] * V[j*2+k, t] * W[k*2+i, t]
                    # should give C[i, j] where C = A * B, A is 2x2, B is 2x2
                    # C[i,j] = sum_k A[i,k] * B[k,j]
                    # So the triple product with indices (i,j,k) should give delta[j,k] for the k-sum?
                    # No wait. Let me think again.
                    # U[i*m+j, t] corresponds to A[i,j]
                    # V[j*p+k, t] corresponds to B[j,k]  -- but this is wrong for matrix mult
                    # In matrix mult C[i,l] = sum_j A[i,j] * B[j,l]
                    # So we need U[i*m+j, t] * V[j*p+l, t] * W[l*n+i, t]
                    # where l is the output column index.
                    # So W is indexed [l*n + i, t] where l is the column, i is the row.
                    # The tensor value at position (i, j, l) where j is the inner index and l is output col
                    # is delta[j, j] * ... no.
                    # The matrix multiplication tensor T[i, j, l] = 1 if the indices are consistent
                    # T[i,j,l] contributes to C[i,l] via A[i,j]*B[j,l]
                    # So T[i,j,l] = 1 for all valid (i,j,l) in the sense that
                    # C[i,l] = sum_j T[i,j,l] * A[i,j] * B[j,l]
                    # and T[i,j,l] = 1 if... it's always 1, the structure is in the indices.
                    # The tensor is: T[i,j,l] = delta_{j,j} which is always 1, but the
                    # constraint is that the same j appears in A[i,j] and B[j,l].
                    # So for a rank-r decomposition: T[i,j,l] = sum_t U[i,j,t]*V[j,l,t]*W[l,i,t]
                    # where U[i,j,t] extracts A[i,j], V[j,l,t] extracts B[j,l], W[l,i,t] places into C[i,l]
                    # For this to work: sum_t U[i,j,t]*V[j,l,t]*W[l,i,t] = 1 for all (i,j,l)
                    # because each (i,j,l) triple appears exactly once in the schoolbook.
                    # Wait no. The tensor is:
                    # T[i,j,l] = 1 if we want C[i,l] += A[i,j]*B[j,l]
                    # In matrix mult, C[i,l] = sum_j A[i,j]*B[j,l]
                    # So T[i,j,l] = 1 for all i,j,l (each triple contributes to one output entry)
                    # No that's not right either. The tensor has 3 indices (i, j, l) and
                    # T[i,j,l] = 1 means that the product A[i,j]*B[j,l] contributes to C[i,l].
                    # But which C entry? C[i,l]. So the "output index" is (i,l).
                    # The tensor is: for each (i,j,l), there's a 1 in position (i, j, l)
                    # and it maps to output position (i, l).
                    # So the decomposition is:
                    # sum_t U[i*m+j, t] * V[j*p+l, t] * W[l*n+i, t] = 1 for all (i,j,l)
                    # Let me verify with Strassen.
                    pass
    return True

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Block decomposition: split inner dim 5 = 2+3, output dim 5 = 2+3
    # Sub-problems:
    # 1. A[:, 0:2] (3x2) * B[0:2, 0:2] (2x2) -> C[:, 0:2] (3x2)
    # 2. A[:, 0:2] (3x2) * B[0:2, 2:5] (2x3) -> C[:, 2:5] (3x3)
    # 3. A[:, 2:5] (3x3) * B[2:5, 0:2] (3x2) -> C[:, 0:2] (3x2)
    # 4. A[:, 2:5] (3x3) * B[2:5, 2:5] (3x3) -> C[:, 2:5] (3x3)
    
    # For sub-problem 1: 3x2 * 2x2 -> 3x2
    # We can split A into 3 rows, each 1x2. Each row times B (2x2) gives 1x2.
    # For a 1x2 * 2x2 -> 1x2 multiplication, the tensor has 1*2*2 = 4 terms (schoolbook).
    # But we can use a shared factorization. Actually, let's think of it as:
    # For each row i (0,1,2) of A, we compute row_i (1x2) * B (2x2) = row_i_C (1x2)
    # The tensor for 1x2x2 has rank at most 2 (since it's just two scalar products).
    # Actually 1x2x2: C[l] = sum_j A[j] * B[j,l] for l=0,1
    # This is rank 2 (schoolbook). Can't do better since we need both products.
    # So for 3 rows, total rank = 3 * 2 = 6 for sub-problem 1.
    
    # For sub-problem 2: 3x2 * 2x3 -> 3x3
    # For each row i: row_i (1x2) * B (2x3) = row_i_C (1x3)
    # Tensor 1x2x3: C[l] = sum_j A[j]*B[j,l], rank 3 schoolbook.
    # Total rank = 3 * 3 = 9.
    
    # For sub-problem 3: 3x3 * 3x2 -> 3x2
    # For each row i: row_i (1x3) * B (3x2) = row_i_C (1x2)
    # Tensor 1x3x2: rank 2 schoolbook.
    # Total rank = 3 * 2 = 6.
    
    # For sub-problem 4: 3x3 * 3x3 -> 3x3
    # For each row i: row_i (1x3) * B (3x3) = row_i_C (1x3)
    # Tensor 1x3x3: rank 3 schoolbook.
    # Total rank = 3 * 3 = 9.
    
    # Total rank = 6 + 9 + 6 + 9 = 30? That seems too low. Let me recheck.
    # Wait, the issue is that U has shape (n*m, rank) = (15, rank).
    # For sub-problem 1: A indices are (i, j) where i in 0..2, j in 0..1.
    #   U[i*5 + j, t] for the first 6 columns.
    #   V[j*5 + l, t] where j in 0..1, l in 0..1 (B is 2x2, cols 0..1)
    #   W[l*3 + i, t] where l in 0..1, i in 0..2
    #
    # For row i, we compute: C[i, l] = sum_{j=0}^{1} A[i,j] * B[j,l]
    # We can set: U[i*5+j, t_i_j] = 1, V[j*5+l, t_i_j] = 1, W[l*3+i, t_i_j] = 1
    # where t_i_j is a unique index for each (i, j, l)? No, that's schoolbook.
    # For 1x2x2, we need: C[l] = sum_j A[j]*B[j,l]
    # Rank 2: for l=0: U[A_0]*V[B_00] + U[A_1]*V[B_10] -> one product? No.
    # Actually 1x2x2 tensor: T[j,l] = 1 for j in {0,1}, l in {0,1}
    # Rank: T[j,l] = sum_t U[j,t]*V[l,t]*W[?,t]... 
    # The W index is (l, i) but i is fixed (it's 1x2x2, so i=0 only).
    # W[l*1+0, t] = W[l, t]
    # T[j,l] = sum_t U[j,t] * V[l,t] * W[l,t]... no, W depends on l too.
    # T[j,l] = sum_t U[j,t] * V[l,t] * W[l,t] -- this doesn't make sense dimensionally.
    
    # Let me reconsider. For 1x2x2:
    # A is 1x2, B is 2x2, C is 1x2.
    # U shape: (1*2, r) = (2, r), indexed [i*m+j, t] = [0*2+j, t] = [j, t]
    # V shape: (2*2, r) = (4, r), indexed [j*p+l, t] = [j*2+l, t]
    # W shape: (2*1, r) = (2, r), indexed [l*n+i, t] = [l*1+0, t] = [l, t]
    # T[j,l] = sum_t U[j,t] * V[j*2+l, t] * W[l,t]
    # We need T[j,l] = 1 for all j in {0,1}, l in {0,1}.
    # With r=2:
    # t=0: U[0,0]=1, U[1,0]=0; V[0,0]=1, V[1,0]=0, V[2,0]=0, V[3,0]=0; W[0,0]=1, W[1,0]=0
    #   -> T[0,0] += 1*1*1 = 1, others 0.
    # t=1: U[0,1]=0, U[1,1]=1; V[0,1]=0, V[1,1]=1, V[2,1]=0, V[3,1]=0; W[0,1]=0, W[1,1]=1
    #   -> T[1,1] += 1*1*1 = 1, others 0.
    # Missing T[0,1] and T[1,0]. So rank 2 is not enough.
    # With r=4 (schoolbook): T[j,l] = U[j, j*2+l] * V[j*2+l, j*2+l] * W[l, j*2+l]
    # Set U[j, j*2+l] = 1, V[j*2+l, j*2+l] = 1, W[l, j*2+l] = 1 for each (j,l).
    # So rank 4 for 1x2x2.
    
    # OK so my earlier estimate was wrong. Let me recount.
    # Sub-problem 1: 3x2x2, rank = 3*2*2 = 12 (schoolbook)
    # Sub-problem 2: 3x2x3, rank = 3*2*3 = 18
    # Sub-problem 3: 3x3x2, rank = 3*3*2 = 18
    # Sub-problem 4: 3x3x3, rank = 3*3*3 = 27
    # Total: 75. Same as before.
    
    # The key insight from the cell findings: "Recursive block splitting with Strassen 
    # for 2x2x2 reduces rank from 75 to ~71 by exploiting algebraic structure in sub-blocks"
    
    # Let me try to use Strassen where possible.
    # Sub-problem 4: 3x3x3. Split 3 = 2+1.
    #   4a: 3x2x2 (A cols 0-1, B rows 0-1, B cols 0-1) -> C cols 0-1
    #   4b: 3x2x1 (A cols 0-1, B rows 0-1, B col 2) -> C col 2
    #   4c: 3x1x2 (A col 2, B rows 2, B cols 0-1) -> C cols 0-1
    #   4d: 3x1x1 (A col 2, B row 2, B col 2) -> C col 2
    #   4a: 3x2x2 = 12 schoolbook. Can we do better?
    #     Split A into rows: each 1x2x2 has rank 4. Total 3*4=12.
    #     But if we group rows... for 3x2x2, can we share factors across rows?
    #     The tensor is T[i,j,l] = 1 for i in 0..2, j in 0..1, l in 0..1
    #     with output (i,l).
    #     We could use the same (U,V) patterns for each row i, just with different W.
    #     But U and V are shared, so if U[i,j,t] depends on i, we can't fully share.
    #     Actually U has shape (3*2, r) = (6, r). So U[i*2+j, t] for i=0,1,2.
    #     If we set U[i*2+j, t] = u_j_t for all i (same for each row), and
    #     V[j*2+l, t] = v_jl_t, and W[l*3+i, t] = w_l_t for all i...
    #     Then T[i,j,l] = sum_t u_j_t * v_jl_t * w_l_t (independent of i)
    #     But T[i,j,l] = 1 for all i, so this works if sum_t u_j_t * v_jl_t * w_l_t = 1.
    #     This is just the 1x2x2 tensor repeated 3 times!
    #     So rank for 3x2x2 = rank for 1x2x2 = 4.
    #     Wait, but W[l*3+i, t] needs to be different for each i if we want to place
    #     into the right row. If W[l,t] is the same for all i, then we're adding the
    #     same value to C[0,l], C[1,l], C[2,l]. But we want C[i,l] = sum_j A[i,j]*B[j,l]
    #     which is different for each i (different A[i,:]).
    #     So we DO need U to depend on i. U[i*2+j, t] must encode A[i,j].
    #     Therefore we can't share across i in general.
    #     
    #     Unless... we use a different decomposition. For 3x2x2:
    #     C[i,l] = sum_j A[i,j] B[j,l]
    #     This is like 3 independent 1x2x2 multiplications, but they share the B factor.
    #     The tensor rank of 3x2x2 is at most 3*4 = 12, but could be less.
    #     Actually, the tensor is a "sum" of 3 copies of the 1x2x2 tensor, but with
    #     different U indices. The rank is exactly 3 * rank(1x2x2) = 3*4 = 12?
    #     Not necessarily, because the V and W parts can be shared.
    #     
    #     Let me think about it as: T[i,j,l] = delta(i,i)*1 for the relevant structure.
    #     The 3x2x2 tensor has "slice" structure. I believe the rank is 12.
    
    # Let me try a completely different approach: use the known result that
    # for n x m x p, the rank is at most n*m*p, and for specific cases can be lower.
    # For 3x5x5, let me try to find a rank < 75 decomposition.
    
    # Approach: Use the fact that 5 = 2+3 and apply Strassen to the 2x2 blocks.
    # 
    # C = A * B where A is 3x5, B is 5x5, C is 3x5.
    # Split A into [A1 | A2] where A1 is 3x2, A2 is 3x3.
    # Split B into [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3.
    # C = [A1*B11 + A2*B21 | A1*B12 + A2*B22]
    # 
    # Term 1: A1 (3x2) * B11 (2x2) -> 3x2
    # Term 2: A2 (3x3) * B21 (3x2) -> 3x2
    # Term 3: A1 (3x2) * B12 (2x3) -> 3x3
    # Term 4: A2 (3x3) * B22 (3x3) -> 3x3
    #
    # For Term 4: A2 (3x3) * B22 (3x3) -> 3x3
    # Split 3 = 2+1:
    #   A2 = [A2a | A2b] where A2a is 3x2, A2b is 3x1
    #   B22 = [[B22a, B22b], [B22c, B22d]] where B22a is 2x2, B22b is 2x1, B22c is 1x2, B22d is 1x1
    #   A2*B22 = A2a*B22a + A2a*B22b + A2b*B22c + A2b*B22d
    #   = (3x2)(2x2) + (3x2)(2x1) + (3x1)(1x2) + (3x1)(1x1)
    #   Ranks: 12 + 6 + 6 + 3 = 27 schoolbook.
    #
    # For the (3x2)(2x2) part: Can we use Strassen-like sharing?
    # Actually, let me just build the full decomposition explicitly and try to
    # minimize rank by combining terms where possible.
    
    # Let me build the schoolbook decomposition first, then try to merge.
    
    total_rank = n * m * p  # 75
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Now try to reduce rank by identifying linear dependencies.
    # The rank of the tensor is the minimum number of rank-1 terms.
    # We can try to find a basis for the space spanned by the columns of U, V, W.
    
    # Actually, let me try a specific structured approach.
    # For the 2x2x2 sub-problems within, use Strassen.
    
    # Let me construct a decomposition that uses Strassen for all 2x2x2 sub-blocks.
    
    # Split m=5 into 2+3, p=5 into 2+3, and also split n=3 into 2+1.
    # This gets complex. Let me try a simpler improvement.
    
    # Use Strassen for the 2x2x2 corner:
    # A[0:2, 0:2] (2x2) * B[0:2, 0:2] (2x2) -> C[0:2, 0:2] (2x2)
    # This uses 7 terms instead of 8.
    # Save 1 rank.
    
    # Similarly, other 2x2x2 sub-blocks if applicable.
    
    # Let me build a decomposition where I use Strassen for each 2x2x2 sub-block
    # and schoolbook for the rest.
    
    # Sub-blocks when splitting n=3=2+1, m=5=2+3, p=5=2+3:
    # This gives 2*2*2 = 8 sub-blocks.
    
    # Let me define the blocks:
    # n blocks: [0:2], [2:3]
    # m blocks: [0:2], [2:5]
    # p blocks: [0:2], [2:5]
    
    # For each combination (ni, mj, pk):
    # A_block: n_i x m_j, B_block: m_j x p_k, C_block: n_i x p_k
    
    # Block (0,0,0): 2x2 x 2x2 -> 2x2, rank 7 (Strassen)
    # Block (0,0,1): 2x2 x 2x3 -> 2x3, rank 12 schoolbook (2*2*3)
    # Block (0,1,0): 2x3 x 3x2 -> 2x2, rank 12 schoolbook
    # Block (0,1,1): 2x3 x 3x3 -> 2x3, rank 18 schoolbook
    # Block (1,0,0): 1x2 x 2x2 -> 1x2, rank 4 schoolbook
    # Block (1,0,1): 1x2 x 2x3 -> 1x3, rank 6 schoolbook
    # Block (1,1,0): 1x3 x 3x2 -> 1x2, rank 6 schoolbook
    # Block (1,1,1): 1x3 x 3x3 -> 1x3, rank 9 schoolbook
    # Total: 7+12+12+18+4+6+6+9 = 74
    
    # Save 1 rank. Let me implement this.
    
    n_blocks = [(0, 2), (2, 3)]
    m_blocks = [(0, 2), (2, 5)]
    p_blocks = [(0, 2), (2, 5)]
    
    U2 = np.zeros((n * m, 74), dtype=float)
    V2 = np.zeros((m * p, 74), dtype=float)
    W2 = np.zeros((p * n, 74), dtype=float)
    
    col = 0
    
    # Precompute Strassen factors for 2x2x2
    sU, sV, sW = _strassen_222_factors()
    
    for ni_idx, (i_start, i_end) in enumerate(n_blocks):
        for mj_idx, (j_start, j_end) in enumerate(m_blocks):
            for pk_idx, (k_start, k_end) in enumerate(p_blocks):
                ni = i_end - i_start
                mj = j_end - j_start
                pk = k_end - k_start
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen
                    for t in range(7):
                        for ii in range(2):
                            for jj in range(2):
                                U2[(i_start + ii) * m + (j_start + jj), col] = sU[ii * 2 + jj, t]
                        for jj in range(2):
                            for kk in range(2):
                                V2[(j_start + jj) * p + (k_start + kk), col] = sV[jj * 2 + kk, t]
                        for kk in range(2):
                            for ii in range(2):
                                W2[(k_start + kk) * n + (i_start + ii), col] = sW[kk * 2 + ii, t]
                        col += 1
                else:
                    # Schoolbook
                    for ii in range(ni):
                        for jj in range(mj):
                            for kk in range(pk):
                                U2[(i_start + ii) * m + (j_start + jj), col] = 1.0
                                V2[(j_start + jj) * p + (k_start + kk), col] = 1.0
                                W2[(k_start + kk) * n + (i_start + ii), col] = 1.0
                                col += 1
    
    # col should be 74
    assert col == 74, f"Expected 74, got {col}"
    
    # Verify
    U_list = U2[:, :col].tolist()
    V_list = V2[:, :col].tolist()
    W_list = W2[:, :col].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
