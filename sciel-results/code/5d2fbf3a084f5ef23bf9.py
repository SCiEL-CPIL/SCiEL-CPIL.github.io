import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 4, 4, 6

    # Strategy: partition the 4x4 input into two 2x4 horizontal strips.
    # Each strip is multiplied by the 4x6 matrix via a rank-6 decomposition
    # for 2x4x3 tensor, but adapted to 2x4x6 by splitting the output into
    # two 2x4x3 blocks (columns 0-2 and 3-5).
    #
    # For a 2x4x3 multiplication (A: 2x4, B: 4x3, C: 2x3), the naive rank is 24.
    # However, we can use a decomposition based on the identity:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    # This is inherently rank 2*4*3 = 24 for naive.
    #
    # Wait - the directive says "apply the known rank-6 decomposition for 2x4x3".
    # Rank 6 for 2x4x3 is NOT known to exist (the best known upper bound for 2x4x3 
    # is actually higher). Let me reconsider.
    #
    # Actually, for 2x2x2, Strassen gives rank 7 (not 6). For 2x3x3, the rank is 
    # known to be between 13 and 18. For 2x4x3, I don't think rank 6 is achievable.
    #
    # Let me re-read the directive: "apply the known rank-6 decomposition for 2x4x3
    # matrix multiplication to each strip to achieve a total rank of 12."
    #
    # This seems incorrect - rank 6 for 2x4x3 is not a known result. The naive rank
    # for 2x4x3 is 24. Even with Strassen-like methods, you can't get to 6.
    #
    # However, the directive is explicit. Let me think about what "rank-6 decomposition
    # for 2x4x3" could mean. Perhaps it refers to a different structure.
    #
    # Actually, wait. Maybe the directive is referring to a decomposition where we
    # exploit the fact that we're multiplying 2x4 by 4x6, and we can view this as
    # two independent 2x4 by 4x3 multiplications. For each 2x4x3 block, if we could
    # use rank 6, total would be 2 (strips) * 2 (blocks) * 6 = 24. That's not 12.
    #
    # Or: 2 strips * 6 per strip = 12 total. So each 2x4x6 multiplication uses rank 6?
    # That's even more aggressive and certainly not known to be possible.
    #
    # I think the directive contains an error about the existence of such a decomposition.
    # Let me instead use a valid improved decomposition.
    #
    # Valid approach: Use Strassen for 2x2 blocks.
    # 4x4 * 4x6 = (2x2 blocks of A) * (2x2 blocks of B, but B is 4x6 = 4x(3*2))
    #
    # Let me use a different valid strategy:
    # - Split A (4x4) into 4 quadrants of 2x2
    # - Split B (4x6) into top/bottom halves of 2x6 each
    # - Use Strassen to compute 2x2 * 2x6 products
    #
    # For 2x2 * 2x6: Write B as [B1 B2 B3] where each Bi is 2x2.
    # C = A * [B1 B2 B3] = [A*B1, A*B2, A*B3]
    # Each A*B_i is 2x2*2x2 = can use Strassen (rank 7)
    # Total: 3 * 7 = 21 per quadrant? No, that's not right either.
    #
    # Let me just use a clean valid decomposition that's better than 96.
    #
    # Simple improvement: Use the fact that C[i,k] = sum_j A[i,j]*B[j,k]
    # Group by j: for each j, we have outer product of column j of A (4x1) and
    # row j of B (1x6). That's still rank 16*6 = 96 for naive.
    #
    # Better: Use Strassen on 2x2 sub-blocks.
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B1], [B2]] where B1 is 2x6, B2 is 2x6
    # C = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    #
    # For each 2x2 * 2x6 product, split B into three 2x2 blocks:
    # 2x2 * 2x6 = 2x2 * [B_left, B_mid, B_right] = [P_left, P_mid, P_right]
    # Each Pi is 2x2*2x2, using Strassen rank 7.
    # So 2x2*2x6 uses 3*7 = 21 terms.
    #
    # We have 4 such products (A11*B1, A12*B2, A21*B1, A22*B2).
    # Total rank: 4 * 21 = 84.
    #
    # That's better than 96! Let me implement this.
    #
    # Actually, let me be more careful. Strassen for 2x2*2x2:
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
    # For 2x2 * 2x6, where B = [B_a, B_b, B_c] each 2x2:
    # Compute A*B_a using Strassen (7 terms), A*B_b (7 terms), A*B_c (7 terms).
    # Total 21 terms for one 2x2*2x6.
    #
    # For the full 4x4*4x6:
    # A = [[A11, A12], [A21, A22]], B = [[B1], [B2]] where B1,B2 are 2x6.
    # C_top = A11*B1 + A12*B2
    # C_bot = A21*B1 + A22*B2
    #
    # Each 2x2*2x6 takes 21 terms. We need 4 such products: A11*B1, A12*B2, A21*B1, A22*B2.
    # Total: 84 terms.
    #
    # But wait, we can be smarter. Let me define the decomposition explicitly.
    #
    # Let me index: A is 4x4, B is 4x6, C is 4x6.
    # Rows of A: 0,1,2,3. Cols of A: 0,1,2,3.
    # Rows of B: 0,1,2,3. Cols of B: 0,1,2,3,4,5.
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    #
    # Strassen decomposition for 4x4*4x6:
    # Split A into 2x2 blocks: A[0:2,0:2], A[0:2,2:4], A[2:4,0:2], A[2:4,2:4]
    # Split B into top 2x6 and bottom 2x6: B[0:2,:], B[2:4,:]
    #
    # C[0:2,:] = A[0:2,0:2]*B[0:2,:] + A[0:2,2:4]*B[2:4,:]
    # C[2:4,:] = A[2:4,0:2]*B[0:2,:] + A[2:4,2:4]*B[2:4,:]
    #
    # For each 2x2*2x6, use Strassen on each 2x2 column block.
    #
    # Let me implement this with rank 84.

    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84

    col = 0

    # Strassen helper: add a 2x2*2x2 Strassen term
    # A_sub is at rows (r0:r0+2), cols (c0:c0+2) of A
    # B_sub is at rows (r0_b:r0_b+2), cols (c0_b:c0_b+2) of B
    # C_sub is at rows (r0_c:r0_c+2), cols (c0_c:c0_c+2) of C
    def add_strassen_2x2(r0_a, c0_a, r0_b, c0_b, r0_c, c0_c, col_offset):
        nonlocal col
        # Strassen multipliers for 2x2*2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # A indices: (r0_a, c0_a) to (r0_a+1, c0_a+1)
        # A11 = A[r0_a, c0_a], A12 = A[r0_a, c0_a+1]
        # A21 = A[r0_a+1, c0_a], A22 = A[r0_a+1, c0_a+1]
        
        # B indices: (r0_b, c0_b) to (r0_b+1, c0_b+1)
        # B11 = B[r0_b, c0_b], B12 = B[r0_b, c0_b+1]
        # B21 = B[r0_b+1, c0_b], B22 = B[r0_b+1, c0_b+1]
        
        # C indices: (r0_c, c0_c) to (r0_c+1, c0_c+1)
        # C11 = C[r0_c, c0_c], C12 = C[r0_c, c0_c+1]
        # C21 = C[r0_c+1, c0_c], C22 = C[r0_c+1, c0_c+1]
        
        A11 = (r0_a, c0_a)
        A12 = (r0_a, c0_a + 1)
        A21 = (r0_a + 1, c0_a)
        A22 = (r0_a + 1, c0_a + 1)
        
        B11 = (r0_b, c0_b)
        B12 = (r0_b, c0_b + 1)
        B21 = (r0_b + 1, c0_b)
        B22 = (r0_b + 1, c0_b + 1)
        
        C11 = (r0_c, c0_c)
        C12 = (r0_c, c0_c + 1)
        C21 = (r0_c + 1, c0_c)
        C22 = (r0_c + 1, c0_c + 1)
        
        def set_uv(u_idx, u_val, v_idx, v_val):
            nonlocal col
            U[u_idx][col] = u_val
            V[v_idx][col] = v_val
        
        # M1 = (A11+A22)(B11+B22)
        set_uv(A11[0]*m + A11[1], 1, B11[0]*p + B11[1], 1)
        set_uv(A22[0]*m + A22[1], 1, B11[0]*p + B11[1], 1)
        # Need to combine: (A11+A22) means U has 1 at both A11 and A22 positions
        # But in our format, each rank-1 term has ONE U entry and ONE V entry.
        # So (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # That's 4 terms, not 1!
        #
        # I made an error. In the tensor decomposition format, each rank-1 term is
        # of the form u * v^T * w where u is a vector in R^{nm}, v in R^{mp}, w in R^{pn}.
        # A "product" like (A11+A22)(B11+B22) cannot be a single rank-1 term unless
        # we expand it.
        #
        # So Strassen's 7 multiplications are 7 rank-1 terms ONLY if each multiplier
        # is a single element. But (A11+A22) is a sum, so it's not a single rank-1 term.
        #
        # The correct way: each of Strassen's 7 terms is itself a rank-1 term where
        # U has entries at multiple positions (the sum), V has entries at multiple positions.
        #
        # Yes! That's the key. U and V can have multiple non-zero entries per column.
        # So for M1 = (A11+A22)(B11+B22):
        # U column has 1 at A11 position and 1 at A22 position
        # V column has 1 at B11 position and 1 at B22 position
        # W column has the appropriate C entries
        #
        # Let me redo this.
        
        # Reset col - I'll restructure
        pass

    # Let me restart with a cleaner approach
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_term(u_entries, v_entries, w_entries, col_idx):
        """Add one rank-1 term. u_entries, v_entries, w_entries are lists of (index, value)."""
        for idx, val in u_entries:
            U[idx][col_idx] += val
        for idx, val in v_entries:
            V[idx][col_idx] += val
        for idx, val in w_entries:
            W[idx][col_idx] += val

    def a_idx(r, c):
        return r * m + c

    def b_idx(r, c):
        return r * p + c

    def c_idx(r, c):
        return c * n + r  # W[k*n+i] where i=row, k=col

    # Strassen for 2x2 * 2x2 at A[r0:r0+2, c0:c0+2] * B[rb0:rb0+2, cb0:cb0+2] -> C[rc0:rc0+2, cc0:cc0+2]
    def add_strassen_2x2x2(r0, c0, rb0, cb0, rc0, cc0, c):
        # A11=A[r0,c0], A12=A[r0,c0+1], A21=A[r0+1,c0], A22=A[r0+1,c0+1]
        # B11=B[rb0,cb0], B12=B[rb0,cb0+1], B21=B[rb0+1,cb0], B22=B[rb0+1,cb0+1]
        # C11=C[rc0,cc0], C12=C[rc0,cc0+1], C21=C[rc0+1,cc0], C22=C[rc0+1,cc0+1]
        
        A11 = a_idx(r0, c0)
        A12 = a_idx(r0, c0 + 1)
        A21 = a_idx(r0 + 1, c0)
        A22 = a_idx(r0 + 1, c0 + 1)
        
        B11 = b_idx(rb0, cb0)
        B12 = b_idx(rb0, cb0 + 1)
        B21 = b_idx(rb0 + 1, cb0)
        B22 = b_idx(rb0 + 1, cb0 + 1)
        
        C11 = c_idx(rc0, cc0)
        C12 = c_idx(rc0, cc0 + 1)
        C21 = c_idx(rc0 + 1, cc0)
        C22 = c_idx(rc0 + 1, cc0 + 1)
        
        # M1 = (A11+A22)(B11+B22)
        add_term([(A11, 1), (A22, 1)], [(B11, 1), (B22, 1)],
                 [(C11, 1), (C22, 1)], c)
        c += 1
        
        # M2 = (A21+A22)B11
        add_term([(A21, 1), (A22, 1)], [(B11, 1)],
                 [(C21, 1)], c)
        c += 1
        
        # M3 = A11(B12-B22)
        add_term([(A11, 1)], [(B12, 1), (B22, -1)],
                 [(C12, 1)], c)
        c += 1
        
        # M4 = A22(B21-B11)
        add_term([(A22, 1)], [(B21, 1), (B11, -1)],
                 [(C11, -1), (C21, 1)], c)
        c += 1
        
        # M5 = (A11+A12)B22
        add_term([(A11, 1), (A12, 1)], [(B22, 1)],
                 [(C11, -1), (C12, 1)], c)
        c += 1
        
        # M6 = (A21-A11)(B11+B12)
        add_term([(A21, 1), (A11, -1)], [(B11, 1), (B12, 1)],
                 [(C22, 1)], c)
        c += 1
        
        # M7 = (A12-A22)(B21+B22)
        add_term([(A12, 1), (A22, -1)], [(B21, 1), (B22, 1)],
                 [(C11, 1), (C22, -1)], c)
        c += 1
        
        return c

    # For 2x2 * 2x6: split B into three 2x2 blocks
    def add_strassen_2x2x6(r0, c0, rb0, cb0, rc0, cc0, c):
        # B has 6 columns: cb0, cb0+1, cb0+2, cb0+3, cb0+4, cb0+5
        # Split into [cb0:cb0+2], [cb0+2:cb0+4], [cb0+4:cb0+6]
        c = add_strassen_2x2x2(r0, c0, rb0, cb0, rc0, cc0, c)
        c = add_strassen_2x2x2(r0, c0, rb0, cb0 + 2, rc0, cc0 + 2, c)
        c = add_strassen_2x2x2(r0, c0, rb0, cb0 + 4, rc0, cc0 + 4, c)
        return c

    # C[0:2,:] = A[0:2,0:2]*B[0:2,:] + A[0:2,2:4]*B[2:4,:]
    # C[2:4,:] = A[2:4,0:2]*B[0:2,:] + A[2:4,2:4]*B[2:4,:]
    
    # A[0:2,0:2] * B[0:2,:] -> C[0:2,:]
    col = add_strassen_2x2x6(0, 0, 0, 0, 0, 0, col)
    
    # A[0:2,2:4] * B[2:4,:] -> C[0:2,:]
    col = add_strassen_2x2x6(0, 2, 2, 0, 0, 0, col)
    
    # A[2:4,0:2] * B[0:2,:] -> C[2:4,:]
    col = add_strassen_2x2x6(2, 0, 0, 0, 2, 0, col)
    
    # A[2:4,2:4] * B[2:4,:] -> C[2:4,:]
    col = add_strassen_2x2x6(2, 2, 2, 0, 2, 0, col)
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 6)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((4, 6))
    for r in range(rank):
        alpha_r = np.dot(U_np[:, r], A_test.flatten())
        beta_r = np.dot(V_np[:, r], B_test.flatten())
        C_computed += alpha_r * beta_r * W_np[:, r].reshape(6, 4).T
    
    error = np.max(np.abs(C_computed - C_expected))
    
    if error > 1e-10:
        # Fall back to naive if verification fails
        rank_naive = n * m * p
        U = [[0] * rank_naive for _ in range(n * m)]
        V = [[0] * rank_naive for _ in range(m * p)]
        W = [[0] * rank_naive for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
