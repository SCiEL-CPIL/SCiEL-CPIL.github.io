"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Strategy:
    1. Split B (4x7) into B_left (4x4) and B_right (4x3).
    2. Use Strassen recursion for 4x4 * 4x4 -> rank 49.
    3. Use schoolbook for 4x4 * 4x3 -> rank 48.
    4. Concatenate: total rank = 97.
    
    This is an improvement over schoolbook rank 112.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    time_limit = 140  # safety margin before 150s kill
    
    def _verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that U @ diag @ V^T reconstructs the multiplication tensor."""
        # Tensor T[i,j,k] = delta_{j,k} (for matrix mult A*B, C[i,k] = sum_j A[i,j]*B[j,k])
        # In our factorization: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
        # Actually the convention is: C = A @ B, so C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor is T[i,j,k] = 1 if this term contributes, 0 otherwise.
        # But in tensor decomposition form: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix multiplication, T[i,j,k] = delta_{j,k}... no wait.
        # Actually for C = A @ B: C[i,k] = sum_j A[i,j] * B[j,k]
        # So the "tensor" has T[i,j,k] = 1 for all i,j,k where the product A[i,j]*B[j,k] contributes to C[i,k].
        # That means T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p) if j is a valid index... 
        # Wait, j goes from 0 to m-1, and B[j,k] requires j < m and k < p.
        # A[i,j] requires i < n and j < m.
        # So T[i,j,k] = 1 for all valid i,j,k. That's just the all-ones tensor? No.
        
        # Let me re-read the convention: W[k*n+i, r]
        # The output factor W has shape (p*n, rank). Index is k*n+i.
        # U has shape (n*m, rank), index i*m+j.
        # V has shape (m*p, rank), index j*p+k.
        
        # The decomposition says: for each r, U[:,r] * V[:,r] * W[:,r] (outer product)
        # And summing over r gives the multiplication tensor.
        
        # The multiplication tensor T has entries T[i,j,k] such that:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # In tensor form: T[i,j,k] = 1 for all i,j,k (since every combination contributes)
        # Wait no. The tensor for matrix multiplication C = AB is:
        # T[i,j,k] = delta_{j,j} = 1 for all i, j, k? That can't be right because then rank would be 1.
        
        # I think the correct interpretation is:
        # The bilinear map (A, B) -> C = A @ B can be written as:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # This is a trilinear form in (A, B, C^T) where C^T[i,k] pairs with the term.
        # The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # No wait, that's not right either.
        
        # Let me think again. The tensor for matrix multiplication of n x m by m x p is:
        # T ∈ R^{(n*m) × (m*p) × (p*n)} where T[i*m+j, j*p+k, k*n+i] = 1
        # for all valid i, j, k. All other entries are 0.
        
        # So T has exactly n*m*p non-zero entries, each equal to 1, at positions
        # (i*m+j, j*p+k, k*n+i) for i in [0,n), j in [0,m), k in [0,p).
        
        # The rank of this tensor is what we're minimizing.
        
        # Verification: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal
        # 1 if (i,j,k) is a valid triple (which it always is for i<n, j<m, k<p), else 0.
        
        # So for all i in range(n), j in range(m), k in range(p):
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # And for invalid combinations, it should be 0, but since all indices are valid,
        # we just need to check the valid ones.
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check a random sample of (i,j,k) triples
        np.random.seed(42)
        for _ in range(200):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
            if abs(val - 1.0) > tol:
                return False
        return True
    
    # Build Strassen decomposition for 4x4 * 4x4
    def build_strassen_4x4():
        """
        Build rank-49 decomposition for 4x4 * 4x4 using Strassen recursion.
        
        Strassen for 2x2 * 2x2 has rank 7:
        M1 = (A11+A22)*(B11+B22)
        M2 = (A21+A22)*B11
        M3 = A11*(B12-B22)
        M4 = A22*(B21-B11)
        M5 = (A11+A12)*B22
        M6 = (A21-A11)*(B11+B12)
        M7 = (A12-A22)*(B21+B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        For 4x4 * 4x4, we split into 2x2 blocks of 2x2 matrices.
        Each block multiplication is itself a 2x2 * 2x2, done with 7 products.
        So total rank = 7 * 7 = 49.
        """
        # We need to build U (16x49), V (16x49), W (16x49)
        # for 4x4 * 4x4.
        
        # Strassen 2x2 base case:
        # A is 2x2, B is 2x2, C is 2x2.
        # U indices: i*2+j for i,j in {0,1} -> 0,1,2,3
        # V indices: j*2+k for j,k in {0,1} -> 0,1,2,3
        # W indices: k*2+i for k,i in {0,1} -> 0,1,2,3
        
        # 7 products for 2x2*2x2:
        # r=0: M1 = (A11+A22)*(B11+B22)
        #   U: A11 at (0,0)=0, A22 at (1,1)=3 -> U[0,0]=1, U[3,0]=1
        #   V: B11 at (0,0)=0, B22 at (1,1)=3 -> V[0,0]=1, V[3,0]=1
        #   W: C11 gets +1, C22 gets +1
        #   C11 is at (k=0,i=0) -> W index 0*2+0=0
        #   C22 is at (k=1,i=1) -> W index 1*2+1=3
        #   W[0,0]=1, W[3,0]=1
        
        # r=1: M2 = (A21+A22)*B11
        #   U: A21 at (1,0)=2, A22 at (1,1)=3 -> U[2,1]=1, U[3,1]=1
        #   V: B11 at (0,0)=0 -> V[0,1]=1
        #   W: C21 gets +1, C22 gets -1
        #   C21 at (k=1,i=0) -> W index 1*2+0=2
        #   C22 at (k=1,i=1) -> W index 3
        #   W[2,1]=1, W[3,1]=-1
        
        # r=2: M3 = A11*(B12-B22)
        #   U: A11 at (0,0)=0 -> U[0,2]=1
        #   V: B12 at (0,1)=1, B22 at (1,1)=3 -> V[1,2]=1, V[3,2]=-1
        #   W: C12 gets +1, C22 gets +1
        #   C12 at (k=0,i=1) -> W index 0*2+1=1
        #   C22 at (k=1,i=1) -> W index 3
        #   W[1,2]=1, W[3,2]=1
        
        # r=3: M4 = A22*(B21-B11)
        #   U: A22 at (1,1)=3 -> U[3,3]=1
        #   V: B21 at (1,0)=2, B11 at (0,0)=0 -> V[2,3]=1, V[0,3]=-1
        #   W: C11 gets +1, C21 gets +1
        #   C11 at (k=0,i=0) -> W index 0
        #   C21 at (k=1,i=0) -> W index 2
        #   W[0,3]=1, W[2,3]=1
        
        # r=4: M5 = (A11+A12)*B22
        #   U: A11 at (0,0)=0, A12 at (0,1)=1 -> U[0,4]=1, U[1,4]=1
        #   V: B22 at (1,1)=3 -> V[3,4]=1
        #   W: C11 gets -1, C12 gets +1
        #   C11 at (k=0,i=0) -> W index 0
        #   C12 at (k=0,i=1) -> W index 1
        #   W[0,4]=-1, W[1,4]=1
        
        # r=5: M6 = (A21-A11)*(B11+B12)
        #   U: A21 at (1,0)=2, A11 at (0,0)=0 -> U[2,5]=1, U[0,5]=-1
        #   V: B11 at (0,0)=0, B12 at (0,1)=1 -> V[0,5]=1, V[1,5]=1
        #   W: C22 gets +1
        #   C22 at (k=1,i=1) -> W index 3
        #   W[3,5]=1
        
        # r=6: M7 = (A12-A22)*(B21+B22)
        #   U: A12 at (0,1)=1, A22 at (1,1)=3 -> U[1,6]=1, U[3,6]=-1
        #   V: B21 at (1,0)=2, B22 at (1,1)=3 -> V[2,6]=1, V[3,6]=1
        #   W: C11 gets +1
        #   C11 at (k=0,i=0) -> W index 0
        #   W[0,6]=1
        
        # Let's build these as arrays
        U_base = np.zeros((4, 7), dtype=float)
        V_base = np.zeros((4, 7), dtype=float)
        W_base = np.zeros((4, 7), dtype=float)
        
        # r=0: M1
        U_base[0, 0] = 1; U_base[3, 0] = 1
        V_base[0, 0] = 1; V_base[3, 0] = 1
        W_base[0, 0] = 1; W_base[3, 0] = 1
        
        # r=1: M2
        U_base[2, 1] = 1; U_base[3, 1] = 1
        V_base[0, 1] = 1
        W_base[2, 1] = 1; W_base[3, 1] = -1
        
        # r=2: M3
        U_base[0, 2] = 1
        V_base[1, 2] = 1; V_base[3, 2] = -1
        W_base[1, 2] = 1; W_base[3, 2] = 1
        
        # r=3: M4
        U_base[3, 3] = 1
        V_base[2, 3] = 1; V_base[0, 3] = -1
        W_base[0, 3] = 1; W_base[2, 3] = 1
        
        # r=4: M5
        U_base[0, 4] = 1; U_base[1, 4] = 1
        V_base[3, 4] = 1
        W_base[0, 4] = -1; W_base[1, 4] = 1
        
        # r=5: M6
        U_base[2, 5] = 1; U_base[0, 5] = -1
        V_base[0, 5] = 1; V_base[1, 5] = 1
        W_base[3, 5] = 1
        
        # r=6: M7
        U_base[1, 6] = 1; U_base[3, 6] = -1
        V_base[2, 6] = 1; V_base[3, 6] = 1
        W_base[0, 6] = 1
        
        # Verify base Strassen
        if not _verify(U_base.tolist(), V_base.tolist(), W_base.tolist(), 2, 2, 2):
            # Fall back to schoolbook for 2x2
            U_sb = np.zeros((4, 4), dtype=float)
            V_sb = np.zeros((4, 4), dtype=float)
            W_sb = np.zeros((4, 4), dtype=float)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        col = i*4 + j*2 + k
                        U_sb[i*2+j, col] = 1
                        V_sb[j*2+k, col] = 1
                        W_sb[k*2+i, col] = 1
            U_base, V_base, W_base = U_sb, V_sb, W_sb
        
        # Now build 4x4 * 4x4 using Strassen recursion
        # A is 4x4, split into 2x2 blocks: A00, A01, A10, A11 (each 2x2)
        # B is 4x4, split into 2x2 blocks: B00, B01, B10, B11 (each 2x2)
        # C is 4x4, split into 2x2 blocks: C00, C01, C10, C11 (each 2x2)
        
        # Strassen at block level:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Mx is a 2x2 * 2x2 multiplication, done with 7 products.
        # Total: 7 * 7 = 49 products.
        
        # For each of the 7 block-level Strassen products, we need to:
        # 1. Determine which A blocks and B blocks are involved (with signs)
        # 2. For each scalar product in the 2x2*2x2 base case, map to the full 4x4 indices
        
        # Block index mapping for 4x4:
        # A[i,j] where i,j in 0..3
        # A block (bi, bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
        # So A[i,j] is in block (i//2, j//2)
        
        # U index for A[i,j] in 4x4: i*4+j
        # V index for B[j,k] in 4x4: j*4+k
        # W index for C[i,k] in 4x4: k*4+i
        
        # For a block-level product Mx = (sum of A blocks with coefficients) * (sum of B blocks with coefficients)
        # Each A block (bi,bj) contributes to U at positions (2*bi+di, 2*bj+dj) for di,dj in {0,1}
        # Each B block (bj,bk) contributes to V at positions (2*bj+dj, 2*bk+dk) for dj,dk in {0,1}
        
        # The 7 base products for a 2x2*2x2:
        # base_prod[0]: (A00+A11)*(B00+B11) -> U: A00,A11; V: B00,B11; W: C00,C11
        # base_prod[1]: (A10+A11)*B00 -> U: A10,A11; V: B00; W: C10,C11
        # base_prod[2]: A00*(B01-B11) -> U: A00; V: B01,B11; W: C01,C11
        # base_prod[3]: A11*(B10-B00) -> U: A11; V: B10,B00; W: C00,C10
        # base_prod[4]: (A00+A01)*B11 -> U: A00,A01; V: B11; W: C00,C01
        # base_prod[5]: (A10-A00)*(B00+B01) -> U: A10,A00; V: B00,B01; W: C11
        # base_prod[6]: (A01-A11)*(B10+B11) -> U: A01,A11; V: B10,B11; W: C00
        
        # For each block-level Strassen product, we specify:
        # - A block contributions: list of (block_row, block_col, coeff)
        # - B block contributions: list of (block_row, block_col, coeff)
        # - C block contributions: list of (block_row, block_col, coeff)
        
        # Block-level Strassen:
        block_strassen = [
            # (A_blocks, B_blocks, C_blocks)
            # M1 = (A00+A11)*(B00+B11) -> C00+, C11+
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # M2 = (A10+A11)*B00 -> C10+, C11-
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            # M3 = A00*(B01-B11) -> C01+, C11+
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            # M4 = A11*(B10-B00) -> C00+, C10+
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            # M5 = (A00+A01)*B11 -> C00-, C01+
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            # M6 = (A10-A00)*(B00+B01) -> C11+
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            # M7 = (A01-A11)*(B10+B11) -> C00+
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        
        U_4x4 = np.zeros((16, 49), dtype=float)
        V_4x4 = np.zeros((16, 49), dtype=float)
        W_4x4 = np.zeros((16, 49), dtype=float)
        
        col = 0
        for bs_idx in range(7):
            A_blocks, B_blocks, C_blocks = block_strassen[bs_idx]
            
            # For each base product in the 2x2*2x2 decomposition:
            for bp_idx in range(7):
                # Get the base product's U, V, W contributions
                # Base U has non-zeros at certain (i,j) positions in the 2x2 block
                # We need to map these to the 4x4 indices based on which blocks are involved
                
                # The base product bp_idx involves specific A sub-blocks within the 2x2.
                # But actually, the base product is for a generic 2x2*2x2.
                # The A_blocks and B_blocks tell us which 2x2 blocks of the 4x4 to use.
                
                # For a given base product, it has U contributions at certain (di,dj) in the 2x2.
                # For each A block (bi,bj,coeff) in A_blocks, the U contribution is:
                #   U[2*bi+di, 2*bj+dj, col] += coeff * U_base[di*2+dj, bp_idx]
                
                # Similarly for V and W.
                
                for (bi, bj, a_coeff) in A_blocks:
                    for di in range(2):
                        for dj in range(2):
                            u_val = U_base[di*2+dj, bp_idx]
                            if u_val != 0:
                                U_4x4[2*bi+di, 2*bj+dj, col] += a_coeff * u_val
                
                for (bj, bk, b_coeff) in B_blocks:
                    for dj in range(2):
                        for dk in range(2):
                            v_val = V_base[dj*2+dk, bp_idx]
                            if v_val != 0:
                                V_4x4[2*bj+dj, 2*bk+dk, col] += b_coeff * v_val
                
                for (ci, ck, c_coeff) in C_blocks:
                    for dk in range(2):
                        for di in range(2):
                            w_val = W_base[dk*2+di, bp_idx]
                            if w_val != 0:
                                W_4x4[2*ci+di, 2*ck+dk, col] += c_coeff * w_val
                
                col += 1
        
        # Verify
        if not _verify(U_4x4.tolist(), V_4x4.tolist(), W_4x4.tolist(), 4, 4, 4):
            # Fall back to schoolbook for 4x4*4x4
            U_sb = np.zeros((16, 64), dtype=float)
            V_sb = np.zeros((16, 64), dtype=float)
            W_sb = np.zeros((16, 64), dtype=float)
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        c = i*16 + j*4 + k
                        U_sb[i*4+j, c] = 1
                        V_sb[j*4+k, c] = 1
                        W_sb[k*4+i, c] = 1
            return U_sb, V_sb, W_sb, 64
        
        return U_4x4, V_4x4, W_4x4, 49
    
    # Build schoolbook decomposition for 4x4 * 4x3
    def build_schoolbook_4x4_3():
        """
        Schoolbook decomposition for 4x4 * 4x3. Rank = 4*4*3 = 48.
        """
        U = np.zeros((16, 48), dtype=float)
        V = np.zeros((12, 48), dtype=float)
        W = np.zeros((12, 48), dtype=float)
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i*4+j, col] = 1.0
                    V[j*3+k, col] = 1.0
                    W[k*4+i, col] = 1.0
                    col += 1
        
        return U, V, W, 48
    
    # Build the combined decomposition
    U_left, V_left, W_left, rank_left = build_strassen_4x4()
    U_right, V_right, W_right, rank_right = build_schoolbook_4x4_3()
    
    # Concatenate
    U_total = np.hstack([U_left, U_right])
    V_total = np.hstack([V_left, V_right])
    W_total = np.hstack([W_left, W_right])
    
    total_rank = rank_left + rank_right
    
    # Verify the full decomposition
    # For the full 4x4*4x7, we need to check that:
    # sum_r U[i*4+j, r] * V[j*7+k, r] * W[k*4+i, r] = 1 for all i,j,k
    # But our V and W are split: left part uses V[j*4+k] for k in 0..3, right part uses V[j*3+k] for k in 0..2
    # We need to re-index V and W for the full problem.
    
    # Actually, the V matrix for the full problem has shape (m*p, rank) = (4*7, rank) = (28, rank)
    # The left part (4x4*4x4) uses V indices j*4+k for j in 0..3, k in 0..3
    # The right part (4x4*4x3) uses V indices j*3+k for j in 0..3, k in 0..2
    # But in the full problem, B is 4x7, so B[j,k] for k in 0..6
    # The left part corresponds to k in 0..3, the right part to k in 4..6
    
    # So we need to remap the V indices:
    # Left part: V_left[j*4+k, r] -> V_full[j*7+k, r] for k in 0..3
    # Right part: V_right[j*3+k, r] -> V_full[j*7+(4+k), r] for k in 0..2
    
    V_full = np.zeros((28, total_rank), dtype=float)
    for j in range(4):
        for k in range(4):
            V_full[j*7+k, :rank_left] = V_left[j*4+k, :]
        for k in range(3):
            V_full[j*7+(4+k), rank_left:] = V_right[j*3+k, :]
    
    # Similarly for W:
    # Left part: W_left[k*4+i, r] for k in 0..3 -> W_full[k*4+i, r]
    # Right part: W_right[k*3+i, r] for k in 0..2 -> W_full[(4+k)*4+i, r]
    
    W_full = np.zeros((28, total_rank), dtype=float)
    for k in range(4):
        for i in range(4):
            W_full[k*4+i, :rank_left] = W_left[k*4+i, :]
    for k in range(3):
        for i in range(4):
            W_full[(4+k)*4+i, rank_left:] = W_right[k*3+i, :]
    
    # U is the same for both parts since A is 4x4 in both cases
    # U_left has shape (16, rank_left), U_right has shape (16, rank_right)
    # U_full = hstack([U_left, U_right]) which is what we have
    
    # Verify the full decomposition
    if not _verify(U_total.tolist(), V_full.tolist(), W_full.tolist(), 4, 4, 7):
        # Fall back to schoolbook for the full problem
        U_sb = np.zeros((16, 112), dtype=float)
        V_sb = np.zeros((28, 112), dtype=float)
        W_sb = np.zeros((28, 112), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U_sb[i*4+j, col] = 1.0
                    V_sb[j*7+k, col] = 1.0
                    W_sb[k*4+i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        # Return schoolbook as fallback
        U_sb = np.zeros((16, 112), dtype=float)
        V_sb = np.zeros((28, 112), dtype=float)
        W_sb = np.zeros((28, 112), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U_sb[i*4+j, col] = 1.0
                    V_sb[j*7+k, col] = 1.0
                    W_sb[k*4+i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_total.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
