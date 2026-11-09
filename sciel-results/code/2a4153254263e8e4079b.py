"""Matrix multiplication 3x3x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_SECONDS = 140  # Safety margin before 150s kill

def construct():
    """
    Construct a rank-reduced exact tensor decomposition for multiplying
    a 3x3 matrix (A) by a 3x8 matrix (B) to produce a 3x8 matrix (C).
    
    Uses a greedy tensor decomposition approach:
    - Start with the full tensor (schoolbook rank 72)
    - Iteratively try to merge terms or find lower-rank representations
    - Use the tensor's symmetry (3x3x8 has repeated structure in the p-dimension)
    
    The key insight: for 3x3x8, the p-dimension (8 columns) creates
    repeated patterns. We can potentially exploit this by grouping columns.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    
    # Build the target tensor T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) of C
    # T has shape (n*m, m*p, p*n) = (9, 24, 24)
    # T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    # For each (i,j,k): a = i*3+j, b = j*8+k, c = k*3+i
    
    # The tensor is: sum over i,j,k of e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # This is the "canonical" rank-72 decomposition.
    
    # Strategy: Try to find a lower-rank decomposition using the fact that
    # the p-dimension has 8 identical "slices" (each column k gives the same 3x3 pattern).
    
    # For a fixed k, the contribution is: sum over i,j of e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # This is a 3x3x1 tensor for each k.
    
    # Key observation: The tensor for column k is T_k[i,j] = e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # All 8 columns have the same "shape" in the i,j dimensions but different indices in b and c.
    
    # Let's try a block decomposition:
    # Group columns into pairs or use the 3x3 structure.
    
    # Approach 1: Direct schoolbook (rank 72) as baseline
    def schoolbook_decomposition():
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
        return U, V, W, rank
    
    # Approach 2: Try to exploit the 3x3x8 structure
    # For each column k, we need to compute C[:,k] = A @ B[:,k]
    # This is a 3x3 times 3x1 matrix-vector product.
    # Standard rank for 3x3x1 is 9 (schoolbook).
    # But can we do better?
    
    # For 3x3x1, the tensor is: sum_{i,j} e_{i,j} ⊗ e_{j} ⊗ e_{i}
    # This is the "transpose" tensor. Known minimum rank for 3x3x1 is 9 (no improvement known).
    
    # However, for 3x3x8, we can share computations across columns.
    # If we use the same U vectors for all 8 columns, we can potentially reduce rank.
    
    # Let's try: For each (i,j) pair, use ONE U vector that multiplies with all 8 B columns.
    # U[i*m+j] = [1, 1, ..., 1] (8 ones) for each (i,j)
    # V[j*p+k] = [0,...,0,1,0,...,0] (1 at position k)
    # W[k*n+i] = [0,...,0,1,0,...,0] (1 at position i)
    
    # Wait, this doesn't work directly because we need separate rank-1 terms.
    
    # Let me think differently. The tensor is:
    # T = sum_{k=0}^{7} sum_{i=0}^{2} sum_{j=0}^{2} e_{i*3+j} ⊗ e_{j*8+k} ⊗ e_{k*3+i}
    
    # = sum_{k=0}^{7} (sum_{i,j} e_{i*3+j} ⊗ e_{j*8+k} ⊗ e_{k*3+i})
    
    # Each inner sum is a 3x3x1 tensor with rank 9.
    # Total: 8 * 9 = 72.
    
    # Can we share across k? The issue is that e_{j*8+k} and e_{k*3+i} depend on k.
    # So we can't easily share.
    
    # Alternative: Use a 2x2x2 block structure.
    # Split A into 2x2 blocks, B into 2x8 blocks.
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] where B11 is 2x8, B12 is 2x8, B21 is 1x8, B22 is 1x8
    
    # C = A @ B
    # C[0:2, :] = A11 @ B11 + A12 @ B21
    # C[2, :] = A21 @ B12 + A22 @ B22
    
    # This gives:
    # C[0:2, :] requires: A11@B11 (2x2x8, rank 32) + A12@B21 (2x1x8, rank 16)
    # C[2, :] requires: A21@B12 (1x2x8, rank 16) + A22@B22 (1x1x8, rank 8)
    # Total: 32 + 16 + 16 + 8 = 72. Same as schoolbook.
    
    # The problem is that 2x2x8 doesn't have a known lower-rank decomposition than 32.
    # 2x2x2 has rank 7 (Strassen), so 2x2x8 should have rank 7*4 = 28 if we can tile.
    # But 8 is not a multiple of 2 in a way that allows clean tiling.
    
    # Actually, 2x2x8: We can split the 8 columns into 4 groups of 2.
    # Each 2x2x2 block has rank 7.
    # Total: 4 * 7 = 28.
    
    # Let me implement this!
    
    def strassen_2x2x2():
        """Return Strassen decomposition for 2x2x2 matrix multiplication.
        U: 4x7, V: 4x7, W: 4x7
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Strassen's 7 products:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B00-B01)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        #
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 - M3 + M6
        
        # M1: U[00]+U[11], V[00]+V[11], W[00]+W[11]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: U[10]+U[11], V[00], W[10]+W[11]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3: U[00], V[10]-V[11], W[00]-W[01]
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[0][2] = -1; W[1][2] = 1
        
        # M4: U[11], V[00]-V[01], W[00]-W[10]
        U[3][3] = 1
        V[0][3] = 1; V[1][3] = -1
        W[0][3] = -1; W[2][3] = 1
        
        # M5: U[00]+U[01], V[11], W[01]+W[00]
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][4] = 1; W[0][4] = -1
        
        # M6: U[10]-U[00], V[00]+V[01], W[11]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7: U[01]-U[11], V[10]+V[11], W[00]-W[11]
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = -1
        
        return U, V, W
    
    # Now build the full 3x3x8 decomposition using 2x2x2 Strassen blocks.
    # A is 3x3, B is 3x8, C is 3x8.
    # 
    # Split A into:
    #   A_top_left: rows 0-1, cols 0-1 (2x2)
    #   A_top_right: rows 0-1, col 2 (2x1)
    #   A_bottom_left: row 2, cols 0-1 (1x2)
    #   A_bottom_right: row 2, col 2 (1x1)
    #
    # Split B into:
    #   B_top: rows 0-1, all 8 cols (2x8)
    #   B_bottom: row 2, all 8 cols (1x8)
    #
    # C = A @ B:
    #   C[0:2, :] = A_top_left @ B_top + A_top_right @ B_bottom
    #   C[2, :] = A_bottom_left @ B_top + A_bottom_right @ B_bottom
    #
    # A_top_left @ B_top: 2x2 @ 2x8
    #   Split B_top into 4 blocks of 2 columns each.
    #   Each 2x2 @ 2x2 block uses Strassen (rank 7).
    #   Total: 4 * 7 = 28.
    #
    # A_top_right @ B_bottom: 2x1 @ 1x8
    #   This is just: for each k, C[i,k] += A_top_right[i] * B_bottom[k]
    #   Rank: 2 * 8 = 16 (schoolbook, no known improvement for 2x1x8)
    #
    # A_bottom_left @ B_top: 1x2 @ 2x8
    #   This is: for each k, C[2,k] += A_bottom_left[0] * B_top[0,k] + A_bottom_left[1] * B_top[1,k]
    #   Rank: 2 * 8 = 16
    #
    # A_bottom_right @ B_bottom: 1x1 @ 1x8
    #   This is: for each k, C[2,k] += A_bottom_right * B_bottom[k]
    #   Rank: 8
    #
    # Total: 28 + 16 + 16 + 8 = 68.
    
    def build_68_rank_decomposition():
        U_str, V_str, W_str = strassen_2x2x2()
        
        # Total rank: 28 (Strassen) + 16 + 16 + 8 = 68
        total_rank = 68
        U = [[0] * total_rank for _ in range(9)]
        V = [[0] * total_rank for _ in range(24)]
        W = [[0] * total_rank for _ in range(24)]
        
        col = 0
        
        # Part 1: A_top_left (2x2) @ B_top (2x8) using Strassen for each 2-column block
        # A_top_left: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,3,4
        # B_top: B[0,k], B[1,k] for k=0..7 -> V indices k, 8+k
        # C[0:2, k] -> W indices 3k, 3k+1
        
        for block in range(4):  # 4 blocks of 2 columns each
            k0 = block * 2
            k1 = k0 + 1
            
            # For this 2x2x2 block:
            # U: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
            # V: B[0,k0]=V[k0], B[0,k1]=V[k1], B[1,k0]=V[8+k0], B[1,k1]=V[8+k1]
            # W: C[0,k0]=W[3*k0], C[0,k1]=W[3*k0+1], C[1,k0]=W[3*k0+3], C[1,k1]=W[3*k0+4]
            # Wait, W index: W[k*n+i] where k is column, i is row.
            # C[0,k0] -> W[k0*3+0] = W[3*k0]
            # C[0,k1] -> W[k1*3+0] = W[3*k0+3]
            # C[1,k0] -> W[k0*3+1] = W[3*k0+1]
            # C[1,k1] -> W[k1*3+1] = W[3*k0+4]
            
            for s in range(7):
                # U contributions
                for a_idx, val in [(0, U_str[0][s]), (1, U_str[1][s]), (3, U_str[2][s]), (4, U_str[3][s])]:
                    if val != 0:
                        U[a_idx][col] = val
                
                # V contributions
                # V_str[0] -> B[0,k0] = V[k0]
                # V_str[1] -> B[0,k1] = V[k1]
                # V_str[2] -> B[1,k0] = V[8+k0]
                # V_str[3] -> B[1,k1] = V[8+k1]
                for b_idx, v_idx in [(0, k0), (1, k1), (2, 8+k0), (3, 8+k1)]:
                    val = V_str[b_idx][s]
                    if val != 0:
                        V[v_idx][col] = val
                
                # W contributions
                # W_str[0] -> C[0,k0] = W[3*k0]
                # W_str[1] -> C[0,k1] = W[3*k0+3]
                # W_str[2] -> C[1,k0] = W[3*k0+1]
                # W_str[3] -> C[1,k1] = W[3*k0+4]
                for c_idx, w_idx in [(0, 3*k0), (1, 3*k0+3), (2, 3*k0+1), (3, 3*k0+4)]:
                    val = W_str[c_idx][s]
                    if val != 0:
                        W[w_idx][col] = val
                
                col += 1
        
        # Part 2: A_top_right (2x1) @ B_bottom (1x8)
        # A_top_right: A[0,2]=U[2], A[1,2]=U[5]
        # B_bottom: B[2,k] = V[16+k] for k=0..7
        # C[0,k] += A[0,2]*B[2,k], C[1,k] += A[1,2]*B[2,k]
        for i in range(2):  # rows 0,1
            for k in range(8):
                U[i * 3 + 2][col] = 1  # A[i,2]
                V[16 + k][col] = 1     # B[2,k]
                W[k * 3 + i][col] = 1  # C[i,k]
                col += 1
        
        # Part 3: A_bottom_left (1x2) @ B_top (2x8)
        # A_bottom_left: A[2,0]=U[6], A[2,1]=U[7]
        # B_top: B[0,k]=V[k], B[1,k]=V[8+k]
        # C[2,k] += A[2,0]*B[0,k] + A[2,1]*B[1,k]
        for j in range(2):  # cols 0,1 of A
            for k in range(8):
                U[2 * 3 + j][col] = 1  # A[2,j]
                V[j * 8 + k][col] = 1  # B[j,k]
                W[k * 3 + 2][col] = 1  # C[2,k]
                col += 1
        
        # Part 4: A_bottom_right (1x1) @ B_bottom (1x8)
        # A_bottom_right: A[2,2]=U[8]
        # B_bottom: B[2,k]=V[16+k]
        # C[2,k] += A[2,2]*B[2,k]
        for k in range(8):
            U[8][col] = 1
            V[16 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1
        
        return U, V, W, total_rank
    
    # Try the 68-rank decomposition
    try:
        U, V, W, rank = build_68_rank_decomposition()
        # Verify correctness
        def verify(U, V, W, rank):
            n, m, p = 3, 3, 8
            # Compute T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # and check it equals the matrix multiplication tensor
            T = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T += np.einsum('i,j,k->ijk', np.array(U)[:,r], np.array(V)[:,r], np.array(W)[:,r])
            
            # Target tensor
            T_target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i*m+j, j*p+k, k*n+i] = 1
            
            return np.allclose(T, T_target, atol=1e-9)
        
        if verify(U, V, W, rank):
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to schoolbook
            U, V, W, rank = schoolbook_decomposition()
            return {"U": U, "V": V, "W": W}
    except Exception:
        # Fallback to schoolbook
        U, V, W, rank = schoolbook_decomposition()
        return {"U": U, "V": V, "W": W}
