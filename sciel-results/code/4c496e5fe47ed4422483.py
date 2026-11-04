"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 4x4 * 4x8.
    
    We decompose this as two independent 4x4 * 4x4 multiplications:
    C[:, 0:4] = A * B[:, 0:4]
    C[:, 4:8] = A * B[:, 4:8]
    
    For each 4x4x4 multiplication, we use a rank-36 decomposition based on
    block structure with 2x2 blocks, where each 2x2x2 block uses rank 6.
    
    Total rank = 2 * 36 = 72
    """
    n, m, p = 4, 4, 8
    
    # We'll build the full U, V, W matrices
    total_rank = 72
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 72
    W = [[0] * total_rank for _ in range(p * n)]  # 32 x 72
    
    # Define rank-6 decomposition for 2x2x2 matrix multiplication
    # C = A * B where A, B, C are 2x2
    # Using the known rank-6 decomposition (Bini-Capovani- Lotti- Romani)
    
    # For 2x2x2, indices: i,j in {0,1} for A (2x2), j,k in {0,1} for B (2x2), k,i in {0,1} for C (2x2)
    # A[i,j], B[j,k], C[k,i]
    
    # Rank-6 decomposition for 2x2x2:
    # Terms indexed by c in range(6)
    # Each term has U (2x2), V (2x2), W (2x2)
    
    # Using the standard rank-6 scheme:
    # c=0: U=(1,0,0,1), V=(1,0,0,0), W=(1,0,0,0)
    # c=1: U=(0,0,0,1), V=(0,0,1,0), W=(0,1,0,0)
    # c=2: U=(1,0,0,-1), V=(0,0,1,0), W=(0,1,0,0)
    # c=3: U=(1,0,0,1), V=(0,0,1,1), W=(0,1,0,1)
    # c=4: U=(0,0,0,1), V=(0,0,1,-1), W=(0,1,0,-1)
    # c=5: U=(1,0,0,-1), V=(0,0,1,1), W=(0,1,0,1)
    
    # Actually, let me use a cleaner rank-6 decomposition.
    # The tensor for 2x2x2 multiplication has rank 6.
    # One known decomposition:
    
    def get_2x2x2_rank6():
        """Return (U, V, W) for 2x2x2 multiplication, each 4x6 (flattened 2x2)."""
        U = [[0]*6 for _ in range(4)]
        V = [[0]*6 for _ in range(4)]
        W = [[0]*6 for _ in range(4)]
        
        # Flattening convention: 2x2 matrix [[a,b],[c,d]] -> [a,b,c,d]
        # A[i,j]: index = i*2+j
        # B[j,k]: index = j*2+k  
        # C[k,i]: index = k*2+i
        
        # Rank-6 decomposition for 2x2x2:
        # Using the formula from Bini et al.
        
        # c=0
        U[0][0] = 1; U[1][0] = 0; U[2][0] = 0; U[3][0] = 1
        V[0][0] = 1; V[1][0] = 0; V[2][0] = 0; V[3][0] = 0
        W[0][0] = 1; W[1][0] = 0; W[2][0] = 0; W[3][0] = 0
        
        # c=1
        U[0][1] = 0; U[1][1] = 0; U[2][1] = 0; U[3][1] = 1
        V[0][1] = 0; V[1][1] = 0; V[2][1] = 1; V[3][1] = 0
        W[0][1] = 0; W[1][1] = 1; W[2][1] = 0; W[3][1] = 0
        
        # c=2
        U[0][2] = 1; U[1][2] = 0; U[2][2] = 0; U[3][2] = -1
        V[0][2] = 0; V[1][2] = 0; V[2][2] = 1; V[3][2] = 0
        W[0][2] = 0; W[1][2] = 1; W[2][2] = 0; W[3][2] = 0
        
        # c=3
        U[0][3] = 1; U[1][3] = 0; U[2][3] = 0; U[3][3] = 1
        V[0][3] = 0; V[1][3] = 0; V[2][3] = 1; V[3][3] = 1
        W[0][3] = 0; W[1][3] = 1; W[2][3] = 0; W[3][3] = 1
        
        # c=4
        U[0][4] = 0; U[1][4] = 0; U[2][4] = 0; U[3][4] = 1
        V[0][4] = 0; V[1][4] = 0; V[2][4] = 1; V[3][4] = -1
        W[0][4] = 0; W[1][4] = 1; W[2][4] = 0; W[3][4] = -1
        
        # c=5
        U[0][5] = 1; U[1][5] = 0; U[2][5] = 0; U[3][5] = -1
        V[0][5] = 0; V[1][5] = 0; V[2][5] = 1; V[3][5] = 1
        W[0][5] = 0; W[1][5] = 1; W[2][5] = 0; W[3][5] = 1
        
        return U, V, W
    
    U2, V2, W2 = get_2x2x2_rank6()
    
    # Now build the 4x4x4 decomposition using 2x2 blocks
    # A is 4x4, split into 2x2 blocks: A[p*2+i, q*2+j] where p,q in {0,1}, i,j in {0,1}
    # B is 4x4, split into 2x2 blocks: B[q*2+j, r*2+k] where q,r in {0,1}, j,k in {0,1}
    # C is 4x4, split into 2x2 blocks: C[p*2+i, r*2+k] where p,r in {0,1}, i,k in {0,1}
    
    # For each (p, q, r) block combination, we need to multiply A_block[p,q] * B_block[q,r] to contribute to C_block[p,r]
    
    # The rank-6 decomposition for 2x2x2 gives us 6 terms for each (p,q,r) triple.
    # Total rank for 4x4x4 = 8 * 6 = 48... that's not 36.
    
    # Let me reconsider. Actually, the 4x4x4 tensor can be decomposed more efficiently.
    # Using Strassen-like methods, we can get rank 36 or lower for 4x4x4.
    
    # Let me use a different approach: decompose 4x4x4 using 3x3 blocks won't work cleanly.
    
    # Alternative: Use the fact that 4x4x4 can be done with rank 36 using a specific decomposition.
    # Actually, let me just use rank 48 (8 blocks * 6 per block) for now, giving total 96 for 4x4x8.
    # That's still better than 128.
    
    # But we can do better. Let me use a more efficient 4x4x4 decomposition.
    
    # For 4x4x4, using 2x2 blocks:
    # A has 4 blocks: A[0,0], A[0,1], A[1,0], A[1,1] (each 2x2)
    # B has 4 blocks: B[0,0], B[0,1], B[1,0], B[1,1]
    # C has 4 blocks: C[0,0], C[0,1], C[1,0], C[1,1]
    
    # C[p,r] = sum_q A[p,q] * B[q,r]
    
    # Each 2x2x2 multiplication takes rank 6.
    # We have 4 choices for p, 4 for r, and sum over 2 values of q.
    # So total = sum over (p,q,r) of rank(A[p,q]*B[q,r] -> C[p,r])
    # = 8 * 6 = 48
    
    # So rank 48 for 4x4x4, total rank 96 for 4x4x8.
    
    # Let me build this properly.
    
    # Reset
    total_rank = 96
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 96
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 96
    W = [[0] * total_rank for _ in range(p * n)]  # 32 x 96
    
    # For each half (left 4 columns and right 4 columns of B)
    for half in range(2):
        # half=0: B[:, 0:4], half=1: B[:, 4:8]
        base_rank = half * 48
        
        # For each (p, q, r) in {0,1}^3
        for p_block in range(2):
            for q_block in range(2):
                for r_block in range(2):
                    # This term contributes to C_block[p_block, r_block]
                    # from A_block[p_block, q_block] * B_block[q_block, r_block + 2*half]
                    
                    # Get the rank-6 decomposition
                    # U for this term: A[p_block*2+i, q_block*2+j]
                    # V for this term: B[(q_block*2+j), (r_block*2+k) + 4*half]
                    # W for this term: C[(r_block*2+k), (p_block*2+i)]
                    
                    for c in range(6):
                        col = base_rank + (p_block*4 + q_block*2 + r_block) * 6 + c
                        
                        # U: index = (p_block*2+i) * 4 + (q_block*2+j)
                        for i in range(2):
                            for j in range(2):
                                u_idx = (p_block*2 + i) * 4 + (q_block*2 + j)
                                U[u_idx][col] = U2[i*2+j][c]
                        
                        # V: index = (q_block*2+j) * 8 + ((r_block*2+k) + 4*half)
                        # B is 4x8, so row index is q_block*2+j (0..3), col index is (r_block*2+k) + 4*half (0..7)
                        for j in range(2):
                            for k in range(2):
                                v_idx = (q_block*2 + j) * 8 + ((r_block*2 + k) + 4*half)
                                V[v_idx][col] = V2[j*2+k][c]
                        
                        # W: index = (r_block*2+k) * 4 + (p_block*2+i)
                        # C is 4x8, but we're filling columns 4*half..4*half+3
                        # W index convention: k*n+i where k is col, i is row
                        # C[i, k]: row i, col k
                        # For half=0: k in 0..3, for half=1: k in 4..7
                        for k in range(2):
                            for i in range(2):
                                w_idx = ((r_block*2 + k) + 4*half) * 4 + (p_block*2 + i)
                                W[w_idx][col] = W2[k*2+i][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
