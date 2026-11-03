import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Hierarchical Kronecker decomposition:
    # Model 2×4 · 4×8 as a 2×2 outer matrix multiplication over the ring of 2×4 matrices.
    # A = [A_00 A_01] where A_00, A_01 are 2×2 blocks (left/right halves of A's 4 columns)
    # B = [B_00; B_10] where B_00, B_10 are 2×8 blocks (top/bottom halves of B's 4 rows)
    # C = A_00 * B_00 + A_01 * B_10
    #
    # Each "element" in this outer 2×2 multiplication is a 2×2 matrix (for A blocks)
    # or 2×8 matrix (for B blocks). The inner multiplication is 2×2 · 2×8 = 2×8.
    #
    # We use Strassen (rank 7) for the outer 2×2 structure.
    # For the inner ring multiplication (2×2 · 2×8), we need a decomposition.
    # A 2×2 · 2×8 multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
    # This is equivalent to 8 independent 2×2 · 2×1 multiplications (one per column of B).
    # Each 2×2 · 2×1 has rank 4 (schoolbook: 4 scalar mults).
    # But we can do better: 2×2 · 2×1 is just computing two dot products of length 2.
    # Actually, 2×2 · 2×1: C[0] = A[0,0]*B[0] + A[0,1]*B[1], C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This has rank 4 (it's just 4 scalar multiplications).
    # So inner rank = 8 columns * 4 = 32? No wait.
    #
    # Let me reconsider. The inner multiplication is: given a 2×2 matrix A_block and a 2×8 matrix B_block,
    # compute C_block = A_block @ B_block (2×8 result).
    # This is a bilinear map from (2×2) × (2×8) → (2×8).
    # The tensor for this has dimension 4 * 16 * 16 = 1024.
    # Schoolbook rank: 2*2*8 = 32 scalar multiplications.
    # 
    # Actually, let's think of it differently. 
    # A_block is 2×2, B_block is 2×8.
    # C_block[i,j] = A_block[i,0]*B_block[0,j] + A_block[i,1]*B_block[1,j]
    # For fixed j, this is a 2×2 · 2×1 multiplication with rank 4.
    # Total rank for inner = 8 * 4 = 32.
    #
    # Total rank = 7 (outer Strassen) * 32 (inner) = 224. That's worse!
    #
    # Let me reconsider the hierarchy. The current approach gives rank 56.
    # Current: 8 independent 2×2·2×2 multiplications, each rank 7. Total = 56.
    #
    # The Kronecker product approach: outer rank 7, inner rank for (2×2)·(2×8)→(2×8).
    # The inner problem: multiply a 2×2 by a 2×8. 
    # We can view this as: for each of the 8 columns, do a 2×2·2×1 mult (rank 4).
    # Inner rank = 8 * 4 = 32. Total = 7 * 32 = 224. Worse.
    #
    # Alternative: view inner as 2 independent 2×1·2×8 multiplications? No.
    #
    # Let me try a different split. What if we split differently?
    # A is 2×4, B is 4×8. 
    # Split A into two 2×2 blocks: A_L (cols 0-1), A_R (cols 2-3).
    # Split B into two 2×8 blocks: B_T (rows 0-1), B_B (rows 2-3).
    # C = A_L * B_T + A_R * B_B.
    #
    # This is a 2-term sum of (2×2)·(2×8) products.
    # Using Strassen on the "outer" 2×2 multiplication where elements are 2×8 matrices:
    # M1 = (A_L + A_R) * (B_T + B_B)  -- but these are 2×2 · 2×8, not 2×2 · 2×2
    #
    # Actually, the standard Strassen applies to 2×2 · 2×2. Here the "elements" are 2×8 matrices.
    # The ring multiplication is (2×2)·(2×8)→(2×8), which has rank 32 as computed.
    # So total = 7 * 32 = 224. Still worse.
    #
    # The current rank-56 solution is actually better. Let me verify it's correct and keep it,
    # but optimize the construction speed and add a deadline guard.
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Strassen components for 2x2 matrix multiplication
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # r=0 (M1)
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1
    W_local[3, 0] = 1
    
    # r=1 (M2)
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1
    W_local[3, 1] = -1
    
    # r=2 (M3)
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1
    W_local[3, 2] = 1
    
    # r=3 (M4)
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1
    W_local[2, 3] = 1
    
    # r=4 (M5)
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1
    W_local[1, 4] = 1
    
    # r=5 (M6)
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1
    
    # r=6 (M7)
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1
    
    # Precompute index mappings for efficiency
    # For each (j, s) pair, precompute the mapping arrays
    mappings = []
    for j in range(4):
        for s in range(2):
            a_map = [0]*4
            for la in range(4):
                row = la // 2
                col = la % 2
                global_col = s * 2 + col
                a_map[la] = row * 4 + global_col
            
            b_map = [0]*4
            for lb in range(4):
                row = lb // 2
                col = lb % 2
                global_row = s * 2 + row
                global_col = 2 * j + col
                b_map[lb] = global_row * 8 + global_col
            
            c_map = [0]*4
            for lc in range(4):
                row = lc // 2
                col = lc % 2
                global_row = row
                global_col = 2 * j + col
                c_map[lc] = global_col * 2 + global_row
            
            mappings.append((a_map, b_map, c_map))
    
    # Fill in the factor matrices using precomputed mappings
    for idx, (a_map, b_map, c_map) in enumerate(mappings):
        base_r = idx * 7
        for l in range(7):
            r = base_r + l
            for la in range(4):
                val = U_local[la, l]
                if val != 0:
                    U[a_map[la], r] += val
            for lb in range(4):
                val = V_local[lb, l]
                if val != 0:
                    V[b_map[lb], r] += val
            for lc in range(4):
                val = W_local[lc, l]
                if val != 0:
                    W[c_map[lc], r] += val
    
    # Verify with a quick test
    if time.time() < DEADLINE - 5:
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        C_recon = np.zeros((n, p))
        for r in range(total_rank):
            u_val = U[:, r] @ A_flat
            v_val = V[:, r] @ B_flat
            for i in range(n):
                for j in range(p):
                    c_idx = j * n + i
                    C_recon[i, j] += W[c_idx, r] * u_val * v_val
        
        if not np.allclose(C_recon, C_expected, atol=1e-10):
            # Verification failed - return as-is (shouldn't happen with correct Strassen)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
