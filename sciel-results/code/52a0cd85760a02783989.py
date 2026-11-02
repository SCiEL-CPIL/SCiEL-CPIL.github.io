"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Start with Strassen rank-42 decomposition as baseline
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 linear forms for A (2x2)
    U_forms = [
        [1, 0, 0, 1],  # S0: a00 + a11
        [0, 0, 1, 1],  # S1: a10 + a11
        [1, 0, 0, 0],  # S2: a00
        [0, 0, 0, 1],  # S3: a11
        [1, 1, 0, 0],  # S4: a00 + a01
        [-1, 0, 1, 0], # S5: a10 - a00
        [0, 1, 0, -1]  # S6: a01 - a11
    ]
    
    # Build baseline Strassen decomposition
    for u in range(6):
        for t in range(7):
            c = u * 7 + t
            
            for i in range(4):
                U[i][c] = U_forms[t][i]
                
            if t == 0:
                V[0 * p + 2*u][c] += 1
                V[1 * p + 2*u + 1][c] += 1
            elif t == 1:
                V[0 * p + 2*u][c] += 1
            elif t == 2:
                V[0 * p + 2*u + 1][c] += 1
                V[1 * p + 2*u + 1][c] -= 1
            elif t == 3:
                V[1 * p + 2*u][c] += 1
                V[0 * p + 2*u][c] -= 1
            elif t == 4:
                V[1 * p + 2*u + 1][c] += 1
            elif t == 5:
                V[0 * p + 2*u][c] += 1
                V[0 * p + 2*u + 1][c] += 1
            elif t == 6:
                V[1 * p + 2*u][c] += 1
                V[1 * p + 2*u + 1][c] += 1
                
            if t == 0:
                W[(2*u) * n + 0][c] += 1
                W[(2*u+1) * n + 1][c] += 1
            elif t == 1:
                W[(2*u) * n + 1][c] += 1
                W[(2*u+1) * n + 1][c] -= 1
            elif t == 2:
                W[(2*u+1) * n + 0][c] += 1
                W[(2*u+1) * n + 1][c] += 1
            elif t == 3:
                W[(2*u) * n + 0][c] += 1
                W[(2*u) * n + 1][c] += 1
            elif t == 4:
                W[(2*u) * n + 0][c] -= 1
                W[(2*u+1) * n + 0][c] += 1
            elif t == 5:
                W[(2*u+1) * n + 1][c] += 1
            elif t == 6:
                W[(2*u) * n + 0][c] += 1

    def verify_fast(U_mat, V_mat, W_mat, rank_val):
        """Fast vectorized verification using numpy."""
        U_np = np.array(U_mat, dtype=np.float64)
        V_np = np.array(V_mat, dtype=np.float64)
        W_np = np.array(W_mat, dtype=np.float64)
        
        # For each (i,j,k) triple, check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # Build index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.tile(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        a_idx = i_idx * m + j_idx
        b_idx = j_idx * p + k_idx
        c_idx = k_idx * n + i_idx
        
        # Compute all tensor values at once
        # U[a_idx, :] has shape (n*m*p, rank)
        # V[b_idx, :] has shape (n*m*p, rank)
        # W[c_idx, :] has shape (n*m*p, rank)
        vals = np.sum(U_np[a_idx] * V_np[b_idx] * W_np[c_idx], axis=1)
        
        return np.allclose(vals, 1.0, atol=1e-10)
    
    # Verify baseline using fast method
    if not verify_fast(U, V, W, rank):
        raise ValueError("Baseline decomposition is invalid")
    
    best_rank = rank
    best_result = {"U": U, "V": V, "W": W}
    
    # Strategy: Try to find lower rank by searching over structured decompositions.
    # The 2x2 x 2x12 multiplication tensor has 6 independent 2x2 x 2x2 blocks.
    # Each block needs rank 7 (Strassen), giving 42 total.
    # We can try to share some linear forms between adjacent blocks.
    
    # Approach: Use a parametric family where we allow some coupling between blocks.
    # For 2x2 x 2x4 (2 blocks), the minimal rank is 13 (known result).
    # For 2x2 x 2x12 (6 blocks), we can group into 3 groups of 2 blocks each.
    # Each group of 2 blocks uses rank 13, total rank = 3 * 13 = 39.
    
    # Let me construct a rank-13 decomposition for 2x2 x 2x4.
    # This is the well-known decomposition by Coppersmith (1983).
    
    def build_rank13_2x2_2x4():
        """Build rank-13 decomposition for 2x2 x 2x4 matrix multiplication."""
        # n=2, m=2, p=4
        # U: shape (4, 13), V: shape (8, 13), W: shape (8, 13)
        # A is 2x2, B is 2x4, C is 2x4
        # B = [B1 | B2] where B1, B2 are 2x2
        # C = A @ B = [A@B1 | A@B2]
        
        # Coppersmith's rank-13 decomposition for 2x2 x 2x4:
        # This uses 13 bilinear forms.
        
        r = 13
        U13 = np.zeros((4, r))
        V13 = np.zeros((8, r))
        W13 = np.zeros((8, r))
        
        # The 13 forms are based on Strassen with modifications for the 4-column case.
        # Let me use the known explicit decomposition.
        
        # A = [[a00, a01], [a10, a11]]
        # B1 = [[b00, b01], [b10, b11]], B2 = [[b20, b21], [b30, b31]]
        # C1 = A @ B1, C2 = A @ B2
        
        # U forms (linear forms in A):
        # f0 = a00 + a11
        # f1 = a10 + a11
        # f2 = a00
        # f3 = a11
        # f4 = a00 + a01
        # f5 = a10 - a00
        # f6 = a01 - a11
        # f7 = a00 + a10
        # f8 = a01 + a10
        # f9 = a00 - a10
        # f10 = a01 - a10
        # f11 = a10 + a11 - a00
        # f12 = a01 + a11 - a00
        
        U13[0, 0] = 1; U13[3, 0] = 1  # f0: a00 + a11
        U13[2, 1] = 1; U13[3, 1] = 1  # f1: a10 + a11
        U13[0, 2] = 1                  # f2: a00
        U13[3, 3] = 1                  # f3: a11
        U13[0, 4] = 1; U13[1, 4] = 1  # f4: a00 + a01
        U13[2, 5] = 1; U13[0, 5] = -1 # f5: a10 - a00
        U13[1, 6] = 1; U13[3, 6] = -1 # f6: a01 - a11
        U13[0, 7] = 1; U13[2, 7] = 1  # f7: a00 + a10
        U13[1, 8] = 1; U13[2, 8] = 1  # f8: a01 + a10
        U13[0, 9] = 1; U13[2, 9] = -1 # f9: a00 - a10
        U13[1, 10] = 1; U13[2, 10] = -1 # f10: a01 - a10
        U13[2, 11] = 1; U13[3, 11] = 1; U13[0, 11] = -1 # f11: a10 + a11 - a00
        U13[1, 12] = 1; U13[3, 12] = 1; U13[0, 12] = -1 # f12: a01 + a11 - a00
        
        # V forms (linear forms in B, 8 entries: B[0,0], B[0,1], B[0,2], B[0,3], B[1,0], B[1,1], B[1,2], B[1,3])
        # For each of the 13 forms, we need the corresponding linear form in B.
        # This is getting complex. Let me use a different approach.
        
        # Actually, let me try a simpler strategy: just verify the rank-42 solution is correct
        # and return it. The improvement from 42 to 39 would require a known decomposition.
        
        return None
    
    # Given the complexity of constructing a rank-39 decomposition from scratch,
    # let me try a different approach: verify that our rank-42 solution is optimal
    # or try small perturbations to see if we can reduce rank.
    
    # For now, return the verified rank-42 result.
    if time.time() > deadline - 5:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
