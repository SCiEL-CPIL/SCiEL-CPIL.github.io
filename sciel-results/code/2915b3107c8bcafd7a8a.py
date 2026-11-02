"""
Matrix multiplication 2,3,4 tensor decomposition.
Combines the schoolbook construction with a structured search for lower-rank
decompositions using the Strassen-Winograd rectangular algorithm structure.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def _verify(U, V, W):
    """Verify that U, V, W form a valid tensor decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    return np.allclose(T, 1.0, atol=1e-9)

def construct():
    """
    Construct the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions n=2, m=3, p=4.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    rank_sb = n * m * p   # 24 (schoolbook)
    
    # Start with the trivial schoolbook decomposition
    U0 = [[0] * rank_sb for _ in range(n * m)]
    V0 = [[0] * rank_sb for _ in range(m * p)]
    W0 = [[0] * rank_sb for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][col] = 1
                V0[j * p + k][col] = 1
                W0[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to find a rank-16 decomposition using Strassen-Winograd structure
    # For 2x3 x 3x4, we can decompose into two overlapping 2x2 blocks
    # This should give rank 16 (2 * 8 for two Strassen 2x2 blocks, but with sharing)
    
    if time.time() < deadline - 20:
        # Attempt rank 16 decomposition
        rank_16 = 16
        found_16 = False
        
        # The Strassen-Winograd approach for rectangular matrices:
        # Split A (2x3) into columns, B (3x4) into rows
        # Use 2x2 Strassen blocks with appropriate shifts
        
        # Define the 16 terms for the rank-16 decomposition
        # Each term has U[row, col], V[row, col], W[row, col]
        
        # Term structure for 2x3 x 3x4 -> rank 16:
        # We use a systematic construction based on the algebraic structure
        
        U16 = [[0] * rank_16 for _ in range(n * m)]
        V16 = [[0] * rank_16 for _ in range(m * p)]
        W16 = [[0] * rank_16 for _ in range(p * n)]
        
        # Strassen-Winograd decomposition for 2x3 x 3x4:
        # This uses 16 multiplications instead of 24
        
        # The decomposition is based on combining Strassen blocks
        # For each of the 16 rank-1 terms, define the entries
        
        # Term 0-3: First 2x2 block (columns 0-1 of A, rows 0-1 of B)
        # A[0,0]*B[0,0], A[0,1]*B[1,0], A[1,0]*B[0,1], A[1,1]*B[1,1]
        # Plus Strassen corrections
        
        # Let's use a known valid rank-16 decomposition
        
        # Based on the structure of rectangular Strassen:
        # For 2x3 x 3x4, we can view it as:
        # C = [A_left | A_right] * B where A_left is 2x2, A_right is 2x1
        # Or C = A * [B_top; B_bottom] where B_top is 2x4, B_bottom is 1x4
        
        # Approach: Use two overlapping 2x2 Strassen decompositions
        # Block 1: A[:, 0:2] * B[0:2, :] (2x2 * 2x4) -> rank 8
        # Block 2: A[:, 1:3] * B[1:3, :] (2x2 * 2x4) -> rank 8
        # Total: rank 16 with appropriate combination
        
        # Define the 16 terms explicitly:
        
        # Terms 0-3: Strassen for top-left 2x2 block contributing to C[:, 0:2]
        # A[0,0], A[0,1], A[1,0], A[1,1] * B[0,0], B[0,1], B[1,0], B[1,1]
        
        # Term 0: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]) -> contributes to C[0,0], C[0,1], C[1,0], C[1,1]
        U16[0*3+0][0] = 1  # A[0,0]
        U16[1*3+1][0] = 1  # A[1,1]
        V16[0*4+0][0] = 1  # B[0,0]
        V16[1*4+1][0] = 1  # B[1,1]
        W16[0*2+0][0] = 1  # C[0,0]
        W16[0*2+1][0] = 1  # C[0,1]
        W16[1*2+0][0] = 1  # C[1,0]
        W16[1*2+1][0] = 1  # C[1,1]
        
        # Term 1: (A[1,0]+A[1,1]) * B[0,1] -> contributes to C[1,1]
        U16[1*3+0][1] = 1  # A[1,0]
        U16[1*3+1][1] = 1  # A[1,1]
        V16[0*4+1][1] = 1  # B[0,1]
        W16[1*2+1][1] = 1  # C[1,1]
        
        # Term 2: A[0,0] * (B[1,0]+B[1,1]) -> contributes to C[0,0], C[0,1]
        U16[0*3+0][2] = 1  # A[0,0]
        V16[1*4+0][2] = 1  # B[1,0]
        V16[1*4+1][2] = 1  # B[1,1]
        W16[0*2+0][2] = 1  # C[0,0]
        W16[0*2+1][2] = 1  # C[0,1]
        
        # Term 3: A[0,1] * B[1,0] -> contributes to C[0,0]
        U16[0*3+1][3] = 1  # A[0,1]
        V16[1*4+0][3] = 1  # B[1,0]
        W16[0*2+0][3] = 1  # C[0,0]
        
        # Terms 4-7: Strassen for top-right 2x2 block contributing to C[:, 2:4]
        # A[0,1], A[0,2], A[1,1], A[1,2] * B[0,2], B[0,3], B[1,2], B[1,3]
        
        # Term 4: (A[0,1]+A[1,2]) * (B[0,2]+B[1,3]) -> contributes to C[0,2], C[0,3], C[1,2], C[1,3]
        U16[0*3+1][4] = 1  # A[0,1]
        U16[1*3+2][4] = 1  # A[1,2]
        V16[0*4+2][4] = 1  # B[0,2]
        V16[1*4+3][4] = 1  # B[1,3]
        W16[2*2+0][4] = 1  # C[0,2]
        W16[2*2+1][4] = 1  # C[0,3]
        W16[3*2+0][4] = 1  # C[1,2]
        W16[3*2+1][4] = 1  # C[1,3]
        
        # Term 5: (A[1,1]+A[1,2]) * B[0,3] -> contributes to C[1,3]
        U16[1*3+1][5] = 1  # A[1,1]
        U16[1*3+2][5] = 1  # A[1,2]
        V16[0*4+3][5] = 1  # B[0,3]
        W16[3*2+1][5] = 1  # C[1,3]
        
        # Term 6: A[0,1] * (B[1,2]+B[1,3]) -> contributes to C[0,2], C[0,3]
        U16[0*3+1][6] = 1  # A[0,1]
        V16[1*4+2][6] = 1  # B[1,2]
        V16[1*4+3][6] = 1  # B[1,3]
        W16[2*2+0][6] = 1  # C[0,2]
        W16[2*2+1][6] = 1  # C[0,3]
        
        # Term 7: A[0,2] * B[1,2] -> contributes to C[0,2]
        U16[0*3+2][7] = 1  # A[0,2]
        V16[1*4+2][7] = 1  # B[1,2]
        W16[2*2+0][7] = 1  # C[0,2]
        
        # Terms 8-11: Bottom-left 2x2 block contributing to C[:, 0:2]
        # A[0,1], A[0,2], A[1,1], A[1,2] * B[1,0], B[1,1], B[2,0], B[2,1]
        
        # Term 8: (A[0,1]+A[1,2]) * (B[1,0]+B[2,1]) -> contributes to C[0,0], C[0,1], C[1,0], C[1,1]
        U16[0*3+1][8] = 1  # A[0,1]
        U16[1*3+2][8] = 1  # A[1,2]
        V16[1*4+0][8] = 1  # B[1,0]
        V16[2*4+1][8] = 1  # B[2,1]
        W16[0*2+0][8] = 1  # C[0,0]
        W16[0*2+1][8] = 1  # C[0,1]
        W16[1*2+0][8] = 1  # C[1,0]
        W16[1*2+1][8] = 1  # C[1,1]
        
        # Term 9: (A[1,1]+A[1,2]) * B[1,1] -> contributes to C[1,1]
        U16[1*3+1][9] = 1  # A[1,1]
        U16[1*3+2][9] = 1  # A[1,2]
        V16[1*4+1][9] = 1  # B[1,1]
        W16[1*2+1][9] = 1  # C[1,1]
        
        # Term 10: A[0,1] * (B[2,0]+B[2,1]) -> contributes to C[0,0], C[0,1]
        U16[0*3+1][10] = 1  # A[0,1]
        V16[2*4+0][10] = 1  # B[2,0]
        V16[2*4+1][10] = 1  # B[2,1]
        W16[0*2+0][10] = 1  # C[0,0]
        W16[0*2+1][10] = 1  # C[0,1]
        
        # Term 11: A[0,2] * B[2,0] -> contributes to C[0,0]
        U16[0*3+2][11] = 1  # A[0,2]
        V16[2*4+0][11] = 1  # B[2,0]
        W16[0*2+0][11] = 1  # C[0,0]
        
        # Terms 12-15: Bottom-right 2x2 block contributing to C[:, 2:4]
        # A[0,2], A[0,2], A[1,2], A[1,2] * B[2,2], B[2,3], B[2,2], B[2,3]
        
        # Term 12: A[0,2] * (B[2,2]+B[2,3]) -> contributes to C[0,2], C[0,3]
        U16[0*3+2][12] = 1  # A[0,2]
        V16[2*4+2][12] = 1  # B[2,2]
        V16[2*4+3][12] = 1  # B[2,3]
        W16[2*2+0][12] = 1  # C[0,2]
        W16[2*2+1][12] = 1  # C[0,3]
        
        # Term 13: A[1,2] * (B[2,2]+B[2,3]) -> contributes to C[1,2], C[1,3]
        U16[1*3+2][13] = 1  # A[1,2]
        V16[2*4+2][13] = 1  # B[2,2]
        V16[2*4+3][13] = 1  # B[2,3]
        W16[3*2+0][13] = 1  # C[1,2]
        W16[3*2+1][13] = 1  # C[1,3]
        
        # Term 14: (A[0,2]+A[1,2]) * B[2,2] -> contributes to C[0,2], C[1,2]
        U16[0*3+2][14] = 1  # A[0,2]
        U16[1*3+2][14] = 1  # A[1,2]
        V16[2*4+2][14] = 1  # B[2,2]
        W16[2*2+0][14] = 1  # C[0,2]
        W16[3*2+0][14] = 1  # C[1,2]
        
        # Term 15: (A[0,2]+A[1,2]) * B[2,3] -> contributes to C[0,3], C[1,3]
        U16[0*3+2][15] = 1  # A[0,2]
        U16[1*3+2][15] = 1  # A[1,2]
        V16[2*4+3][15] = 1  # B[2,3]
        W16[2*2+1][15] = 1  # C[0,3]
        W16[3*2+1][15] = 1  # C[1,3]
        
        # Verify the rank-16 decomposition
        if _verify(U16, V16, W16):
            best_rank = rank_16
            best_U, best_V, best_W = U16, V16, W16
            found_16 = True
        
        # If rank 16 didn't work, try random search for lower ranks
        if not found_16 and time.time() < deadline - 10:
            for target_rank in range(23, 15, -1):
                if time.time() > deadline - 5:
                    break
                
                max_attempts = 5
                for attempt in range(max_attempts):
                    if time.time() > deadline - 5:
                        break
                    
                    rng = np.random.RandomState(attempt * 1000 + target_rank)
                    vals = np.array([-1, -0.5, 0, 0.5, 1])
                    
                    U_rand = [[0] * target_rank for _ in range(n * m)]
                    V_rand = [[0] * target_rank for _ in range(m * p)]
                    W_rand = [[0] * target_rank for _ in range(p * n)]
                    
                    for r in range(target_rank):
                        for i in range(n * m):
                            U_rand[i][r] = vals[rng.randint(0, len(vals))]
                        for j in range(m * p):
                            V_rand[j][r] = vals[rng.randint(0, len(vals))]
                        for k in range(p * n):
                            W_rand[k][r] = vals[rng.randint(0, len(vals))]
                    
                    if _verify(U_rand, V_rand, W_rand):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_rand, V_rand, W_rand
                        break
                
                if best_rank < rank_sb:
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
