"""Optimized matrix multiplication tensor decomposition for 6x6."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Strategy: Use a direct Kronecker product decomposition.
    # 6x6 multiplication can be viewed as a tensor product of 3x3 and 2x2 multiplications.
    # Specifically, we can use the Kronecker product of the 3x3 schoolbook multiplication
    # (rank 27) and the 2x2 Strassen multiplication (rank 7).
    # This yields a rank of 27 * 7 = 189, which is the same as the block-wise tiling approach
    # but constructed differently.
    
    # However, the prompt suggests a "rank-63" decomposition using "3x3 schoolbook tensor (rank 9)"?
    # Wait, the rank of 3x3 matrix multiplication is NOT 9. It is at least 19 (lower bound) and known upper bound is around 23-27.
    # The rank-9 decomposition mentioned in the strategy description is likely incorrect or refers to something else (maybe scalar multiplication of 3x3 matrices? No, that's 27).
    # If the strategy implies rank 63, it assumes rank(3x3 mult) = 9, which is impossible.
    # Therefore, the "rank-63" hypothesis is mathematically flawed for standard matrix multiplication.
    
    # Let's re-read the strategy: "Kronecker product ... 2x2 Strassen (rank 7) and 3x3 schoolbook (rank 9)".
    # This is a trap or a misunderstanding in the prompt's inspiration.
    # R(3) is definitely not 9. R(2)=7, R(3) >= 19.
    # So we cannot achieve rank 63.
    
    # The current program achieves 189.
    # Is there a better decomposition?
    # R(6) <= R(3)^2 * R(2)? No.
    # R(6) <= R(2)^3 * R(3)? No.
    # R(6) <= R(2) * R(3) is not a standard bound.
    # Standard bound: R(n*m) <= R(n) * R(m).
    # So R(6) <= R(2) * R(3) = 7 * R(3).
    # If we use the best known R(3) (which is 23? 24? 27?), we get 7*23=161, 7*24=168, 7*27=189.
    # The block-wise tiling of 2x2 Strassen into 6x6 effectively uses R(3) = 27 (schoolbook for the 3x3 block grid).
    # If we had a better 3x3 decomposition, we could improve the rank.
    
    # However, finding an optimal 3x3 decomposition is a hard open problem.
    # The best known explicit decomposition for 3x3 is rank 23 (Bini et al.?) or maybe 24?
    # Actually, the best known upper bound for 3x3 is 23 (Bini, Capovani, Lotti, Romani, 1987).
    # If we use a rank-23 3x3 decomposition and Kronecker product with 2x2 Strassen (rank 7),
    # we would get rank 23 * 7 = 161.
    
    # But constructing the explicit U, V, W matrices for a rank-23 3x3 decomposition is complex.
    # The current program uses rank 189.
    # Can we easily construct a rank-23 3x3 decomposition?
    # It involves specific integer/half-integer coefficients.
    
    # Let's check if we can implement the Bini et al. rank-23 3x3 algorithm.
    # It's quite involved.
    
    # Alternative: Is there a simpler improvement?
    # Maybe we can mix Strassen for 2x2 and schoolbook for 3x3 in a different way?
    
    # Given the constraints and the difficulty of implementing the rank-23 3x3 algorithm from scratch
    # within the time limit and ensuring correctness, and the fact that the "rank-63" idea is flawed,
    # the safest bet is to stick with the known correct rank-189 decomposition.
    
    # However, the directive says "investigate improvements within its family".
    # The family is "structured decompositions".
    # If I can't easily get 161, I should ensure the 189 implementation is robust and fast.
    
    # Let's verify the current implementation's correctness and efficiency.
    # The current code constructs U, V, W as numpy arrays and converts to lists.
    # This is efficient enough.
    
    # Is there any low-hanging fruit?
    # The rank is the score. Lower is better.
    # 189 is the current score.
    # If I can't lower it, I should just return the 189 solution.
    
    # Let's double check if there's a rank < 189 that is easy to construct.
    # R(6) <= R(2)*R(3). If R(3)=23, R(6) <= 161.
    # R(6) <= R(3)*R(2) = 161.
    # R(6) <= R(4)*R(1.5)? No.
    # R(6) <= R(2)*R(2)*R(3)? No, that's for 12x12?
    
    # Actually, there is a known algorithm for 6x6 with rank 161?
    # Or maybe 154?
    
    # Without the explicit coefficients for the 3x3 rank-23 algorithm, I cannot implement it.
    # I will stick to the rank-189 solution as it is correct and known.
    
    rank = 189
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen 2x2 multiplication components
    # A is 2x2, B is 2x2
    # Indices: A[0,0], A[0,1], A[1,0], A[1,1]
    #           B[0,0], B[0,1], B[1,0], B[1,1]
    
    # strassen_U[t] = list of (row, col, coeff) for A
    # strassen_V[t] = list of (row, col, coeff) for B
    # strassen_W[t] = list of (row, col, coeff) for C
    
    strassen_U = [
        [(0,0,1), (1,1,1)],  # M1: (A00+A11)
        [(1,0,1), (1,1,1)],  # M2: (A10+A11)
        [(0,0,1)],           # M3: A00
        [(1,1,1)],           # M4: A11
        [(0,0,1), (0,1,1)],  # M5: (A00+A01)
        [(1,0,1), (0,0,-1)], # M6: (A10-A00)
        [(0,1,1), (1,1,-1)]  # M7: (A01-A11)
    ]
    
    strassen_V = [
        [(0,0,1), (1,1,1)],  # M1: (B00+B11)
        [(0,0,1)],           # M2: B00
        [(0,1,1), (1,1,-1)], # M3: (B01-B11)
        [(1,0,1), (0,0,-1)], # M4: (B10-B00)
        [(1,1,1)],           # M5: B11
        [(0,0,1), (0,1,1)],  # M6: (B00+B01)
        [(1,0,1), (1,1,1)]   # M7: (B10+B11)
    ]
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # strassen_W[t] = list of (row, col, coeff) for C
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    strassen_W = [
        [(0,0,1), (1,1,1)],   # M1: C00+1, C11+1
        [(1,0,1), (1,1,-1)],  # M2: C10+1, C11-1
        [(0,1,1), (1,1,1)],   # M3: C01+1, C11+1
        [(0,0,1), (1,0,1)],   # M4: C00+1, C10+1
        [(0,0,-1), (0,1,1)],  # M5: C00-1, C01+1
        [(1,1,1)],            # M6: C11+1
        [(0,0,1)]             # M7: C00+1
    ]
    
    # Tiling 6x6 with 2x2 blocks
    # 3x3 grid of blocks
    for i in range(3):  # row block
        for j in range(3):  # middle block
            for k in range(3):  # col block
                for t in range(7):  # Strassen term
                    col_idx = (i * 9 + j * 3 + k) * 7 + t
                    
                    # Update U
                    for lr, lc, coeff in strassen_U[t]:
                        u_row = (i * 2 + lr) * m + (j * 2 + lc)
                        U[u_row, col_idx] += coeff
                    
                    # Update V
                    for lr, lc, coeff in strassen_V[t]:
                        v_row = (j * 2 + lr) * p + (k * 2 + lc)
                        V[v_row, col_idx] += coeff
                    
                    # Update W
                    # W index convention: W[k*n+i, r] where i is row of C, k is col of C
                    # So W[row_in_W, col_in_W] where row_in_W = col_of_C * n + row_of_C
                    for lr, lc, coeff in strassen_W[t]:
                        w_row = (k * 2 + lc) * n + (i * 2 + lr)
                        W[w_row, col_idx] += coeff
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
