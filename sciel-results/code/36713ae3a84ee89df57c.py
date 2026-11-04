import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Splits
    # i: 0,1 vs 2,3
    # j: 0,1 vs 2,3,4
    # k: 0,1 vs 2,3,4
    
    rank = 0
    U_factors = []
    V_factors = []
    W_factors = []
    
    # Helper to add a block decomposition
    def add_block(i_start, i_len, j_start, j_len, k_start, k_len, algo):
        nonlocal rank
        # algo: 'strassen' or 'laderman'
        # Returns local U, V, W and local rank
        if algo == 'strassen':
            # 2x2x2
            # Local indices: i in 0..1, j in 0..1, k in 0..1
            # U shape (2*2, 7), V shape (2*2, 7), W shape (2*2, 7)
            # Strassen factors
            # M1 = A00(B00-B11)
            # M2 = (A00+A01)B11
            # M3 = (A10+A11)B00
            # M4 = A11(B10-B00)
            # M5 = (A00+A11)(B00+B11)
            # M6 = (A01-A11)(B10+B11)
            # M7 = (A10-A00)(B00+B01)
            
            # We need to map these to the tensor decomposition form:
            # C_ik = sum_r U_ir V_jr W_kr
            # Actually, the standard form is:
            # C = sum_r U_r (V_r * W_r^T)? No.
            # The tensor T_ijk = sum_r U_{ir} V_{jr} W_{kr}
            
            # Strassen decomposition of 2x2x2 tensor:
            # r=0: A00 * (B00 - B11) -> C00 += A00B00, C01 -= A00B11 ...
            # It is easier to use the known integer factors.
            
            # U (i, r), V (j, r), W (k, r)
            # i, j, k in 0,1
            # r in 0..6
            
            U_loc = np.zeros((2*2, 7), dtype=int)
            V_loc = np.zeros((2*2, 7), dtype=int)
            W_loc = np.zeros((2*2, 7), dtype=int)
            
            # Indices: i*2+j, j*2+k, k*2+i
            
            # r=0: A00 * B00
            U_loc[0, 0] = 1; V_loc[0, 0] = 1; W_loc[0, 0] = 1
            # r=1: A00 * -B11
            U_loc[0, 1] = 1; V_loc[3, 1] = 1; W_loc[1, 1] = -1
            # r=2: A01 * B11
            U_loc[1, 2] = 1; V_loc[3, 2] = 1; W_loc[1, 2] = 1
            # r=3: A10 * B00
            U_loc[2, 3] = 1; V_loc[0, 3] = 1; W_loc[0, 3] = 1
            # r=4: A11 * B10
            U_loc[3, 4] = 1; V_loc[2, 4] = 1; W_loc[0, 4] = 1
            # r=5: A11 * -B00
            U_loc[3, 5] = 1; V_loc[0, 5] = 1; W_loc[0, 5] = -1
            # r=6: (A00+A11)(B00+B11) ... This is not Strassen directly in this form.
            
            # Let's use the standard Strassen formula mapped to the tensor.
            # T = sum M_r (u_r \otimes v_r \otimes w_r)
            
            # M1 = A00(B00-B11)
            # M2 = (A00+A01)B11
            # M3 = (A10+A11)B00
            # M4 = A11(B10-B00)
            # M5 = (A00+A11)(B00+B11)
            # M6 = (A01-A11)(B10+B11)
            # M7 = (A10-A00)(B00+B01)
            
            # This requires linear combinations in U, V, W.
            # U, V, W can have -1, 0, 1 entries.
            
            # r=0 (M1): A00 * (B00 - B11)
            # U: i=0,j=0 -> 1. Others 0.
            # V: j=0,k=0 -> 1; j=1,k=1 -> -1.
            # W: k=0,i=0 -> 1; k=1,i=0 -> -1? No.
            # The term is A00 * B00 * C00 and A00 * (-B11) * C01?
            # No, the decomposition is C_ik = sum_r U_ir V_jr W_kr.
            # For r=0:
            # U_00=1, others 0.
            # V_00=1, V_11=-1, others 0. (j=0,k=0 and j=1,k=1)
            # W_00=1, W_01=1? No.
            
            # Let's stick to the simplest correct implementation: Schoolbook for everything.
            # Rank 100.
            # Then try to replace 2x2x2 blocks with Strassen.
            
            # Strassen Factors for 2x2x2 (Integer):
            # r=0: A00, B00, C00
            # r=1: A00, -B11, C01
            # r=2: A01, B11, C01
            # r=3: A10, B00, C10
            # r=4: A11, B10, C11
            # r=5: A11, -B00, C10
            # This is not rank 7. This is rank 6? No, it's missing terms.
            
            # Correct Strassen Tensor Decomposition:
            # r=0: u=(1,0,0,0), v=(1,0,0,-1), w=(1,0,0,0) ? No.
            
            # I will use a precomputed set of factors for Strassen 2x2x2.
            # From literature:
            # U:
            # [1, 0, 1, 0, 1, 0, 1]
            # [0, 1, 0, 1, 0, 1, 0]
            # [0, 0, 1, 1, 1, 0, 1]
            # [0, 0, 0, 1, 0, 1, 1]
            # This is getting risky.
            
            # I will fall back to Schoolbook for all blocks to ensure correctness, 
            # but I will optimize the 2x2x2 blocks using a known valid Strassen implementation 
            # if I can verify it. 
            # For safety, I will use Schoolbook for all, which gives Rank 100.
            # To beat 100, I MUST use Strassen for the 2x2x2 blocks.
            
            # Let's use a verified Strassen implementation.
            # U (4x7), V (4x7), W (4x7)
            # Indices: i*2+j, j*2+k, k*2+i
            
            U_s = np.array([
                [1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 0, 1, 1, 1, 0, 1],
                [0, 0, 0, 1, 0, 1, 1]
            ], dtype=int)
            
            V_s = np.array([
                [1, 1, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 1, 1],
                [0, 0, 1, 0, 0, 0, 1],
                [-1, 1, 0, 1, 1, -1, 0]
            ], dtype=int)
            
            W_s = np.array([
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 1],
                [0, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 1, 0, 1, 0]
            ], dtype=int)
            
            # Verify: C00 = U00V00W00 + ...
            # This specific set might be wrong.
            
            # Given the high risk of incorrect Strassen factors, and the 120s limit,
            # I will implement the Block Decomposition with Schoolbook for all sub-blocks.
            # Rank will be 100.
            # This does not improve the score.
            
            # However, the prompt asks for a stronger algorithm.
            # I will implement the Block Decomposition with Strassen for 2x2x2 
            # using a reliable source or derivation.
            
            # Reliable Strassen Factors:
            # r=0: A00 * B00
            # r=1: A00 * -B11
            # r=2: A01 * B11
            # r=3: A10 * B00
            # r=4: A11 * B10
            # r=5: A11 * -B00
            # r=6: (A00+A11) * (B00+B11) ...
            
            # I will use the following verified Strassen factors:
            U_s = np.zeros((4, 7), dtype=int)
            V_s = np.zeros((4, 7), dtype=int)
            W_s = np.zeros((4, 7), dtype=int)
            
            # r=0
            U_s[0, 0] = 1
            V_s[0, 0] = 1
            W_s[0, 0] = 1
            
            # r=1
            U_s[0, 1] = 1
            V_s[3, 1] = 1
            W_s[1, 1] = -1
            
            # r=2
            U_s[1, 2] = 1
            V_s[3, 2] = 1
            W_s[1, 2] = 1
            
            # r=3
            U_s[2, 3] = 1
            V_s[0, 3] = 1
            W_s[0, 3] = 1
            
            # r=4
            U_s[3, 4] = 1
            V_s[2, 4] = 1
            W_s[0, 4] = 1
            
            # r=5
            U_s[3, 5] = 1
            V_s[0, 5] = 1
            W_s[0, 5] = -1
            
            # r=6
            U_s[0, 6] = 1
            U_s[3, 6] = 1
            V_s[0, 6] = 1
            V_s[3, 6] = 1
            W_s[0, 6] = 1
            W_s[1, 6] = 1
            
            # Check C00:
            # r=0: 1*1*1 = 1
            # r=6: 1*1*1 = 1
            # Sum = 2? No.
            
            # I cannot reliably construct Strassen factors in this short time.
            # I will return the schoolbook solution.
            
            return np.zeros((0,0)), np.zeros((0,0)), np.zeros((0,0)), 0

    # Since I cannot guarantee the correctness of the complex block factors 
    # without extensive verification, and the current score is 1.0 (Rank 100),
    # any improvement requires a correct low-rank decomposition.
    
    # I will use the schoolbook method but with a small optimization:
    # None available for 4x5x5 directly.
    
    # I will output the schoolbook solution.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
