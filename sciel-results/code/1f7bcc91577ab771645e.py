import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Rank 44 decomposition
    # Strategy: Split A (3x4) into A_top (2x4) and A_bot (1x4)
    # C_top = A_top * B (2x4) using Strassen 4x4 padding (Rank 28)
    # C_bot = A_bot * B (1x4) using standard (Rank 16)
    
    # For the 2x4 * 4x4 part, we can view it as a 4x4 * 4x4 multiplication where
    # the bottom 2 rows of the result are discarded, but we still need the full
    # Strassen structure to get rank 28? 
    # Actually, for 2x4 * 4x4, we can use the fact that it's a "thin" matrix multiplication.
    # Let's try to construct a rank 28 decomposition for 2x4 * 4x4.
    # This is equivalent to multiplying a 4x4 matrix (with bottom 2 rows zero) by a 4x4 matrix,
    # and taking the top 2 rows of the result.
    # The rank of the tensor [2,4,4] is known to be 28? 
    # Let's check: rank([2,4,4]) <= rank([4,4,4]) = 28.
    # Can we do better? 
    # Schoolbook for 2x4*4x4 is 2*4*4 = 32.
    # Strassen for 4x4*4x4 is 28. Since we are only using half the input, maybe we can save?
    # No, the intermediate products M1..M28 in Strassen 4x4 involve all elements.
    # However, if the bottom half of A is zero, some M's might simplify?
    # Let's assume rank 28 for the top part.
    
    # For the 1x4 * 4x4 part, schoolbook is 1*4*4 = 16.
    # Can we do better? rank([1,4,4]) is 16? 
    # It's essentially 4 dot products of length 4. 
    # The rank of [1,4,4] is 16 (it's the sum of 4 rank-4 tensors? No, it's 4 independent dot products).
    # Actually, [1,4,4] is isomorphic to 4 copies of [1,4,1] concatenated? 
    # No, C_k = sum_j A_j B_jk.
    # This is a sum of 4 tensors of shape [1,4,1] for each output k? No.
    # For each output k, C_k = A . B[:,k].
    # This is a bilinear form.
    # The rank is 16.
    
    # Total rank = 28 + 16 = 44.
    
    # Let's build the factors.
    # We will use a dictionary to store the terms.
    # Each term is (u, v, w) where u is a vector in R^(n*m), v in R^(m*p), w in R^(p*n).
    # Or we can build the matrices directly.
    
    U = np.zeros((n * m, 44))
    V = np.zeros((m * p, 44))
    W = np.zeros((p * n, 44))
    
    col = 0
    
    # Part 1: Top 2 rows (2x4 * 4x4)
    # We need a rank 28 decomposition for 2x4 * 4x4.
    # Let's use the Strassen 4x4 decomposition and project.
    # Strassen 4x4 factors are known. 
    # Let's find them or construct them.
    # Since I don't have the explicit 28 factors for 4x4 memorized, I will generate them via recursion or lookup.
    # Alternatively, I can use the known rank 28 for 4x4 and just take the top half of the W factors?
    # Wait, if I use the 4x4 Strassen, I get a 4x4 output. I only need the top 2x4 output.
    # The U and V factors are the same as 4x4 Strassen?
    # For 4x4 Strassen, U is 16x28, V is 16x28, W is 16x28.
    # For 2x4 * 4x4, A is 8x1 (vector), B is 16x1.
    # If I embed A into a 4x4 matrix with bottom 2 rows zero, and use 4x4 Strassen,
    # the result C will be 4x4. I take top 2 rows.
    # The U factors for the 4x4 Strassen will have non-zero entries in the bottom 2 rows of the "A" block?
    # No, the U factors multiply the flattened A. If A_bottom is zero, those terms vanish in the product sum?
    # No, the decomposition is exact. U @ lambda @ V^T = T.
    # If I restrict A to the subspace where bottom 2 rows are zero, the rank of the restricted tensor might be lower?
    # But I'm using the full 28 factors. Some might be redundant?
    # Let's assume I can use the 28 factors of 4x4 Strassen, and just zero out the parts of W that correspond to the bottom 2 rows of C?
    # No, W is the output factor.
    # Let's try to construct the 4x4 Strassen factors explicitly.
    
    # Strassen 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # 4x4 Strassen is built from 4x4 blocks? No, it's a specific decomposition.
    # I will use a helper to generate Strassen factors recursively.
    
    def strassen_factors_2x2():
        # Returns U, V, W for 2x2 * 2x2
        # U is 4x7, V is 4x7, W is 4x7
        # Indices: A11=0, A12=1, A21=2, A22=3
        # B11=0, B12=1, B21=2, B22=3
        # C11=0, C12=1, C21=2, C22=3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C11, C22
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1 # C21, C22 (M2 is in C21 and C22? No)
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # So M2 contributes +1 to C21, -1 to C22.
        W[2, 1] = 1
        W[3, 1] = -1
        
        # M3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        # C12 = M3 + M5
        # C22 = ... + M3 ...
        W[1, 2] = 1
        W[3, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        # C11 = ... + M4 ...
        # C21 = M2 + M4
        W[0, 3] = 1
        W[2, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C11 = ... - M5 ...
        # C12 = M3 + M5
        W[0, 4] = -1
        W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C22 = ... + M6 ...
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C11 = ... + M7 ...
        W[0, 6] = 1
        
        return U, V, W
    
    def strassen_factors_4x4():
        # Construct 4x4 Strassen from 2x2 Strassen?
        # No, 4x4 Strassen is not simply 4 copies of 2x2.
        # It is a specific rank 28 decomposition.
        # I will use the fact that rank([4,4,4]) = 28.
        # I can find the factors online or derive them.
        # For now, I will use the schoolbook rank 64 for 4x4 to be safe?
        # No, I need rank 44 total.
        # If I use schoolbook for 2x4 (rank 32) and 1x4 (rank 16), total is 48.
        # The current program returns 48.
        # To get 44, I MUST use a better decomposition for 2x4*4x4.
        # Let's try to find a rank 28 decomposition for 2x4*4x4.
        
        # Since I cannot easily code the 28 factors from scratch, I will use a trick.
        # I will use the 4x4 Strassen factors (rank 28) and "trim" them.
        # If I have the 4x4 Strassen factors U4, V4, W4 (16x28),
        # For 2x4 * 4x4, I take A as the top 8 elements of the 16-element vector.
        # So U_eff = U4[:8, :].
        # V_eff = V4.
        # W_eff = W4[:8, :] (top 8 elements of C, which correspond to top 2 rows).
        # Does this still decompose the tensor?
        # The tensor T_{i,j,k} = delta_{i,j} for the multiplication.
        # If I restrict i to top 2 rows, the tensor is T_{i,j,k} for i<2.
        # The decomposition sum_r U_r V_r W_r = T.
        # If I take the same U_r, V_r, W_r but only use the parts where U_r and W_r have support in the valid indices,
        # it might not work because the sum might cancel out or be incomplete.
        
        # However, it is known that rank([2,4,4]) = 28.
        # I will hardcode the 28 factors for [2,4,4] if I can find them.
        # Alternatively, I can use the fact that [2,4,4] is a sub-tensor of [4,4,4].
        # Let's assume I can use the 28 factors of 4x4 Strassen, and just zero out the rows of U and W that correspond to the bottom 2 rows of A and C?
        # If U_r has non-zero entries in the bottom 2 rows of A, and A is zero there, the term is 0?
        # No, the decomposition is an identity.
        
        # Let's stick to rank 48 for now to ensure correctness, as the prompt says "Keep the function names...".
        # The current program is rank 48.
        # The goal is to improve it.
        # If I cannot find a valid rank 44 decomposition quickly, I should return rank 48.
        
        pass
    
    # Since I am not confident in constructing the rank 28 factors for 2x4x4 without external lookup,
    # and the deadline is tight, I will return the schoolbook rank 48 decomposition.
    # This is the same as the current program.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
