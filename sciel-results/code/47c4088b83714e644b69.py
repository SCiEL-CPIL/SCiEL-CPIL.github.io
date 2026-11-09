"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    
    # We will use a block decomposition strategy.
    # We decompose 3x4x6 into 3x3x6 and 3x1x6.
    # C = A B where A is 3x4, B is 4x6.
    # Split A into [A1 | A2] where A1 is 3x3, A2 is 3x1.
    # Split B into [B1; B2] where B1 is 3x6, B2 is 1x6.
    # C = A1 B1 + A2 B2.
    
    # We need decompositions for:
    # 1. 3x3x6 multiplication
    # 2. 3x1x6 multiplication (outer product of col vector and row vector)
    
    # For 3x1x6:
    # A2 is 3x1 (col), B2 is 1x6 (row).
    # C2 = A2 B2 is 3x6.
    # The tensor is T[i, j, k] = 1 if j=0 (since B2 has only 1 row, index 0) and k corresponds to col of B2?
    # Wait, standard matrix mult tensor:
    # C[i, k] = sum_j A[i, j] B[j, k].
    # For 3x1x6:
    # A is 3x1. B is 1x6.
    # C[i, k] = A[i, 0] B[0, k].
    # Tensor T[i, j, k] = 1 if j=0 else 0.
    # Rank is 3*1*6? No.
    # C[i, k] = A[i, 0] B[0, k].
    # This is an outer product of vec(A) and vec(B).
    # Actually, the bilinear form is sum_j A_ij B_jk.
    # Here j only takes value 0.
    # So C_ik = A_i0 B_0k.
    # This is rank 1? No, it's a single term in the sum.
    # The tensor has non-zeros at (i, 0, k) for all i in 0..2, k in 0..5.
    # Total 18 non-zeros.
    # Can we do better than 18?
    # C_ik = A_i0 B_0k.
    # This is exactly the outer product of the column vector A_col and row vector B_row.
    # The rank of the tensor T_ijk = delta_{j,0} is 1?
    # Let's check. T_ijk = 1 if j=0.
    # Decomposition: U_i0 V_0k W_ki? No.
    # Standard form: sum_r U_ir V_rj W_kr? No.
    # Let's stick to the U, V, W convention:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i,j,k].
    # For 3x1x6: n=3, m=1, p=6.
    # T[i, 0, k] = 1.
    # We can set R=1.
    # U[i*1+0, 0] = 1.
    # V[0*6+k, 0] = 1.
    # W[k*3+i, 0] = 1.
    # Then sum = 1*1*1 = 1.
    # So rank 1 is sufficient for 3x1x6?
    # Wait.
    # U shape (3*1, 1) = (3, 1). All 1s.
    # V shape (1*6, 1) = (6, 1). All 1s.
    # W shape (6*3, 1) = (18, 1). All 1s.
    # Check:
    # i=0, j=0, k=0: U[0,0]V[0,0]W[0,0] = 1. T[0,0,0]=1. OK.
    # i=0, j=0, k=1: U[0,0]V[1,0]W[3,0] = 1. T[0,0,1]=1. OK.
    # i=1, j=0, k=0: U[1,0]V[0,0]W[1,0] = 1. T[1,0,0]=1. OK.
    # So yes, rank 1 works for 3x1x6.
    
    # Now for 3x3x6.
    # We can split 3x3x6 into two 3x3x3 blocks.
    # B1 is 3x6. Split into [B1a | B1b] where B1a is 3x3, B1b is 3x3.
    # C1 = A1 B1 = A1 B1a + A1 B1b.
    # So we need decompositions for 3x3x3.
    # Rank of 3x3x3 is known to be 23.
    # So total rank for 3x3x6 is 23 + 23 = 46.
    # Total rank for 3x4x6 is 46 + 1 = 47.
    
    # We need to find a rank 23 decomposition for 3x3x3.
    # I will implement a search for this.
    # If it fails, I will fall back to a higher rank, e.g., schoolbook (27) or something else.
    # Actually, schoolbook for 3x3x3 is 27.
    # If I can't find 23, I can use 27.
    # Then total rank would be 27 + 27 + 1 = 55.
    # Still better than 72.
    
    # Let's try to find a rank 23 solution for 3x3x3.
    # This is a hard problem. I might not find it quickly.
    # I will try a few random initializations and local search.
    
    def find_decomp_333(R_target):
        n_s, m_s, p_s = 3, 3, 3
        # T[i*m+j, j*p+k, k*n+i] = 1
        
        # Random search
        import random
        random.seed(123)
        
        best_solution = None
        
        # Try a few random initializations
        for attempt in range(20):
            U = np.zeros((n_s*m_s, R_target))
            V = np.zeros((m_s*p_s, R_target))
            W = np.zeros((p_s*n_s, R_target))
            
            # Initialize with small integers
            for r in range(R_target):
                for i in range(n_s*m_s):
                    U[i, r] = random.choice([-1, 0, 1])
                for j in range(m_s*p_s):
                    V[j, r] = random.choice([-1, 0, 1])
                for k in range(p_s*n_s):
                    W[k, r] = random.choice([-1, 0, 1])
            
            # Check
            success = True
            for i in range(n_s):
                for j in range(m_s):
                    for k in range(p_s):
                        val = np.dot(U[i*m_s+j] * V[j*p_s+k], W[k*n_s+i])
                        if abs(val - 1) > 1e-9:
                            success = False
                            break
                    if not success: break
                if not success: break
            if success:
                best_solution = (U, V, W)
                break
        
        if best_solution is None:
            # Fallback to schoolbook for 3x3x3 (rank 27)
            R_school = n_s * m_s * p_s
            U = np.zeros((n_s*m_s, R_school))
            V = np.zeros((m_s*p_s, R_school))
            W = np.zeros((p_s*n_s, R_school))
            col = 0
            for i in range(n_s):
                for j in range(m_s):
                    for k in range(p_s):
                        U[i*m_s+j, col] = 1
                        V[j*p_s+k, col] = 1
                        W[k*n_s+i, col] = 1
                        col += 1
            best_solution = (U, V, W)
        
        return best_solution
    
    # Get decomposition for 3x3x3
    sol_333 = find_decomp_333(23)
    R_333 = sol_333[0].shape[1]
    
    # Construct 3x3x6 decomposition
    # A1 is 3x3, B1 is 3x6.
    # B1 = [B1a | B1b]
    # C1 = A1 B1a + A1 B1b
    # We use two copies of the 3x3x3 decomposition.
    
    U_333, V_333, W_333 = sol_333
    
    # For 3x3x6:
    # n=3, m=3, p=6.
    # U shape (9, R_total)
    # V shape (18, R_total)
    # W shape (18, R_total)
    
    R_total_336 = 2 * R_333
    
    U_336 = np.zeros((9, R_total_336))
    V_336 = np.zeros((18, R_total_336))
    W_336 = np.zeros((18, R_total_336))
    
    # First 3x3x3 block (B1a)
    # j in 0..2, k in 0..2
    # V indices for this block: j*6 + k. j in 0..2, k in 0..2.
    # So V[0:3, :], V[6:9, :], V[12:15, :]
    # Wait, V index is j*p + k = j*6 + k.
    # For j=0, k=0..2 -> 0, 1, 2.
    # For j=1, k=0..2 -> 6, 7, 8.
    # For j=2, k=0..2 -> 12, 13, 14.
    
    # W indices: k*n + i = k*3 + i.
    # For k=0..2, i=0..2.
    # W[0:3, :], W[3:6, :], W[6:9, :]
    
    # U indices: i*m + j = i*3 + j.
    # i=0..2, j=0..2.
    # U[0:3, :], U[3:6, :], U[6:9, :]
    
    # Copy U
    U_336[:, 0:R_333] = U_333
    U_336[:, R_333:2*R_333] = U_333
    
    # Copy V
    # For the first block (k=0..2), V[j*6+k] should be V_333[j*3+k]
    for j in range(3):
        for k in range(3):
            V_336[j*6+k, 0:R_333] = V_333[j*3+k]
            V_336[j*6+k, R_333:2*R_333] = 0 # Wait, this is for B1a. B1b is separate.
            
    # Actually, let's be careful.
    # C1 = A1 B1a + A1 B1b.
    # Term 1: A1 B1a. This is a 3x3x3 multiplication.
    # Term 2: A1 B1b. This is also a 3x3x3 multiplication.
    
    # For Term 1:
    # B1a corresponds to columns 0..2 of B1.
    # So k in 0..2.
    # V indices for B1a: j*6 + k where k in 0..2.
    # These are V_333[j*3+k].
    
    # For Term 2:
    # B1b corresponds to columns 3..5 of B1.
    # So k in 3..5. Let k' = k-3.
    # V indices for B1b: j*6 + k where k in 3..5.
    # These should map to V_333[j*3+k'].
    
    # Let's fill V_336 properly.
    for j in range(3):
        for k in range(3):
            # B1a part (columns 0-2)
            V_336[j*6+k, 0:R_333] = V_333[j*3+k]
            # B1b part (columns 3-5)
            V_336[j*6+(k+3), R_333:2*R_333] = V_333[j*3+k]
            
    # W indices: k*3 + i.
    # For Term 1 (k=0..2): W[k*3+i] should be W_333[k*3+i].
    # For Term 2 (k=3..5): W[k*3+i] should be W_333[(k-3)*3+i].
    
    for k in range(3):
        for i in range(3):
            W_336[k*3+i, 0:R_333] = W_333[k*3+i]
            W_336[(k+3)*3+i, R_333:2*R_333] = W_333[k*3+i]
            
    # Now we have a decomposition for 3x3x6.
    # Rank is 2 * R_333.
    
    # Now combine with 3x1x6.
    # Total n=3, m=4, p=6.
    # A is 3x4. B is 4x6.
    # A = [A1 | A2]. A1 is 3x3, A2 is 3x1.
    # B = [B1; B2]. B1 is 3x6, B2 is 1x6.
    
    # C = A1 B1 + A2 B2.
    
    # We have decomposition for A1 B1 (3x3x6).
    # We have decomposition for A2 B2 (3x1x6) with rank 1.
    
    # Total Rank R_total = 2 * R_333 + 1.
    
    R_total = 2 * R_333 + 1
    
    U_total = np.zeros((12, R_total)) # n*m = 3*4 = 12
    V_total = np.zeros((24, R_total)) # m*p = 4*6 = 24
    W_total = np.zeros((18, R_total)) # p*n = 6*3 = 18
    
    # Fill U_total
    # U indices: i*4 + j.
    # j in 0..2 (A1): U[i*4+j] should be U_336[i*3+j].
    # j in 3 (A2): U[i*4+3] should be U_316[i*1+0].
    
    # U_316 is all 1s of shape (3, 1).
    
    for i in range(3):
        for j in range(3):
            U_total[i*4+j, 0:2*R_333] = U_336[i*3+j]
        # j=3
        U_total[i*4+3, 2*R_333] = 1 # Rank 1 part
        
    # Fill V_total
    # V indices: j*6 + k.
    # j in 0..2 (B1): V[j*6+k] should be V_336[j*6+k].
    # j in 3 (B2): V[3*6+k] should be V_316[0*6+k].
    
    # V_316 is all 1s of shape (6, 1).
    
    for j in range(3):
        for k in range(6):
            V_total[j*6+k, 0:2*R_333] = V_336[j*6+k]
    for k in range(6):
        V_total[3*6+k, 2*R_333] = 1
        
    # Fill W_total
    # W indices: k*3 + i.
    # W is the same for both parts?
    # C = A1 B1 + A2 B2.
    # W depends on k and i.
    # For A1 B1, W is W_336.
    # For A2 B2, W is W_316.
    # W_316 is all 1s of shape (18, 1).
    
    for k in range(6):
        for i in range(3):
            W_total[k*3+i, 0:2*R_333] = W_336[k*3+i]
            W_total[k*3+i, 2*R_333] = 1
            
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U_total]
    V_list = [[float(x) for x in row] for row in V_total]
    W_list = [[float(x) for x in row] for row in W_total]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
