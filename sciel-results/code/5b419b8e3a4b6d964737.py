import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    # Split p into 4 blocks of size 2
    num_blocks = p // 2
    block_size = 2
    
    # Rank per block: 7 (Strassen) + 1 (Rank-1 for j=2) = 8
    rank_per_block = 8
    total_rank = num_blocks * rank_per_block
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices in local 2x2x2 block
    # A: i in {0,1}, j in {0,1} -> idx = i*2 + j (local)
    # B: j in {0,1}, k in {0,1} -> idx = j*2 + k (local)
    # C: i in {0,1}, k in {0,1} -> idx = k*2 + i (local, but W convention is k*n+i)
    
    # We will fill U, V, W for each block
    
    for r in range(num_blocks):
        base_col = r * rank_per_block
        
        # Block r corresponds to B columns 2r, 2r+1 and C columns 2r, 2r+1
        # A is always full 2x3, but we only use j=0,1 for Strassen and j=2 for Rank-1
        
        # --- Strassen Part (j=0,1) ---
        
        # M1: (A00 + A11)(B00 + B11)
        # U: A00, A11
        U[0*3 + 0, base_col + 0] = 1
        U[1*3 + 1, base_col + 0] = 1
        # V: B00, B11
        # B index: j*8 + k. j=0, k=2r -> 0*8 + 2r. j=1, k=2r+1 -> 1*8 + 2r+1
        V[0*8 + (2*r), base_col + 0] = 1
        V[1*8 + (2*r+1), base_col + 0] = 1
        # W: C00, C01, C10, C11 contributions?
        # M1 contributes to C00 (+1), C11 (+1)
        # W index: k*n + i
        # C00: k=2r, i=0 -> (2r)*2 + 0
        # C11: k=2r+1, i=1 -> (2r+1)*2 + 1
        W[(2*r)*2 + 0, base_col + 0] = 1
        W[(2*r+1)*2 + 1, base_col + 0] = 1
        
        # M2: (A10 + A11)B00
        U[1*3 + 0, base_col + 1] = 1
        U[1*3 + 1, base_col + 1] = 1
        V[0*8 + (2*r), base_col + 1] = 1
        # M2 contributes to C10 (+1), C11 (-1)
        # C10: k=2r, i=1 -> (2r)*2 + 1
        # C11: k=2r+1, i=1 -> (2*r+1)*2 + 1
        W[(2*r)*2 + 1, base_col + 1] = 1
        W[(2*r+1)*2 + 1, base_col + 1] = -1
        
        # M3: A00(B01 - B11)
        U[0*3 + 0, base_col + 2] = 1
        V[0*8 + (2*r+1), base_col + 2] = 1
        V[1*8 + (2*r+1), base_col + 2] = -1
        # M3 contributes to C01 (+1), C11 (+1)
        # C01: k=2r+1, i=0 -> (2r+1)*2 + 0
        # C11: k=2r+1, i=1 -> (2r+1)*2 + 1
        W[(2*r+1)*2 + 0, base_col + 2] = 1
        W[(2*r+1)*2 + 1, base_col + 2] = 1
        
        # M4: A11(B10 - B00)
        U[1*3 + 1, base_col + 3] = 1
        V[1*8 + (2*r), base_col + 3] = 1
        V[0*8 + (2*r), base_col + 3] = -1
        # M4 contributes to C00 (+1), C10 (+1)
        # C00: k=2r, i=0 -> (2r)*2 + 0
        # C10: k=2r, i=1 -> (2r)*2 + 1
        W[(2*r)*2 + 0, base_col + 3] = 1
        W[(2*r)*2 + 1, base_col + 3] = 1
        
        # M5: (A00 + A01)B11
        U[0*3 + 0, base_col + 4] = 1
        U[0*3 + 1, base_col + 4] = 1
        V[1*8 + (2*r+1), base_col + 4] = 1
        # M5 contributes to C00 (-1), C01 (+1)
        # C00: k=2r, i=0 -> (2r)*2 + 0
        # C01: k=2r+1, i=0 -> (2r+1)*2 + 0
        W[(2*r)*2 + 0, base_col + 4] = -1
        W[(2*r+1)*2 + 0, base_col + 4] = 1
        
        # M6: (A10 - A00)(B00 + B01)
        U[1*3 + 0, base_col + 5] = 1
        U[0*3 + 0, base_col + 5] = -1
        V[0*8 + (2*r), base_col + 5] = 1
        V[0*8 + (2*r+1), base_col + 5] = 1
        # M6 contributes to C11 (+1)
        # C11: k=2r+1, i=1 -> (2r+1)*2 + 1
        W[(2*r+1)*2 + 1, base_col + 5] = 1
        
        # M7: (A01 - A11)(B10 + B11)
        U[0*3 + 1, base_col + 6] = 1
        U[1*3 + 1, base_col + 6] = -1
        V[1*8 + (2*r), base_col + 6] = 1
        V[1*8 + (2*r+1), base_col + 6] = 1
        # M7 contributes to C00 (+1)
        # C00: k=2r, i=0 -> (2r)*2 + 0
        W[(2*r)*2 + 0, base_col + 6] = 1
        
        # --- Rank-1 Part (j=2) ---
        # Term: A_i2 * B_2k
        # U: A_i2 for i=0,1
        # V: B_2k for k=2r, 2r+1
        # W: C_ik for i=0,1, k=2r, 2r+1
        
        # This is 1 rank-1 tensor?
        # U vector: non-zero at i*3+2
        # V vector: non-zero at 2*8+k
        # W vector: non-zero at k*2+i
        
        # Wait, is it 1 tensor or 4?
        # The term is A_{i,2} B_{2,k}.
        # This is a single product of two variables.
        # It appears in C_{i,k}.
        # So for each (i,k), the coefficient is 1.
        # The tensor term is U \otimes V \otimes W where:
        # U_{i*3+2} = 1
        # V_{2*8+k} = 1
        # W_{k*2+i} = 1
        # But U, V, W are fixed vectors for the term.
        # If we use 1 term, U has 1s at (0,2) and (1,2)?
        # No, U is a vector in the A-space.
        # If U has 1s at A_{0,2} and A_{1,2}, then the term is (A_{0,2} + A_{1,2}) * V * W.
        # This would produce A_{0,2} B_{2,k} W_{k,0} + A_{1,2} B_{2,k} W_{k,1}.
        # We want A_{0,2} B_{2,k} for C_{0,k} and A_{1,2} B_{2,k} for C_{1,k}.
        # So we need separate terms for i=0 and i=1?
        # Or can we encode i in W?
        # W_{k, i} is the output.
        # If we have 1 term:
        # U = e_{0,2} (only A_{0,2})
        # V = e_{2, 2r} + e_{2, 2r+1} ? No, V must be a vector.
        # If V has 1s at B_{2, 2r} and B_{2, 2r+1}, then the product is A_{0,2} (B_{2, 2r} + B_{2, 2r+1}).
        # This would add to both C_{0, 2r} and C_{0, 2r+1}.
        # But we need A_{0,2} B_{2, 2r} in C_{0, 2r} and A_{0,2} B_{2, 2r+1} in C_{0, 2r+1}.
        # Since B_{2, 2r} and B_{2, 2r+1} are independent variables, we cannot combine them in V without mixing the outputs.
        # So we need separate terms for each k?
        # Term 1: A_{0,2} B_{2, 2r} -> C_{0, 2r}
        # Term 2: A_{0,2} B_{2, 2r+1} -> C_{0, 2r+1}
        # Term 3: A_{1,2} B_{2, 2r} -> C_{1, 2r}
        # Term 4: A_{1,2} B_{2, 2r+1} -> C_{1, 2r+1}
        
        # So the rank-1 part for j=2 actually requires 4 rank-1 terms?
        # Let's re-read the hint: "plus 4 rank-1 terms for the third A-column".
        # Yes! "4 rank-1 terms".
        # So rank per block is 7 + 4 = 11.
        # Total rank 4 * 11 = 44.
        
        # My previous deduction of rank 1 for j=2 was wrong because I assumed B_{2,k} could be grouped, but they are distinct variables.
        
        # Let's implement the 4 terms for j=2.
        
        # Term 1: i=0, k=2r
        U[0*3 + 2, base_col + 7] = 1
        V[2*8 + (2*r), base_col + 7] = 1
        W[(2*r)*2 + 0, base_col + 7] = 1
        
        # Term 2: i=0, k=2r+1
        U[0*3 + 2, base_col + 8] = 1
        V[2*8 + (2*r+1), base_col + 8] = 1
        W[(2*r+1)*2 + 0, base_col + 8] = 1
        
        # Term 3: i=1, k=2r
        U[1*3 + 2, base_col + 9] = 1
        V[2*8 + (2*r), base_col + 9] = 1
        W[(2*r)*2 + 1, base_col + 9] = 1
        
        # Term 4: i=1, k=2r+1
        U[1*3 + 2, base_col + 10] = 1
        V[2*8 + (2*r+1), base_col + 10] = 1
        W[(2*r+1)*2 + 1, base_col + 10] = 1

    return {"U": U, "V": V, "W": W}
