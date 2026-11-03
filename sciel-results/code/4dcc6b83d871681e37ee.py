import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 42 (2 blocks of j * 3 blocks of k * 7 Strassen terms)
    rank = 42
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # P1 = A00(B00 - B11)
    # P2 = (A00 + A01)B11
    # P3 = (A10 + A11)B00
    # P4 = A11(B01 - B10)
    # P5 = (A00 + A11)(B00 + B11)
    # P6 = (A01 - A11)(B10 + B11)
    # P7 = (A00 - A10)(B00 + B01)
    
    # Indices for local 2x2x2 tensor:
    # A: i_local in {0,1}, j_local in {0,1} -> idx = i_local * 2 + j_local
    # B: j_local in {0,1}, k_local in {0,1} -> idx = j_local * 2 + k_local
    # C: k_local in {0,1}, i_local in {0,1} -> idx = k_local * 2 + i_local
    
    # Strassen factors (U_local, V_local, W_local) for each term
    # Term 1: A00(B00 - B11)
    # U: [1, 0, 0, 0] (A00 is 0)
    # V: [1, 0, 0, -1] (B00 is 0, B11 is 3)
    # W: C00+=1, C01+=0, C10+=0, C11+=1 -> W indices: C00(0)=1, C11(1)=1? 
    # Wait, C index in local is k*2+i.
    # C00: k=0, i=0 -> idx 0. C11: k=1, i=1 -> idx 3.
    # P1 contributes to C00 and C11.
    # C00 = P1 - P2 + P5 + P7
    # C01 = P2 + P4
    # C10 = P3 + P6
    # C11 = P1 - P3 + P5 - P6
    
    strassen_terms = []
    
    # T1: P1
    # U: A00=1, A01=0, A10=0, A11=0 -> [1,0,0,0]
    # V: B00=1, B01=0, B10=0, B11=-1 -> [1,0,0,-1]
    # W: C00+=1, C01+=0, C10+=0, C11+=1 -> [1,0,0,1] (indices 0,1,2,3)
    strassen_terms.append(([1,0,0,0], [1,0,0,-1], [1,0,0,1]))
    
    # T2: P2
    # U: A00=1, A01=1, A10=0, A11=0 -> [1,1,0,0]
    # V: B00=0, B01=0, B10=0, B11=1 -> [0,0,0,1]
    # W: C00+=-1, C01+=1, C10+=0, C11+=0 -> [-1,1,0,0]
    strassen_terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    
    # T3: P3
    # U: A00=0, A01=0, A10=1, A11=1 -> [0,0,1,1]
    # V: B00=1, B01=0, B10=0, B11=0 -> [1,0,0,0]
    # W: C00+=0, C01+=0, C10+=1, C11+=-1 -> [0,0,1,-1]
    strassen_terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    
    # T4: P4
    # U: A00=0, A01=0, A10=0, A11=1 -> [0,0,0,1]
    # V: B00=0, B01=1, B10=-1, B11=0 -> [0,1,-1,0]
    # W: C00+=0, C01+=1, C10+=0, C11+=0 -> [0,1,0,0]
    strassen_terms.append(([0,0,0,1], [0,1,-1,0], [0,1,0,0]))
    
    # T5: P5
    # U: A00=1, A01=0, A10=0, A11=1 -> [1,0,0,1]
    # V: B00=1, B01=0, B10=0, B11=1 -> [1,0,0,1]
    # W: C00+=1, C01+=0, C10+=0, C11+=1 -> [1,0,0,1]
    strassen_terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # T6: P6
    # U: A00=0, A01=1, A10=0, A11=-1 -> [0,1,0,-1]
    # V: B00=0, B01=0, B10=1, B11=1 -> [0,0,1,1]
    # W: C00+=0, C01+=0, C10+=1, C11+=-1 -> [0,0,1,-1]
    # Wait, C10 = P3 + P6. C11 = P1 - P3 + P5 - P6.
    # So P6 contributes +1 to C10 and -1 to C11.
    # Local indices: C10 is k=0, i=1 -> idx 2. C11 is k=1, i=1 -> idx 3.
    # So W is [0,0,1,-1]. Correct.
    strassen_terms.append(([0,1,0,-1], [0,0,1,1], [0,0,1,-1]))
    
    # T7: P7
    # U: A00=1, A01=0, A10=-1, A11=0 -> [1,0,-1,0]
    # V: B00=1, B01=1, B10=0, B11=0 -> [1,1,0,0]
    # W: C00+=1, C01+=0, C10+=0, C11+=0 -> [1,0,0,0]
    strassen_terms.append(([1,0,-1,0], [1,1,0,0], [1,0,0,0]))
    
    # Loop over blocks
    # j_block in {0, 1} corresponding to j indices {0,1} and {2,3}
    # k_block in {0, 1, 2} corresponding to k indices {0,1}, {2,3}, {4,5}
    
    col = 0
    for j_block in range(2):
        j_base = j_block * 2
        
        for k_block in range(3):
            k_base = k_block * 2
            
            for u_loc, v_loc, w_loc in strassen_terms:
                # Fill U
                for i_loc in range(2):
                    for j_loc in range(2):
                        val = u_loc[i_loc * 2 + j_loc]
                        if val != 0:
                            i_global = i_loc
                            j_global = j_base + j_loc
                            idx_u = i_global * m + j_global
                            U[idx_u][col] += val
                
                # Fill V
                for j_loc in range(2):
                    for k_loc in range(2):
                        val = v_loc[j_loc * 2 + k_loc]
                        if val != 0:
                            j_global = j_base + j_loc
                            k_global = k_base + k_loc
                            idx_v = j_global * p + k_global
                            V[idx_v][col] += val
                
                # Fill W
                # Note: W index is k*n + i.
                # In local tensor, C index was k_loc * 2 + i_loc.
                # So w_loc index corresponds to k_loc * 2 + i_loc.
                for k_loc in range(2):
                    for i_loc in range(2):
                        val = w_loc[k_loc * 2 + i_loc]
                        if val != 0:
                            k_global = k_base + k_loc
                            i_global = i_loc
                            idx_w = k_global * n + i_global
                            W[idx_w][col] += val
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
