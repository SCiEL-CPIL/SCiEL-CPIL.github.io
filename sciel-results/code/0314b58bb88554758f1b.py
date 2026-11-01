import numpy as np

def construct():
    n, m, p = 2, 3, 9
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Local indices for 2x3x3 block
    # A: 2x3, flat i*3+j
    # B: 3x3, flat j*3+k
    # C: 3x2, flat k*2+i
    
    def add_local_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col, b):
        # u_idx, v_idx, w_idx are lists of local indices
        # b is the block index (0, 1, 2)
        
        # Map A indices (same for all blocks)
        for idx, val in zip(u_idx, u_val):
            U[idx][col + b*17] = val # Wait, rank is 51, so offset is b*17? No, rank is total.
            # Actually, I should just use a global column counter.
            
    # Let's use a global column counter.
    col = 0
    
    # Define the 17 local terms
    # Term structure: (u_idx, u_val, v_idx, v_val, w_idx, w_val)
    
    # 1. Strassen 2x2x2 for A[0:2, 0:2] and B[0:2, 0:2] -> C[0:2, 0:2]
    # A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B[0,0]=0, B[0,1]=1, B[1,0]=3, B[1,1]=4
    # C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    
    # m1 = (A00+A11)(B00+B11) -> C00, C11
    # m2 = (A10+A11)B00 -> C10
    # m3 = A00(B01-B11) -> C01
    # m4 = A11(B10-B00) -> C10
    # m5 = (A00+A01)B11 -> C01
    # m6 = (A10-A00)(B00+B01) -> C11
    # m7 = (A01-A11)(B10+B11) -> C00, C11
    
    terms = []
    
    # m1
    terms.append(([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]))
    # m2
    terms.append(([3, 4], [1, 1], [0], [1], [1], [1]))
    # m3
    terms.append(([0], [1], [1, 4], [1, -1], [2], [1]))
    # m4
    terms.append(([4], [1], [3, 0], [1, -1], [1], [1]))
    # m5
    terms.append(([0, 1], [1, 1], [4], [1], [2], [1]))
    # m6
    terms.append(([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]))
    # m7
    terms.append(([1, 4], [1, -1], [3, 4], [1, 1], [0, 3], [1, -1]))
    
    # 2. 2x2x1 for A[0:2, 0:2] and B[0:2, 2] -> C[0:2, 2]
    # A[0,0], A[0,1], A[1,0], A[1,1]
    # B[0,2]=2, B[1,2]=5
    # C[0,2]=4, C[1,2]=5
    
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    terms.append(([0], [1], [2], [1], [4], [1]))
    terms.append(([1], [1], [5], [1], [4], [1]))
    terms.append(([3], [1], [2], [1], [5], [1]))
    terms.append(([4], [1], [5], [1], [5], [1]))
    
    # 3. Outer Product A[0:2, 2] and B[2, 0:2] -> C[0:2, 0:2]
    # A[0,2]=2, A[1,2]=5
    # B[2,0]=6, B[2,1]=7
    # C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    
    # C00 += A02*B20
    # C01 += A02*B21
    # C10 += A12*B20
    # C11 += A12*B21
    
    terms.append(([2], [1], [6], [1], [0], [1]))
    terms.append(([2], [1], [7], [1], [2], [1]))
    terms.append(([5], [1], [6], [1], [1], [1]))
    terms.append(([5], [1], [7], [1], [3], [1]))
    
    # Total terms: 7 + 4 + 4 = 15?
    # Wait, Strassen has 7. 2x2x1 has 4. Outer product has 4.
    # 7+4+4 = 15.
    # But I expected 17.
    # Where are the missing 2?
    # Ah, $A_L B_T$ was $2 \times 2 \times 3$.
    # I split $B_T$ into $B_{TL}$ ($2 \times 2$) and $B_{TR}$ ($2 \times 1$).
    # $A_L B_{TL}$ used Strassen (7).
    # $A_L B_{TR}$ used 4 mults.
    # $A_R B_B$ used 6 mults (Outer product $2 \times 1 \times 3$).
    # My Outer product above was $A_R$ ($2 \times 1$) and $B_{BL}$ ($1 \times 2$)?
    # No, $B_B$ is row 2 of B. $B_B$ is $1 \times 3$.
    # I only used cols 0 and 1 of $B_B$ in the "Outer Product" section above?
    # Let's check.
    # $A_R$ is col 2 of A.
    # $B_B$ is row 2 of B.
    # $A_R B_B$ contributes to $C_{0:2, 0:3}$.
    # Cols 0 and 1 are handled above.
    # Col 2 of $B_B$ (i.e. $B_{2,2}$) is NOT handled yet!
    
    # Missing: $A_R$ ($2 \times 1$) and $B_{2,2}$ ($1 \times 1$) -> $C_{0:2, 2}$.
    # C02 += A02*B22
    # C12 += A12*B22
    
    terms.append(([2], [1], [8], [1], [4], [1]))
    terms.append(([5], [1], [8], [1], [5], [1]))
    
    # Now I have 7 + 4 + 4 + 2 = 17 terms.
    
    # Apply to global matrices
    for b in range(3):
        for u_idx, u_val, v_idx, v_val, w_idx, w_val in terms:
            for idx, val in zip(u_idx, u_val):
                U[idx][col] = val
            for idx, val in zip(v_idx, v_val):
                # Global V index: j * 9 + (3*b + k_local)
                # v_idx is local j*3+k.
                # We need to map local (j,k) to global (j, 3b+k)
                j = idx // 3
                k = idx % 3
                global_v = j * 9 + (3 * b + k)
                V[global_v][col] = val
            for idx, val in zip(w_idx, w_val):
                # Global W index: (3*b + k_local) * 2 + i
                # w_idx is local k*2+i.
                # We need to map local (k,i) to global (3b+k, i)
                k = idx // 2
                i = idx % 2
                global_w = (3 * b + k) * 2 + i
                W[global_w][col] = val
            col += 1
            
    return {"U": U, "V": V, "W": W}
