import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=4
    
    # Strategy: Split B into two 3x2 blocks B_L and B_R.
    # C_L = A B_L (2x2 result)
    # C_R = A B_R (2x2 result)
    
    # Decompose A into A_L (2x2, cols 0,1) and a_R (2x1, col 2)
    # Decompose B_L into B_L1 (2x2, rows 0,1) and b_L2 (1x2, row 2)
    # C_L = A_L B_L1 + a_R b_L2
    
    # Rank of A_L B_L1 (2x2x2) is 7 (Strassen)
    # Rank of a_R b_L2 (2x1x2) is 2
    # Total rank for C_L is 9.
    # Total rank for C_R is 9.
    # Total rank 18.
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)] # 2*3 = 6 rows
    V = [[0] * rank for _ in range(m * p)] # 3*4 = 12 rows
    W = [[0] * rank for _ in range(p * n)] # 4*2 = 8 rows
    
    col = 0
    
    # Helper to add Strassen 2x2 decomposition
    # A indices: (0,0), (0,1), (1,0), (1,1) in U
    # B indices: (0,0), (0,1), (1,0), (1,1) in V
    # C indices: (0,0), (0,1), (1,0), (1,1) in W
    # Note: W index is k*n + i.
    # C is 2x2. i in 0..1, k in 0..1.
    # W[0*2+0]=C00, W[1*2+0]=C10? No.
    # W[k*n+i].
    # k=0, i=0 -> W[0] -> C[0,0]
    # k=1, i=0 -> W[2] -> C[0,1]
    # k=0, i=1 -> W[1] -> C[1,0]
    # k=1, i=1 -> W[3] -> C[1,1]
    
    def add_strassen(u_base, v_base, w_base, offset):
        # u_base: index of A00 in U
        # v_base: index of B00 in V
        # w_base: index of C00 in W (k=0, i=0)
        
        # Strassen products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Coefficients for U (A)
        # A00, A01, A10, A11
        u00 = u_base + 0 # i=0, j=0
        u01 = u_base + 1 # i=0, j=1
        u10 = u_base + 2 # i=1, j=0
        u11 = u_base + 3 # i=1, j=1
        
        # Coefficients for V (B)
        # B is 2x2. V index is j*p + k.
        # Here B is 2x2, so p=2.
        # v00 = j=0, k=0 -> 0
        # v01 = j=0, k=1 -> 1
        # v10 = j=1, k=0 -> 2
        # v11 = j=1, k=1 -> 3
        v00 = v_base + 0
        v01 = v_base + 1
        v10 = v_base + 2
        v11 = v_base + 3
        
        # Coefficients for W (C)
        # C is 2x2. W index is k*n + i. n=2.
        # w00 = k=0, i=0 -> 0
        # w10 = k=0, i=1 -> 1  (C[1,0])
        # w01 = k=1, i=0 -> 2  (C[0,1])
        # w11 = k=1, i=1 -> 3  (C[1,1])
        w00 = w_base + 0
        w10 = w_base + 1
        w01 = w_base + 2
        w11 = w_base + 3
        
        # M1
        U[u00][offset] += 1
        U[u11][offset] += 1
        V[v00][offset] += 1
        V[v11][offset] += 1
        W[w00][offset] += 1
        W[w11][offset] += 1
        W[w10][offset] -= 1
        W[w01][offset] -= 1
        
        # M2
        U[u10][offset] += 1
        U[u11][offset] += 1
        V[v00][offset] += 1
        W[w10][offset] += 1
        W[w11][offset] += 1
        
        # M3
        U[u00][offset] += 1
        V[v10][offset] += 1
        V[v11][offset] -= 1
        W[w01][offset] += 1
        W[w11][offset] += 1
        
        # M4
        U[u11][offset] += 1
        V[v01][offset] += 1
        V[v00][offset] -= 1
        W[w00][offset] += 1
        W[w10][offset] += 1
        
        # M5
        U[u00][offset] += 1
        U[u01][offset] += 1
        V[v11][offset] += 1
        W[w00][offset] += 1
        W[w01][offset] += 1
        
        # M6
        U[u10][offset] += 1
        U[u00][offset] -= 1
        V[v00][offset] += 1
        V[v01][offset] += 1
        W[w10][offset] += 1
        W[w01][offset] += 1
        
        # M7
        U[u01][offset] += 1
        U[u11][offset] -= 1
        V[v10][offset] += 1
        V[v11][offset] += 1
        W[w00][offset] += 1
        W[w11][offset] += 1
        
        return offset + 7

    def add_2x1_by_1x2(u_base, v_base, w_base, offset):
        # A is 2x1 (u_base, u_base+1)
        # B is 1x2 (v_base, v_base+1)
        # C is 2x2 (w_base, w_base+1, w_base+2, w_base+3)
        
        # C00 = A0 B0
        U[u_base][offset] = 1
        V[v_base][offset] = 1
        W[w_base][offset] = 1
        offset += 1
        
        # C01 = A0 B1
        U[u_base][offset] = 1
        V[v_base+1][offset] = 1
        W[w_base+2][offset] = 1 # k=1, i=0
        offset += 1
        
        # C10 = A1 B0
        U[u_base+1][offset] = 1
        V[v_base][offset] = 1
        W[w_base+1][offset] = 1 # k=0, i=1
        offset += 1
        
        # C11 = A1 B1
        U[u_base+1][offset] = 1
        V[v_base+1][offset] = 1
        W[w_base+3][offset] = 1 # k=1, i=1
        offset += 1
        
        return offset + 4 # 4 mults? No, 2x1 by 1x2 is 2 mults if we share?
        # Wait.
        # C = [[a0*b0, a0*b1], [a1*b0, a1*b1]]
        # This is 4 scalar mults.
        # Can it be done in 2?
        # No.
        # So rank 4.
        
        # My previous estimate of rank 2 for 2x1 by 1x2 was wrong.
        # 2x1 by 1x2 is outer product. Rank 4.
        
        # So C_L rank = 7 + 4 = 11.
        # Total rank = 22.
        
        # Schoolbook is 24.
        # 22 is better.
        
        # Can we do better?
        # 2x3 by 3x2.
        # A = [A_L, a_R].
        # B = [B_L1; b_L2].
        # C = A_L B_L1 + a_R b_L2.
        
        # Maybe there is a rank 10 for 2x3 by 3x2?
        
        # For now, 22 is safe.
        
        pass

    # Let's just use schoolbook for the cross terms if Strassen is too complex to verify.
    # Or just use the 22 rank solution.
    
    # Re-evaluating 2x1 by 1x2.
    # It is 4 mults.
    
    # So Rank 22.
    
    # Let's implement Rank 22.
    
    col = 0
    
    # Left part: B cols 0,1
    # A_L (2x2): U rows 0,1,2,3?
    # U index i*m+j.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 2
    # i=1, j=1 -> 3
    # a_R (2x1): i=0, j=2 -> 4; i=1, j=2 -> 5
    
    # B_L1 (2x2): V rows 0,1,2,3?
    # V index j*p+k. p=4.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 4
    # j=1, k=1 -> 5
    
    # b_L2 (1x2): j=2, k=0 -> 8; j=2, k=1 -> 9
    
    # C_L (2x2): W rows
    # k=0, i=0 -> 0
    # k=1, i=0 -> 2
    # k=0, i=1 -> 1
    # k=1, i=1 -> 3
    
    # Strassen for A_L B_L1
    # U base 0, V base 0, W base 0
    col = add_strassen(0, 0, 0, col)
    
    # 2x1 by 1x2 for a_R b_L2
    # U base 4, V base 8, W base 0 (same C_L)
    # But we need to add to existing W.
    # The add_2x1_by_1x2 function overwrites W.
    # We need to modify it to add.
    
    def add_2x1_by_1x2_add(u_base, v_base, w_base, offset):
        # C00
        U[u_base][offset] += 1
        V[v_base][offset] += 1
        W[w_base][offset] += 1
        offset += 1
        
        # C01
        U[u_base][offset] += 1
        V[v_base+1][offset] += 1
        W[w_base+2][offset] += 1
        offset += 1
        
        # C10
        U[u_base+1][offset] += 1
        V[v_base][offset] += 1
        W[w_base+1][offset] += 1
        offset += 1
        
        # C11
        U[u_base+1][offset] += 1
        V[v_base+1][offset] += 1
        W[w_base+3][offset] += 1
        offset += 1
        
        return offset + 4

    col = add_2x1_by_1x2_add(4, 8, 0, col)
    
    # Right part: B cols 2,3
    # B_R1 (2x2): j=0, k=2 -> 2; j=0, k=3 -> 3; j=1, k=2 -> 6; j=1, k=3 -> 7
    # b_R2 (1x2): j=2, k=2 -> 10; j=2, k=3 -> 11
    
    # C_R (2x2): W rows
    # k=2, i=0 -> 4
    # k=3, i=0 -> 6
    # k=2, i=1 -> 5
    # k=3, i=1 -> 7
    
    col = add_strassen(0, 2, 4, col)
    col = add_2x1_by_1x2_add(4, 10, 4, col)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
