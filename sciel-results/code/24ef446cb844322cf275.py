import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (15, rank)
    # W: (p*n, rank) = (10, rank)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x2(i_offset, j_offset, k_offset):
        # Adds 7 terms for i in {i_offset, i_offset+1}, j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
        # Global indices:
        # i is 0 or 1 (since n=2, offset must be 0)
        # j is 0,1 or 1,2? 
        # In our case, we only use j_offset=0 for Strassen.
        # k_offset can be 0 or 3.
        nonlocal col
        # Strassen multipliers M1..M7
        # M1 = (a00+a11)(b00+b11) -- Note: indices are local 0,1
        # We need to map local (i_loc, j_loc, k_loc) to global
        # i_glob = i_offset + i_loc
        # j_glob = j_offset + j_loc
        # k_glob = k_offset + k_loc
        
        # M1
        U[i_offset*3 + j_offset][col] = 1
        U[i_offset*3 + j_offset+1][col] = 1
        V[j_offset*5 + k_offset][col] = 1
        V[j_offset*5 + k_offset+1][col] = 1
        W[k_offset*2 + i_offset][col] = 1
        W[k_offset*2 + i_offset+1][col] = 1
        W[k_offset*2 + i_offset][col] += 0 # No, W index is k*n+i.
        # W[k_offset*2 + i_offset] corresponds to c_{i_offset, k_offset}? No.
        # W index is k*n + i.
        # For M1, it contributes to C_{00}, C_{01}, C_{10}, C_{11}.
        # C_{00} (i=0, k=0): W[0*2+0] = W[0]
        # C_{01} (i=0, k=1): W[1*2+0] = W[2]
        # C_{10} (i=1, k=0): W[0*2+1] = W[1]
        # C_{11} (i=1, k=1): W[1*2+1] = W[3]
        
        # Let's re-verify Strassen coefficients for W.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We will define the 7 terms explicitly.
        
        # Term 1: M1 = (a00+a11)(b00+b11)
        # a00: U[i0, j0]=1, a11: U[i1, j1]=1
        # b00: V[j0, k0]=1, b11: V[j1, k1]=1
        # W: C00+=1, C01+=1, C10+=1, C11+=1
        U[i_offset*3 + j_offset][col] = 1
        U[i_offset*3 + j_offset+1][col] = 1
        V[j_offset*5 + k_offset][col] = 1
        V[j_offset*5 + k_offset+1][col] = 1
        W[k_offset*2 + i_offset][col] = 1
        W[(k_offset+1)*2 + i_offset][col] = 1
        W[k_offset*2 + i_offset+1][col] = 1
        W[(k_offset+1)*2 + i_offset+1][col] = 1
        col += 1
        
        # Term 2: M2 = (a10+a11)b00
        # a10: U[i1, j0]=1, a11: U[i1, j1]=1
        # b00: V[j0, k0]=1
        # W: C10+=1
        U[i_offset*3 + j_offset][col] = 1
        U[i_offset*3 + j_offset+1][col] = 1
        V[j_offset*5 + k_offset][col] = 1
        W[k_offset*2 + i_offset+1][col] = 1
        col += 1
        
        # Term 3: M3 = a00(b01-b11)
        # a00: U[i0, j0]=1
        # b01: V[j0, k1]=1, b11: V[j1, k1]=-1
        # W: C01+=1
        U[i_offset*3 + j_offset][col] = 1
        V[j_offset*5 + k_offset+1][col] = 1
        V[j_offset+1*5 + k_offset+1][col] = -1
        W[(k_offset+1)*2 + i_offset][col] = 1
        col += 1
        
        # Term 4: M4 = a11(b10-b00)
        # a11: U[i1, j1]=1
        # b10: V[j1, k0]=1, b00: V[j0, k0]=-1
        # W: C00+=1, C10+=1
        U[i_offset*3 + j_offset+1][col] = 1
        V[j_offset+1*5 + k_offset][col] = 1
        V[j_offset*5 + k_offset][col] = -1
        W[k_offset*2 + i_offset][col] = 1
        W[k_offset*2 + i_offset+1][col] = 1
        col += 1
        
        # Term 5: M5 = (a00+a01)b11
        # a00: U[i0, j0]=1, a01: U[i0, j1]=1
        # b11: V[j1, k1]=1
        # W: C00-=1, C01+=1
        U[i_offset*3 + j_offset][col] = 1
        U[i_offset*3 + j_offset+1][col] = 1
        V[j_offset+1*5 + k_offset+1][col] = 1
        W[k_offset*2 + i_offset][col] = -1
        W[(k_offset+1)*2 + i_offset][col] = 1
        col += 1
        
        # Term 6: M6 = (a10-a00)(b00+b01)
        # a10: U[i1, j0]=1, a00: U[i0, j0]=-1
        # b00: V[j0, k0]=1, b01: V[j0, k1]=1
        # W: C11+=1
        U[i_offset*3 + j_offset][col] = -1
        U[i_offset*3 + j_offset+1][col] = 1
        V[j_offset*5 + k_offset][col] = 1
        V[j_offset*5 + k_offset+1][col] = 1
        W[(k_offset+1)*2 + i_offset+1][col] = 1
        col += 1
        
        # Term 7: M7 = (a01-a11)(b10+b11)
        # a01: U[i0, j1]=1, a11: U[i1, j1]=-1
        # b10: V[j1, k0]=1, b11: V[j1, k1]=1
        # W: C00+=1, C11+=1
        U[i_offset*3 + j_offset+1][col] = -1
        U[i_offset*3 + j_offset+1][col] = 1 # Wait, U[i0, j1] is a01. U[i1, j1] is a11.
        # Correction:
        U[i_offset*3 + j_offset+1][col] = 1 # a01
        U[i_offset*3 + j_offset+1][col] = -1 # a11 -- Conflict!
        # a01 is at i=0, j=1. Index 0*3+1 = 1.
        # a11 is at i=1, j=1. Index 1*3+1 = 4.
        # So:
        U[i_offset*3 + j_offset+1][col] = 1 # This is U[1] if i_offset=0, j_offset=0. Correct for a01.
        U[i_offset+1*3 + j_offset+1][col] = -1 # This is U[4]. Correct for a11.
        V[j_offset+1*5 + k_offset][col] = 1
        V[j_offset+1*5 + k_offset+1][col] = 1
        W[k_offset*2 + i_offset][col] = 1
        W[(k_offset+1)*2 + i_offset+1][col] = 1
        col += 1
        
        # Reset U[1] for Term 7? 
        # In Term 7, I set U[1]=1 and U[4]=-1.
        # But in the code above for Term 7:
        # U[i_offset*3 + j_offset+1][col] = 1
        # U[i_offset+1*3 + j_offset+1][col] = -1
        # This is correct.
        
    def add_schoolbook_2x1x2(i_offset, j_idx, k_offset):
        # Adds 4 terms for i in {i_offset, i_offset+1}, j=j_idx, k in {k_offset, k_offset+1}
        nonlocal col
        for i_loc in range(2):
            for k_loc in range(2):
                i_glob = i_offset + i_loc
                k_glob = k_offset + k_loc
                U[i_glob*3 + j_idx][col] = 1
                V[j_idx*5 + k_glob][col] = 1
                W[k_glob*2 + i_glob][col] = 1
                col += 1

    def add_schoolbook_2x3x1(i_offset, k_idx):
        # Adds 6 terms for i in {i_offset, i_offset+1}, j in {0,1,2}, k=k_idx
        nonlocal col
        for i_loc in range(2):
            for j_idx in range(3):
                i_glob = i_offset + i_loc
                U[i_glob*3 + j_idx][col] = 1
                V[j_idx*5 + k_idx][col] = 1
                W[k_idx*2 + i_glob][col] = 1
                col += 1

    # Block 1: k in {0, 1, 2}
    # Sub-block 1a: k in {0, 1}
    # j in {0, 1}: Strassen
    add_strassen_2x2x2(0, 0, 0)
    # j=2: Schoolbook
    add_schoolbook_2x1x2(0, 2, 0)
    
    # Sub-block 1b: k=2
    # j in {0, 1, 2}: Schoolbook
    add_schoolbook_2x3x1(0, 2)
    
    # Block 2: k in {3, 4}
    # j in {0, 1}: Strassen
    add_strassen_2x2x2(0, 0, 3)
    # j=2: Schoolbook
    add_schoolbook_2x1x2(0, 2, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
