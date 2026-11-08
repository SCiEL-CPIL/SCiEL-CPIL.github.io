import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition manually
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
    # Indices:
    # U: (n*m, rank) -> (9, rank)
    # V: (m*p, rank) -> (21, rank)
    # W: (p*n, rank) -> (21, rank)
    
    # We split A (3x3) into:
    # A_LL (2x2), A_LR (2x1)
    # A_RL (1x2), A_RR (1x1)
    # B (3x7) into:
    # B_LL (2x7)
    # B_RL (1x7)
    
    # C (3x7) into:
    # C_LL (2x7)
    # C_RL (1x7)
    
    # C_LL = A_LL B_LL + A_LR B_RL
    # C_RL = A_RL B_LL + A_RR B_RL
    
    # 1. A_LL B_LL (2x2x7)
    # Split 7 cols into 2, 2, 2, 1
    # Block 1: 2x2x2 (Strassen)
    # Block 2: 2x2x2 (Strassen)
    # Block 3: 2x2x2 (Strassen)
    # Block 4: 2x2x1 (Schoolbook)
    
    def add_strassen_222(base_u, base_v, base_w, col_offset):
        # base_u, base_v, base_w are offsets in the global indices
        # u_idx = i*m + j
        # v_idx = j*p + k
        # w_idx = k*n + i
        
        # A_LL is rows 0,1 cols 0,1 of A
        # B_LL is rows 0,1 cols 0..6 of B
        
        # Strassen M1..M7
        # M1 = (A00+A11)(B00+B11)
        # u: A00, A11
        # v: B00, B11
        # w: C00, C11
        
        # We need to map local 2x2 indices to global 3x3 indices.
        # Local (i,j) -> Global (i, j) for A_LL
        # Local (j,k) -> Global (j, k) for B_LL (k is 0..1 for this block)
        # Local (k,i) -> Global (k, i) for C_LL (k is 0..1 for this block)
        
        # col_offset is the column index in B (0..6)
        # So B col k_local corresponds to B col col_offset + k_local
        
        # M1
        # A00+A11: u[0,0]=1, u[1,1]=1
        # B00+B11: v[0,0]=1, v[1,1]=1
        # C00, C11: w[0,0]=1, w[1,1]=1
        # Wait, W is (p*n, rank). w_idx = k*n + i.
        # C_LL is top-left 2x2 of C? No, C is 3x7.
        # C_LL is rows 0,1 cols 0..6.
        # So C_LL[k, i] is C[k, i]? No.
        # C is 3x7. C_LL is 2x7.
        # C_LL[i, k] is C[i, k].
        # W index is k*n + i.
        # So for C_LL[i, k], w_idx = k*3 + i.
        
        # Let's define the terms for Strassen 2x2x2
        # Local A indices (0,0), (1,1)
        # Local B indices (0,0), (1,1)
        # Local C indices (0,0), (1,1)
        
        # Term 1: (A00+A11)(B00+B11) -> C00, C11
        # u: [1, 1, 0, 0] (A00, A11)
        # v: [1, 0, 1, 0] (B00, B11) -- wait, B is 2x2. B00 is (0,0), B11 is (1,1).
        # v_idx = j*p + k.
        # B00: j=0, k=0 -> 0*p+0 = 0
        # B11: j=1, k=1 -> 1*p+1 = p+1
        # w: C00: k=0, i=0 -> 0*n+0 = 0
        # C11: k=1, i=1 -> 1*n+1 = n+1
        
        # We need to add 7 terms.
        
        # M1
        u = [0]*9
        u[0] = 1 # A00
        u[3] = 1 # A11
        v = [0]*21
        v[0] = 1 # B00
        v[p+1] = 1 # B11
        w = [0]*21
        w[0] = 1 # C00
        w[n+1] = 1 # C11
        add_term(u, v, w)
        
        # M2: (A10+A11)B00 -> C10, C11
        u = [0]*9
        u[2] = 1 # A10
        u[3] = 1 # A11
        v = [0]*21
        v[0] = 1 # B00
        w = [0]*21
        w[n] = 1 # C10 (k=0, i=1)
        w[n+1] = 1 # C11
        add_term(u, v, w)
        
        # M3: A00(B10-B11) -> C00, C01
        u = [0]*9
        u[0] = 1 # A00
        v = [0]*21
        v[p] = 1 # B10 (j=1, k=0)
        v[p+1] = -1 # B11
        w = [0]*21
        w[0] = 1 # C00
        w[1] = 1 # C01 (k=1, i=0)
        add_term(u, v, w)
        
        # M4: A11(B01-B00) -> C10, C11
        u = [0]*9
        u[3] = 1 # A11
        v = [0]*21
        v[1] = 1 # B01 (j=0, k=1)
        v[0] = -1 # B00
        w = [0]*21
        w[n] = 1 # C10
        w[n+1] = 1 # C11
        add_term(u, v, w)
        
        # M5: (A00+A01)B11 -> C01, C11
        u = [0]*9
        u[0] = 1 # A00
        u[1] = 1 # A01
        v = [0]*21
        v[p+1] = 1 # B11
        w = [0]*21
        w[1] = 1 # C01
        w[n+1] = 1 # C11
        add_term(u, v, w)
        
        # M6: (A10-A00)(B00+B01) -> C10, C11
        u = [0]*9
        u[2] = 1 # A10
        u[0] = -1 # A00
        v = [0]*21
        v[0] = 1 # B00
        v[1] = 1 # B01
        w = [0]*21
        w[n] = 1 # C10
        w[n+1] = 1 # C11
        add_term(u, v, w)
        
        # M7: (A01-A11)(B10+B11) -> C00, C01
        u = [0]*9
        u[1] = 1 # A01
        u[3] = -1 # A11
        v = [0]*21
        v[p] = 1 # B10
        v[p+1] = 1 # B11
        w = [0]*21
        w[0] = 1 # C00
        w[1] = 1 # C01
        add_term(u, v, w)
        
    # We need to call add_strassen_222 for col_offset 0, 2, 4
    # But the indices in v and w depend on col_offset.
    # The helper above assumes col_offset 0.
    # We need to shift v and w indices.
    
    # Let's rewrite helper to take col_offset
    def add_strassen_222_shifted(col_offset):
        # v_idx = j*p + (k + col_offset)
        # w_idx = (k + col_offset)*n + i
        
        # M1
        u = [0]*9
        u[0] = 1; u[3] = 1
        v = [0]*21
        v[0*p + col_offset] = 1
        v[1*p + 1 + col_offset] = 1
        w = [0]*21
        w[col_offset*n + 0] = 1
        w[(1+col_offset)*n + 1] = 1
        add_term(u, v, w)
        
        # M2
        u = [0]*9
        u[2] = 1; u[3] = 1
        v = [0]*21
        v[0*p + col_offset] = 1
        w = [0]*21
        w[col_offset*n + 1] = 1
        w[(1+col_offset)*n + 1] = 1
        add_term(u, v, w)
        
        # M3
        u = [0]*9
        u[0] = 1
        v = [0]*21
        v[1*p + col_offset] = 1
        v[1*p + 1 + col_offset] = -1
        w = [0]*21
        w[col_offset*n + 0] = 1
        w[(1+col_offset)*n + 0] = 1
        add_term(u, v, w)
        
        # M4
        u = [0]*9
        u[3] = 1
        v = [0]*21
        v[0*p + 1 + col_offset] = 1
        v[0*p + col_offset] = -1
        w = [0]*21
        w[col_offset*n + 1] = 1
        w[(1+col_offset)*n + 1] = 1
        add_term(u, v, w)
        
        # M5
        u = [0]*9
        u[0] = 1; u[1] = 1
        v = [0]*21
        v[1*p + 1 + col_offset] = 1
        w = [0]*21
        w[(1+col_offset)*n + 0] = 1
        w[(1+col_offset)*n + 1] = 1
        add_term(u, v, w)
        
        # M6
        u = [0]*9
        u[2] = 1; u[0] = -1
        v = [0]*21
        v[0*p + col_offset] = 1
        v[0*p + 1 + col_offset] = 1
        w = [0]*21
        w[col_offset*n + 1] = 1
        w[(1+col_offset)*n + 1] = 1
        add_term(u, v, w)
        
        # M7
        u = [0]*9
        u[1] = 1; u[3] = -1
        v = [0]*21
        v[1*p + col_offset] = 1
        v[1*p + 1 + col_offset] = 1
        w = [0]*21
        w[col_offset*n + 0] = 1
        w[(1+col_offset)*n + 0] = 1
        add_term(u, v, w)

    add_strassen_222_shifted(0)
    add_strassen_222_shifted(2)
    add_strassen_222_shifted(4)
    
    # Block 4: 2x2x1 (col 6)
    # Schoolbook 4 terms
    # C00 = A00 B00
    u = [0]*9; u[0]=1
    v = [0]*21; v[0*p+6]=1
    w = [0]*21; w[6*n+0]=1
    add_term(u, v, w)
    
    # C01 = A01 B10 (B col 6 is B10? No, B is 2x1. B00, B10)
    # B_LL is 2x7. Col 6 is the 7th col.
    # B_LL[0, 6] and B_LL[1, 6].
    # C_LL[0, 6] = A00 B00 + A01 B10
    # C_LL[1, 6] = A10 B00 + A11 B10
    
    # Term 2: A01 B10
    u = [0]*9; u[1]=1
    v = [0]*21; v[1*p+6]=1
    w = [0]*21; w[6*n+0]=1
    add_term(u, v, w)
    
    # Term 3: A10 B00
    u = [0]*9; u[2]=1
    v = [0]*21; v[0*p+6]=1
    w = [0]*21; w[6*n+1]=1
    add_term(u, v, w)
    
    # Term 4: A11 B10
    u = [0]*9; u[3]=1
    v = [0]*21; v[1*p+6]=1
    w = [0]*21; w[6*n+1]=1
    add_term(u, v, w)
    
    # 2. A_LR B_RL (2x1x7)
    # A_LR is 2x1 (rows 0,1 col 2 of A)
    # B_RL is 1x7 (row 2 of B)
    # C_LL is 2x7
    # C_LL[i, k] = A_LR[i, 0] B_RL[0, k]
    # 7 terms
    for k in range(7):
        u = [0]*9
        u[0*m + 2] = 1 # A[0,2]
        u[1*m + 2] = 1 # A[1,2]
        v = [0]*21
        v[2*p + k] = 1 # B[2,k]
        w = [0]*21
        w[k*n + 0] = 1 # C[k,0]
        w[k*n + 1] = 1 # C[k,1]
        add_term(u, v, w)
        
    # 3. A_RL B_LL (1x2x7)
    # A_RL is 1x2 (row 2 cols 0,1 of A)
    # B_LL is 2x7
    # C_RL is 1x7
    # C_RL[0, k] = A_RL[0,0] B_LL[0,k] + A_RL[0,1] B_LL[1,k]
    # 14 terms?
    # C_RL[0, k] = A[2,0] B[0,k] + A[2,1] B[1,k]
    for k in range(7):
        u = [0]*9; u[2*m+0]=1
        v = [0]*21; v[0*p+k]=1
        w = [0]*21; w[k*n+2]=1
        add_term(u, v, w)
        
        u = [0]*9; u[2*m+1]=1
        v = [0]*21; v[1*p+k]=1
        w = [0]*21; w[k*n+2]=1
        add_term(u, v, w)
        
    # 4. A_RR B_RL (1x1x7)
    # A_RR is 1x1 (A[2,2])
    # B_RL is 1x7
    # C_RL[0, k] = A[2,2] B[2,k]
    for k in range(7):
        u = [0]*9; u[2*m+2]=1
        v = [0]*21; v[2*p+k]=1
        w = [0]*21; w[k*n+2]=1
        add_term(u, v, w)
        
    # Convert lists to matrices
    rank = len(U_list)
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U, "V": V, "W": W}
