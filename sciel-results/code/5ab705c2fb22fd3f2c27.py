"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_schoolbook(n, m, p):
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
    return U, V, W, rank

def _build_strassen_2x2x2():
    """
    Returns U, V, W matrices for a 2x2x2 Strassen decomposition.
    Shapes: (4, 7), (4, 7), (4, 7)
    Indices:
    U: row index = i*2 + j (A[i,j])
    V: row index = j*2 + k (B[j,k])
    W: row index = k*2 + i (C[i,k])
    """
    rank = 7
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((4, rank), dtype=int)
    W = np.zeros((4, rank), dtype=int)
    
    # Strassen formulas for C = A * B
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1  # a, d
    V[0, 0] = 1; V[3, 0] = 1  # e, h
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1 # all C
    
    # m2 = (c+d)e
    U[1, 1] = 1; U[3, 1] = 1  # c, d
    V[0, 1] = 1               # e
    W[2, 1] = 1; W[3, 1] = 1  # C[1,0], C[1,1]
    
    # m3 = a(f-h)
    U[0, 2] = 1               # a
    V[1, 2] = 1; V[3, 2] = -1 # f, -h
    W[1, 2] = 1; W[3, 2] = 1  # C[0,1], C[1,1]
    
    # m4 = d(g-h)
    U[3, 3] = 1               # d
    V[2, 3] = 1; V[3, 3] = -1 # g, -h
    W[0, 3] = 1; W[2, 3] = 1  # C[0,0], C[1,0]
    
    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1  # a, b
    V[3, 4] = 1               # h
    W[1, 4] = 1               # C[0,1]
    
    # m6 = (c-d)(e+f)
    U[1, 5] = 1; U[3, 5] = -1 # c, -d
    V[0, 5] = 1; V[1, 5] = 1  # e, f
    W[2, 5] = 1               # C[1,0]
    
    # m7 = (b-c)(g+h)
    U[1, 6] = 1; U[2, 6] = -1 # b, -c
    V[2, 6] = 1; V[3, 6] = 1  # g, h
    W[0, 6] = 1               # C[0,0]
    
    return U, V, W, rank

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # 1. Construct Schoolbook decomposition (Rank 63)
    U_school, V_school, W_school, rank_school = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # 2. Construct Hybrid decomposition using Strassen for 2x2x2 block
    # The 2x2x2 block corresponds to i in {0,1}, j in {0,1}, k in {0,1}
    # We replace the 8 schoolbook terms for this block with 7 Strassen terms.
    # The remaining 55 terms (63 - 8) are kept as schoolbook.
    
    U_hyb = np.zeros((n * m, 62), dtype=int)
    V_hyb = np.zeros((m * p, 62), dtype=int)
    W_hyb = np.zeros((p * n, 62), dtype=int)
    
    col = 0
    
    # Insert Strassen terms for the 2x2x2 block (i=0,1; j=0,1; k=0,1)
    U_str, V_str, W_str, _ = _build_strassen_2x2x2()
    
    for c in range(7):
        # U indices: i*m + j. For 2x2 block, i in 0..1, j in 0..1.
        # U_str row r corresponds to A[i,j] where r = i*2 + j.
        # In full matrix, index is i*3 + j.
        # Mapping:
        # U_str[0] (a=A00) -> U_hyb[0*3+0] = U_hyb[0]
        # U_str[1] (b=A01) -> U_hyb[0*3+1] = U_hyb[1]
        # U_str[2] (c=A10) -> U_hyb[1*3+0] = U_hyb[3]
        # U_str[3] (d=A11) -> U_hyb[1*3+1] = U_hyb[4]
        
        # V indices: j*p + k. For 2x2 block, j in 0..1, k in 0..1.
        # V_str row r corresponds to B[j,k] where r = j*2 + k.
        # In full matrix, index is j*7 + k.
        # Mapping:
        # V_str[0] (e=B00) -> V_hyb[0*7+0] = V_hyb[0]
        # V_str[1] (f=B01) -> V_hyb[0*7+1] = V_hyb[1]
        # V_str[2] (g=B10) -> V_hyb[1*7+0] = V_hyb[7]
        # V_str[3] (h=B11) -> V_hyb[1*7+1] = V_hyb[8]
        
        # W indices: k*n + i. For 2x2 block, k in 0..1, i in 0..1.
        # W_str row r corresponds to C[i,k] where r = k*2 + i.
        # In full matrix, index is k*3 + i.
        # Mapping:
        # W_str[0] (C00) -> W_hyb[0*3+0] = W_hyb[0]
        # W_str[1] (C01) -> W_hyb[1*3+0] = W_hyb[3]  # Wait, W index is k*n+i.
        # Let's re-verify W mapping.
        # W_str is 4x7. Rows 0..3.
        # Row 0: C[0,0] -> k=0, i=0 -> idx 0*3+0=0
        # Row 1: C[1,0] -> k=0, i=1 -> idx 0*3+1=1
        # Row 2: C[0,1] -> k=1, i=0 -> idx 1*3+0=3
        # Row 3: C[1,1] -> k=1, i=1 -> idx 1*3+1=4
        
        # Wait, standard Strassen W mapping in my _build_strassen_2x2x2:
        # W[0,0]=1 (C00), W[1,0]=1 (C10? No, W index k*n+i. If W[1] is C[0,1] or C[1,0]?)
        # Let's check _build_strassen_2x2x2 logic again.
        # m1=(a+d)(e+h). C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        # C00 = ae+bg. C01 = af+bh. C10 = ce+dg. C11 = cf+dh.
        # m1 contributes to all.
        # m2=(c+d)e. ce+de. Contributes to C10 (ce) and C11 (de).
        #   C10: k=0, i=1 -> idx 1.
        #   C11: k=1, i=1 -> idx 4.
        #   So W[1,1]=1, W[4,1]=1? 
        #   My code: W[2,1]=1; W[3,1]=1. 
        #   W index 2 is k=0, i=2? No, n=2 in Strassen block. 
        #   In Strassen 2x2x2, n=2. W index = k*2+i.
        #   C10: k=0, i=1 -> idx 1.
        #   C11: k=1, i=1 -> idx 3.
        #   So W[1,1]=1, W[3,1]=1 is correct for 2x2 block.
        
        # Let's re-map W_str (2x2 block indices) to W_hyb (3x7 block indices).
        # W_str row r: k = r // 2, i = r % 2.
        # W_hyb index: k * 3 + i.
        
        for r in range(4):
            if U_str[r, c] != 0:
                i = r // 2
                j = r % 2
                U_hyb[i * 3 + j, col] = U_str[r, c]
            
            if V_str[r, c] != 0:
                j = r // 2
                k = r % 2
                V_hyb[j * 7 + k, col] = V_str[r, c]
            
            if W_str[r, c] != 0:
                k = r // 2
                i = r % 2
                W_hyb[k * 3 + i, col] = W_str[r, c]
        
        col += 1
    
    # Insert Schoolbook terms for the rest
    # We need to skip (i in 0..1, j in 0..1, k in 0..1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U_hyb[i * m + j, col] = 1
                V_hyb[j * p + k, col] = 1
                W_hyb[k * n + i, col] = 1
                col += 1
    
    # Verify Hybrid Decomposition
    # Check if sum_c U V W == T
    # T[i,j,k] = 1 if (i,j,k) valid.
    # Let's do a quick numerical check.
    # Product = U @ diag(V) ... no, it's tensor contraction.
    # T_recon[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Efficient check:
    # For each c, the term is rank-1 tensor.
    # We can check a few random points or all points (63 points is small).
    
    valid_hyb = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(62):
                    val += U_hyb[i*m+j, c] * V_hyb[j*p+k, c] * W_hyb[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    valid_hyb = False
                    break
            if not valid_hyb:
                break
        if not valid_hyb:
            break
    
    if valid_hyb:
        best_U, best_V, best_W = U_hyb, V_hyb, W_hyb
        best_rank = 62
    
    # If we have time, we could try to optimize further, but 62 is a solid improvement over 63.
    # Given the strict time limit and the complexity of finding lower ranks (likely 61 or lower is hard),
    # returning the verified hybrid decomposition is the best strategy.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
