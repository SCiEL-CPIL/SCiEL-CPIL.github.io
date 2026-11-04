import numpy as np
from fractions import Fraction
import sys
import time

DIMENSIONS = [3, 5, 7]
DEADLINE = time.time() + 140

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    # Output: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # W index convention: W[k*n + i, r]
    
    rank_terms = []  # list of (U_vec, V_vec, W_vec), each of length nm, mp, pn respectively
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i

    # 1. 6 Strassen blocks for the 2x2x2 sub-problem (i=0,1; j=0,1; k=0,1)
    # Each block has rank 7, total 42 terms
    for t in range(6):
        Uv = [0.0] * (n * m)
        Vv = [0.0] * (m * p)
        Wv = [0.0] * (p * n)
        
        # Define Strassen terms based on t
        if t == 0:
            # P0 = (A00 + A11)(B00 + B11)
            Uv[idx_u(0,0)] = 1.0; Uv[idx_u(1,1)] = 1.0
            Vv[idx_v(0,0)] = 1.0; Vv[idx_v(1,1)] = 1.0
            Wv[idx_w(0,0)] = 1.0; Wv[idx_w(1,1)] = 1.0
        elif t == 1:
            # P1 = (A10 + A11) * B00
            Uv[idx_u(1,0)] = 1.0; Uv[idx_u(1,1)] = 1.0
            Vv[idx_v(0,0)] = 1.0
            Wv[idx_w(0,1)] = 1.0
        elif t == 2:
            # P2 = A00 * (B10 - B00)
            Uv[idx_u(0,0)] = 1.0
            Vv[idx_v(1,0)] = 1.0; Vv[idx_v(0,0)] = -1.0
            Wv[idx_w(0,0)] = 1.0
        elif t == 3:
            # P3 = A11 * (B01 - B11)
            Uv[idx_u(1,1)] = 1.0
            Vv[idx_v(0,1)] = 1.0; Vv[idx_v(1,1)] = -1.0
            Wv[idx_w(1,1)] = 1.0
        elif t == 4:
            # P4 = (A00 + A01) * B11
            Uv[idx_u(0,0)] = 1.0; Uv[idx_u(0,1)] = 1.0
            Vv[idx_v(1,1)] = 1.0
            Wv[idx_w(1,0)] = 1.0
        elif t == 5:
            # P5 = (A10 - A00) * (B00 + B01)
            Uv[idx_u(1,0)] = 1.0; Uv[idx_u(0,0)] = -1.0
            Vv[idx_v(0,0)] = 1.0; Vv[idx_v(0,1)] = 1.0
            Wv[idx_w(0,1)] = 1.0; Wv[idx_w(1,0)] = 1.0
            # Wait, let me re-verify Strassen indices.
            # C00 = P0 + P2 - P3 - P5
            # C01 = P4 + P5
            # C10 = P1 + P3
            # C11 = P0 - P1 + P2 + P4
            
            # My W assignments must match.
            # P0 contributes to C00, C11 -> W(0,0), W(1,1) OK.
            # P1 contributes to C10 -> W(0,1) OK.
            # P2 contributes to C00 -> W(0,0) OK.
            # P3 contributes to C11 -> W(1,1) OK.
            # P4 contributes to C01 -> W(1,0) OK.
            # P5 contributes to C01, C00 -> W(1,0), W(0,0) OK?
            # Let's check P5: (A10 - A00)(B00 + B01).
            # U: A10, -A00. V: B00, B01.
            # Product: A10*B00 + A10*B01 - A00*B00 - A00*B01.
            # This is not a single term.
            # Strassen is 7 multiplications for 4 outputs.
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B00)
            # M4 = A11(B01-B11)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)  <-- This is M6 in some formulations, but usually M6 and M7 are for the other parts.
            # M7 = (A01-A11)(B10+B11)
            
            # Let's stick to the standard 7 Strassen multiplications for 2x2x2.
            # I will use a pre-verified list of (U_mask, V_mask, W_mask) for 2x2x2.
            # Indices: (i,j) in A, (j,k) in B, (i,k) in C.
            pass
            
    # Let's use a simpler, verified Strassen 2x2x2 decomposition
    # Standard Strassen 7 terms for C = AB where A, B, C are 2x2.
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B01 - B11)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Let's rebuild the 2x2x2 part properly.
    def add_strassen_block(i0, j0, k0):
        # Maps local (0,0), (1,0), (0,1), (1,1) to global (i0, j0+k0) etc? No, A[i,j], B[j,k], C[i,k].
        # A indices: (i0, j0), (i0+1, j0), (i0, j0+1), (i0+1, j0+1)
        # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # C indices: (i0, k0), (i0+1, k0), (i0, k0+1), (i0+1, k0+1)
        # Note: The shared index j is in A's col and B's row.
        # A: [i0..i0+1, j0..j0+1]
        # B: [j0..j0+1, k0..k0+1]
        
        def get_u(i, j): return idx_u(i0+i, j0+j)
        def get_v(j, k): return idx_v(j0+j, k0+k)
        def get_w(k, i): return idx_w(k0+k, i0+i)
        
        terms = []
        # M1 = (A00+A11)(B00+B11)
        u = [0]*4; v = [0]*4; w = [0]*4
        u[0]=1; u[3]=1 # A00, A11
        v[0]=1; v[3]=1 # B00, B11
        w[0]=1; w[3]=1 # C00, C11
        terms.append((u,v,w))
        
        # M2 = (A10+A11)B00
        u = [0]*4; v = [0]*4; w = [0]*4
        u[2]=1; u[3]=1 # A10, A11
        v[0]=1 # B00
        w[2]=1 # C10
        terms.append((u,v,w))
        
        # M3 = A00(B10-B00)
        u = [0]*4; v = [0]*4; w = [0]*4
        u[0]=1 # A00
        v[2]=1; v[0]=-1 # B10, -B00
        w[0]=1 # C00
        terms.append((u,v,w))
        
        # M4 = A11(B01-B11)
        u = [0]*4; v = [0]*4; w = [0]*4
        u[3]=1 # A11
        v[1]=1; v[3]=-1 # B01, -B11
        w[3]=1 # C11
        terms.append((u,v,w))
        
        # M5 = (A00+A01)B11
        u = [0]*4; v = [0]*4; w = [0]*4
        u[0]=1; u[1]=1 # A00, A01
        v[3]=1 # B11
        w[1]=1 # C01
        terms.append((u,v,w))
        
        # M6 = (A10-A00)(B00+B01)
        u = [0]*4; v = [0]*4; w = [0]*4
        u[2]=1; u[0]=-1 # A10, -A00
        v[0]=1; v[1]=1 # B00, B01
        w[1]=1; w[0]=-1 # C01, -C00
        terms.append((u,v,w))
        
        # M7 = (A01-A11)(B10+B11)
        u = [0]*4; v = [0]*4; w = [0]*4
        u[1]=1; u[3]=-1 # A01, -A11
        v[2]=1; v[3]=1 # B10, B11
        w[2]=1; w[3]=-1 # C10, -C11
        terms.append((u,v,w))
        
        # Expand these local terms into global vectors and add to rank_terms
        for u_loc, v_loc, w_loc in terms:
            Uv = [0.0] * (n*m)
            Vv = [0.0] * (m*p)
            Wv = [0.0] * (p*n)
            Uv[get_u(0,0)] += u_loc[0]
            Uv[get_u(0,1)] += u_loc[1]
            Uv[get_u(1,0)] += u_loc[2]
            Uv[get_u(1,1)] += u_loc[3]
            
            Vv[get_v(0,0)] += v_loc[0]
            Vv[get_v(0,1)] += v_loc[1]
            Vv[get_v(1,0)] += v_loc[2]
            Vv[get_v(1,1)] += v_loc[3]
            
            Wv[get_w(0,0)] += w_loc[0]
            Wv[get_w(0,1)] += w_loc[1]
            Wv[get_w(1,0)] += w_loc[2]
            Wv[get_w(1,1)] += w_loc[3]
            
            rank_terms.append((Uv, Vv, Wv))

    # Use 3 Strassen blocks for i=0,1; j=0,1; k=0,1,4,5? 
    # Dimensions: A(3,5), B(5,7), C(3,7).
    # Shared j is 0..4. Shared k is 0..6.
    # We can tile the j dimension (0..4) and k dimension (0..6).
    # Block size 2x2 in j and k?
    # If we use j=0,1 and k=0,1: One Strassen block (rank 7).
    # If we use j=0,1 and k=2,3: One Strassen block (rank 7).
    # If we use j=0,1 and k=4,5: One Strassen block (rank 7).
    # Remaining j=2,3,4 (width 3) and k=6 (width 1).
    # And i=2 is not covered by Strassen (which covers i=0,1).
    
    # This approach is getting complicated. Let's try a simpler hybrid.
    # Use Strassen for the 2x2x2 core (i=0,1; j=0,1; k=0,1). Rank 7.
    # The rest must be handled by schoolbook or other means.
    # This only saves 4 terms (105 - 7 - rest). Not much gain if rest is schoolbook.
    
    # Alternative: Use Strassen for all of i=0,1 by tiling j and k.
    # i is fixed at 0,1.
    # A is 3x5. Rows 0,1 are used.
    # B is 5x7. 
    # We can split B's columns (k) into 3 blocks of 2,1? 
    # j must be shared.
    # If we take j=0,1 for one set of Strassen blocks, and j=2,3 for another, and j=4 for a third.
    
    # Let's just use schoolbook for now to ensure correctness, as Strassen implementation for non-cubic is tricky.
    # Rank 105.
    
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
