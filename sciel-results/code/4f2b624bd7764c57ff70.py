import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank for remainders
    # Strassen rank for 2x2x2 blocks
    
    # Initialize U, V, W with 0s. We don't know the final rank exactly until we count,
    # but we can estimate: 8 blocks * 7 + 20 + 16 = 92.
    # Let's allocate a bit more to be safe, or just append.
    # Appending is safer to avoid index errors if my count is wrong.
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        U.append([0] * len(U[0]) if U else []) # This is inefficient. 
        # Better to use lists and extend or pre-allocate.
        pass

    # Pre-allocation is better.
    # Max rank: 100.
    # Expected rank: 92.
    # Let's use 100 to be safe, then trim? Or just use lists and zip at the end?
    # The interface expects {'U': ..., 'V': ..., 'W': ...}.
    # If I use lists of lists, I can just append columns.
    
    # Let's use a list of columns for each matrix.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        U_cols.append([0.0] * (n * m))
        V_cols.append([0.0] * (m * p))
        W_cols.append([0.0] * (p * n))
        U_cols[-1][u_idx] = u_val
        V_cols[-1][v_idx] = v_val
        W_cols[-1][w_idx] = w_val

    # Strassen coefficients for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices in 2x2x2 block (local i, j, k in 0..1)
    # A indices: (i, j) -> u_idx = i*2 + j
    # B indices: (j, k) -> v_idx = j*2 + k
    # C indices: (k, i) -> w_idx = k*2 + i (Note: W is k*n+i, local n=2)
    
    # We need to map local indices to global indices.
    
    # Block 1: i0=0, j0=0, k0=0
    # Block 2: i0=0, j0=0, k0=2
    # Block 3: i0=0, j0=2, k0=0
    # Block 4: i0=0, j0=2, k0=2
    # Block 5: i0=2, j0=0, k0=0
    # Block 6: i0=2, j0=0, k0=2
    # Block 7: i0=2, j0=2, k0=0
    # Block 8: i0=2, j0=2, k0=2
    
    def add_strassen_block(i0, j0, k0):
        # Local indices 0,1
        # M1
        # u: (0,0)=1, (1,1)=1
        # v: (0,0)=1, (1,1)=1
        # w: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1 ? No, w is sum of products.
        # The rank-1 decomposition adds to the output tensor T[i,j,k].
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # For Strassen, we define 7 rank-1 terms.
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C01=1, C10=1, C11=1 ?
        # Wait, C is 2x2. C00, C01, C10, C11.
        # W index is k*n + i.
        # C00 (k=0, i=0) -> W[0*2+0] = W[0]
        # C01 (k=0, i=1) -> W[0*2+1] = W[1]
        # C10 (k=1, i=0) -> W[1*2+0] = W[2]
        # C11 (k=1, i=1) -> W[1*2+1] = W[3]
        
        # M1 contributes to C00, C01, C10, C11 with coeff 1?
        # C00 = M1 + ...
        # C01 = ...
        # C10 = ...
        # C11 = M1 + ...
        
        # Let's define the 7 terms explicitly.
        
        # Term 1: M1
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1 (Wait, C00 and C11 get M1. C01 and C10 do not get M1 directly in the standard form?
        # Standard Strassen:
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        
        # c11 = p1 + p4 - p5 + p7
        # c12 = p3 + p5
        # c21 = p2 + p4
        # c22 = p1 - p2 + p3 + p6
        
        # So P1 contributes to c11 and c22.
        # c11 is (i=0, k=0) in C? No, C[i,k].
        # c11 is i=0, k=0.
        # c22 is i=1, k=1.
        
        # So for Term 1 (P1):
        # U: A[0,0]=1, A[1,1]=1
        # V: B[0,0]=1, B[1,1]=1
        # W: C[0,0]=1 (idx 0), C[1,1]=1 (idx 3)
        
        def add_local_term(u_mask, v_mask, w_mask):
            # u_mask: dict local (i,j) -> val
            # v_mask: dict local (j,k) -> val
            # w_mask: dict local (k,i) -> val
            
            u_idx_global = {}
            v_idx_global = {}
            w_idx_global = {}
            
            for (li, lj), val in u_mask.items():
                gi = i0 + li
                gj = j0 + lj
                u_idx_global[gi * m + gj] = val
                
            for (lj, lk), val in v_mask.items():
                gi = i0 + li # Wait, li is not in v_mask. v is j,k.
                # v_mask keys are (lj, lk)
                gj = j0 + lj
                gk = k0 + lk
                v_idx_global[gj * p + gk] = val
                
            for (lk, li), val in w_mask.items():
                gi = i0 + li
                gk = k0 + lk
                w_idx_global[gk * n + gi] = val
                
            # Convert to lists
            u_list = [0.0] * (n * m)
            v_list = [0.0] * (m * p)
            w_list = [0.0] * (p * n)
            
            for idx, val in u_idx_global.items():
                u_list[idx] = val
            for idx, val in v_idx_global.items():
                v_list[idx] = val
            for idx, val in w_idx_global.items():
                w_list[idx] = val
                
            U_cols.append(u_list)
            V_cols.append(v_list)
            W_cols.append(w_list)
            
        # Term 1: P1
        add_local_term(
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        )
        
        # Term 2: P2 = (c+d)e
        # U: A[1,0]=1, A[1,1]=1
        # V: B[0,0]=1
        # W: C[1,0]=1 (c21)
        add_local_term(
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1} # c21 is i=1, k=0 -> W[0*2+1] = W[1]?
            # Wait. W index is k*n + i.
            # c21 is row 1, col 0 of C?
            # C[i,k]. c21 is i=1, k=0?
            # Standard notation C_{ij} where i is row, j is col.
            # Here C is p x n? No, C is n x p.
            # "multiplying a 4 by 5 matrix by a 5 by 5 matrix".
            # A is 4x5. B is 5x5. C is 4x5.
            # C[i,k] = sum_j A[i,j] B[j,k].
            # So i is 0..3, k is 0..4.
            # In the 2x2 block, i is 0..1, k is 0..1.
            # c11 is i=0, k=0.
            # c12 is i=0, k=1.
            # c21 is i=1, k=0.
            # c22 is i=1, k=1.
            
            # W index: k*n + i.
            # c21 (i=1, k=0) -> W[0*2 + 1] = W[1].
            # My previous mapping:
            # C00 (i=0, k=0) -> W[0]
            # C01 (i=0, k=1) -> W[2] ? No. W[k*n+i]. k=0, n=2, i=1 -> W[1].
            # C10 (i=1, k=0) -> W[0*2+1] = W[1].
            # C11 (i=1, k=1) -> W[1*2+1] = W[3].
            
            # Let's re-verify W indices for 2x2 block.
            # k=0, i=0 -> W[0]
            # k=0, i=1 -> W[1]
            # k=1, i=0 -> W[2]
            # k=1, i=1 -> W[3]
            
            # So:
            # c11 (i=0, k=0) -> W[0]
            # c12 (i=0, k=1) -> W[2]
            # c21 (i=1, k=0) -> W[1]
            # c22 (i=1, k=1) -> W[3]
            
            # Correcting Term 1:
            # P1 contributes to c11 (W[0]) and c22 (W[3]).
            
            # Correcting Term 2:
            # P2 contributes to c21 (W[1]).
        )
        
        # I need to be very careful with indices.
        # Let's restart the local term definitions with correct W indices.
        
        # Clear previous appends for this block if I made mistakes?
        # It's better to define the 7 terms correctly in one go.
        
        # Let's redefine add_strassen_block cleanly.
        
    # Redefine add_strassen_block
    
    def add_strassen_block(i0, j0, k0):
        # Local indices
        # U: (i,j)
        # V: (j,k)
        # W: (k,i) -> global W[k*n+i]
        
        # Term 1: P1 = (A00+A11)(B00+B11) -> C00 + C11
        # U: (0,0)=1, (1,1)=1
        # V: (0,0)=1, (1,1)=1
        # W: C00 (k=0,i=0)->W[0]=1, C11 (k=1,i=1)->W[3]=1
        add_local_term(
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1} # Wait, W keys are (k,i).
            # C00 is k=0, i=0. Key (0,0).
            # C11 is k=1, i=1. Key (1,1).
        )
        
        # Term 2: P2 = (A10+A11)B00 -> C10 (i=1, k=0)
        # U: (1,0)=1, (1,1)=1
        # V: (0,0)=1
        # W: C10 (k=0, i=1) -> Key (0,1)
        add_local_term(
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1}
        )
        
        # Term 3: P3 = A00(B01-B11) -> C01 (i=0, k=1)
        # U: (0,0)=1
        # V: (0,1)=1, (1,1)=-1
        # W: C01 (k=1, i=0) -> Key (1,0)
        add_local_term(
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1}
        )
        
        # Term 4: P4 = A11(B10-B00) -> C00 + C10
        # U: (1,1)=1
        # V: (1,0)=1, (0,0)=-1
        # W: C00 (k=0,i=0)->W[0]=1, C10 (k=0,i=1)->W[1]=1
        add_local_term(
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        )
        
        # Term 5: P5 = (A00+A01)B11 -> C01 - C00
        # U: (0,0)=1, (0,1)=1
        # V: (1,1)=1
        # W: C01 (k=1,i=0)->W[2]=1, C00 (k=0,i=0)->W[0]=-1
        add_local_term(
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(1,0): 1, (0,0): -1}
        )
        
        # Term 6: P6 = (A10-A00)(B00+B01) -> C11
        # U: (1,0)=1, (0,0)=-1
        # V: (0,0)=1, (0,1)=1
        # W: C11 (k=1,i=1)->W[3]=1
        add_local_term(
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        )
        
        # Term 7: P7 = (A01-A11)(B10+B11) -> C00 + C11
        # U: (0,1)=1, (1,1)=-1
        # V: (1,0)=1, (1,1)=1
        # W: C00 (k=0,i=0)->W[0]=1, C11 (k=1,i=1)->W[3]=1
        add_local_term(
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        )

    # Define add_local_term properly
    def add_local_term(u_mask, v_mask, w_mask):
        u_list = [0.0] * (n * m)
        v_list = [0.0] * (m * p)
        w_list = [0.0] * (p * n)
        
        for (li, lj), val in u_mask.items():
            gi = i0 + li
            gj = j0 + lj
            u_list[gi * m + gj] = val
            
        for (lj, lk), val in v_mask.items():
            gj = j0 + lj
            gk = k0 + lk
            v_list[gj * p + gk] = val
            
        for (lk, li), val in w_mask.items():
            gk = k0 + lk
            gi = i0 + li
            w_list[gk * n + gi] = val
            
        U_cols.append(u_list)
        V_cols.append(v_list)
        W_cols.append(w_list)

    # Add 8 Strassen blocks
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                add_strassen_block(i0, j0, k0)

    # Remainder 1: j=4 (0-based), i in 0..3, k in 0..4
    # A[i,4] B[4,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            U_cols.append([0.0] * (n * m))
            V_cols.append([0.0] * (m * p))
            W_cols.append([0.0] * (p * n))
            U_cols[-1][u_idx] = 1.0
            V_cols[-1][v_idx] = 1.0
            W_cols[-1][w_idx] = 1.0

    # Remainder 2: k=4 (0-based), i in 0..3, j in 0..3
    # A[i,j] B[j,4] -> C[i,4]
    for i in range(n):
        for j in range(4): # j in 0..3
            u_idx = i * m + j
            v_idx = j * p + 4
            w_idx = 4 * n + i
            U_cols.append([0.0] * (n * m))
            V_cols.append([0.0] * (m * p))
            W_cols.append([0.0] * (p * n))
            U_cols[-1][u_idx] = 1.0
            V_cols[-1][v_idx] = 1.0
            W_cols[-1][w_idx] = 1.0

    # Convert to lists of rows
    rank = len(U_cols)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
