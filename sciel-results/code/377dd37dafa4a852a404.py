import numpy as np
import time

def _strassen_2x2x2(i_offset, j_offset, k_offset):
    """
    Returns U, V, W contributions for a 2x2x2 Strassen block.
    i_offset, j_offset, k_offset are the starting indices for i, j, k.
    The block covers i in [i_offset, i_offset+2), j in [j_offset, j_offset+2), k in [k_offset, k_offset+2).
    """
    # Strassen 2x2x2 rank 7
    # Indices: i0,i1; j0,j1; k0,k1
    # U is 2x2 (i,j), V is 2x2 (j,k), W is 2x2 (k,i)
    # We need to map these to the global indices.
    
    # Local indices
    # i: 0, 1
    # j: 0, 1
    # k: 0, 1
    
    # Strassen terms:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We represent this as 7 rank-1 terms.
    # Each term c has U[i,j,c], V[j,k,c], W[k,i,c].
    
    terms = []
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        # u_vals: dict {(i_local, j_local): val}
        # v_vals: dict {(j_local, k_local): val}
        # w_vals: dict {(k_local, i_local): val}
        terms.append((u_vals, v_vals, w_vals))
        
    # M1: (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C11=1
    # Wait, Strassen computes C = AB.
    # The term is U * V * W.
    # U corresponds to A, V to B, W to C.
    # For M1:
    # A part: A00 + A11. So U[0,0]=1, U[1,1]=1.
    # B part: B00 + B11. So V[0,0]=1, V[1,1]=1.
    # C part: C00 + C11. So W[0,0]=1, W[1,1]=1.
    add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    
    # M2: (A10+A11)B00
    # U: A10=1, A11=1 -> U[1,0]=1, U[1,1]=1
    # V: B00=1 -> V[0,0]=1
    # W: C10=1 -> W[0,1]=1
    add_term({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1})
    
    # M3: A00(B01-B11)
    # U: A00=1 -> U[0,0]=1
    # V: B01=1, B11=-1 -> V[0,1]=1, V[1,1]=-1
    # W: C01=1 -> W[1,0]=1
    add_term({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1})
    
    # M4: A11(B10-B00)
    # U: A11=1 -> U[1,1]=1
    # V: B10=1, B00=-1 -> V[1,0]=1, V[0,0]=-1
    # W: C10=1 -> W[0,1]=1
    add_term({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,1):1})
    
    # M5: (A00+A01)B11
    # U: A00=1, A01=1 -> U[0,0]=1, U[0,1]=1
    # V: B11=1 -> V[1,1]=1
    # W: C00=-1, C11=1? No, C = ... - M5 ...
    # C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    # So M5 contributes -1 to C00 and 0 to others?
    # Wait, M5 = (A00+A01)B11.
    # This product affects C01? No.
    # A00*B11 is part of C01? No, A00*B01 is C01.
    # A01*B11 is part of C01? No.
    # Let's check standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M5 contributes to C00 (-1) and C01 (+1).
    # W: C00=-1, C01=1.
    # W[0,0] corresponds to C00? No, W is indexed by (k,i).
    # C is indexed by (i,k).
    # W[k,i,c] is the coefficient for C[i,k,c]?
    # The tensor equation is sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k].
    # So W[k,i,c] multiplies the result for C[i,k].
    # So for M5:
    # W[0,0] (k=0, i=0) -> C00. Coeff -1.
    # W[1,0] (k=1, i=0) -> C01. Coeff 1.
    add_term({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1})
    
    # M6: (A10-A00)(B00+B01)
    # U: A10=1, A00=-1 -> U[1,0]=1, U[0,0]=-1
    # V: B00=1, B01=1 -> V[0,0]=1, V[0,1]=1
    # W: C11=1 -> W[1,1]=1
    add_term({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})
    
    # M7: (A01-A11)(B10+B11)
    # U: A01=1, A11=-1 -> U[0,1]=1, U[1,1]=-1
    # V: B10=1, B11=1 -> V[1,0]=1, V[1,1]=1
    # W: C00=1 -> W[0,0]=1
    add_term({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    
    return terms

def construct():
    n, m, p = 4, 5, 5
    rank = 92
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Part 1: 8 Strassen blocks for j in 0..3, k in 0..3
    # j_blocks: [0,1], [2,3]
    # k_blocks: [0,1], [2,3]
    # i is full 0..3, so we split i into [0,1] and [2,3]
    i_blocks = [(0, 2), (2, 4)]
    j_blocks = [(0, 2), (2, 4)]
    k_blocks = [(0, 2), (2, 4)]
    
    for i_start, i_end in i_blocks:
        for j_start, j_end in j_blocks:
            for k_start, k_end in k_blocks:
                # Get Strassen terms for this 2x2x2 block
                terms = _strassen_2x2x2(i_start, j_start, k_start)
                for u_vals, v_vals, w_vals in terms:
                    # u_vals: {(i_local, j_local): val}
                    for (li, lj), val in u_vals.items():
                        gi = i_start + li
                        gj = j_start + lj
                        U[gi*m + gj, col] = val
                    for (lj, lk), val in v_vals.items():
                        gj = j_start + lj
                        gk = k_start + lk
                        V[gj*p + gk, col] = val
                    for (lk, li), val in w_vals.items():
                        gk = k_start + lk
                        gi = i_start + li
                        W[gk*n + gi, col] = val
                    col += 1
                    
    # Part 2: Schoolbook for remaining terms
    # Remaining: j=4 or k=4
    # Case 1: j=4, k in 0..4
    for i in range(n):
        for k in range(p):
            U[i*m + 4, col] = 1
            V[4*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
            
    # Case 2: k=4, j in 0..3
    for i in range(n):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 4, col] = 1
            W[4*n + i, col] = 1
            col += 1
            
    # Verify col == rank
    if col != rank:
        # Fallback to schoolbook if logic error
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
