import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    
    # Split middle dimension into two halves of size 2
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    # Each is a 2x2x8 multiplication with rank 16
    # Total rank = 32
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 8 x 32
    V = [[0] * rank for _ in range(m * p)]  # 32 x 32
    W = [[0] * rank for _ in range(p * n)]  # 16 x 32
    
    # For each half (h=0,1), we have a 2x2x8 multiplication
    # A_half: 2x2 (rows 0-1, cols h*2 to h*2+1)
    # B_half: 2x8 (rows h*2 to h*2+1, all 8 cols)
    # C: 2x8
    
    col = 0
    for h in range(2):  # two halves
        for r in range(2):  # row of A (0,1)
            for c in range(8):  # col of C (0..7)
                # Term: A[r, h*2 + k] * B[h*2 + k, c] for k=0,1
                # This is a 2x2x8 schoolbook: for each (k, c) pair, one rank-1 term
                # Wait, no. For 2x2x8: C[r,c] = sum_k A[r,k]*B[k,c]
                # Rank-1 term for (k,c): u_r = delta(r,k), v_k = delta(k,k), w_c = delta(c,c)
                # So for each (k,c): U[r=k] += 1, V[k*8+c] += 1, W[c*2+r] += 1
                for k in range(2):
                    # U: row r, but only when r==k... 
                    # Actually for schoolbook: term (r,k,c) contributes to C[r,c]
                    # U[r*m + (h*2+k)] = 1  (but this would make it 4x8 per half, not 2x2x8)
                    pass
        col += 16
    
    # Let me redo this more carefully.
    # 2x2x8 schoolbook: 16 terms, one for each (k, c) where k in {0,1}, c in {0..7}
    # Term (k,c): U[row r] gets 1 if r==k, V[row k, col c] gets 1, W[row c, col r] gets 1
    # But U has shape (2*2, 16) = (4, 16) for the 2x2 part of A
    # V has shape (2*8, 16) = (16, 16) for the 2x8 part of B
    # W has shape (8*2, 16) = (16, 16) for the 8x2 part of C
    
    # Actually, let me just build it directly.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for h in range(2):  # half index (0 or 1)
        for k in range(2):  # middle index within half (0 or 1)
            for c in range(8):  # output column
                # This term: A[r, h*2+k] * B[h*2+k, c] contributes to C[r, c] for r in {0,1}
                # But it's a single rank-1 term that handles both r values? No.
                # For schoolbook 2x2x8: each (k,c) pair gives a term where:
                #   u(r) = delta(r,k), v(k') = delta(k',k), w(c') = delta(c',c)
                # So for fixed (k,c):
                #   U[r, col] = 1 if r == k, else 0  (for r in 0,1)
                #   V[k, c, col] = 1
                #   W[c, r, col] = 1 if... no, W[c*n + r, col] = 1 for the r that matches
                # 
                # Wait, I'm overcomplicating. The 2x2x8 tensor has entries T[r,k,c] = delta(r,k) * ... no.
                # The multiplication tensor for 2x2x8: T[r,k,c] such that C[r,c] = sum_{k} A[r,k] B[k,c]
                # So T[r,k,c] = 1 for all r,k,c? No, T[r,k,c] = delta_{r,k} is wrong.
                # 
                # Actually: C[r,c] = sum_k A[r,k] * B[k,c]
                # The tensor T has T[r,k,c] = 1 for all valid r,k,c (it's the "1" tensor in the sense
                # that the bilinear form is sum_{r,k,c} T[r,k,c] A[r,k] B[k,c] = sum_{r,c} C[r,c])
                # No wait: sum_{r,k,c} T[r,k,c] A[r,k] B[k,c] should equal sum_{r,c} C[r,c] * (something)
                # 
                # The correct tensor: T[r,k,c] = 1 for all r in {0,1}, k in {0,1}, c in {0..7}
                # Then sum T[r,k,c] A[r,k] B[k,c] = sum_{r,k,c} A[r,k] B[k,c] which is NOT what we want.
                # 
                # We want: C[r,c] = sum_k A[r,k] B[k,c]
                # So the tensor is: for each (r,c), C[r,c] = sum_k A[r,k] B[k,c]
                # The rank-1 decomposition: C[r,c] = sum_{k,c'} delta(r,k) delta(c,c') A[r,k] B[k,c']
                # No: C[r,c] = sum_k A[r,k] B[k,c] = sum_{(k,c')} [delta(r,k) * A[r,k]] * [delta(c,c') * B[k,c']]
                # So the rank-1 terms are indexed by (k, c'):
                #   u_{(k,c')}(r) = delta(r, k)
                #   v_{(k,c')}(k') = delta(k', k) ... no, v should be a function of (k', c'') 
                # 
                # I think I'm confusing myself. Let me just use the standard schoolbook:
                # For 2x2x8: 16 terms, indexed by (k, c) where k in {0,1}, c in {0..7}
                # Term (k,c):
                #   u(r) = 1 if r == k, else 0
                #   v(k', c') = 1 if k' == k and c' == c, else 0
                #   w(c'', r'') = 1 if c'' == c and r'' == r... no, w should depend on (c, r)
                # 
                # The output C has shape 2x8, indexed by (r, c).
                # W has shape (p*n, rank) = (16, rank), indexed by (c*n + r) or (r*p + c)?
                # The problem says W[k*n+i, r] where k is col, i is row. So W[c*n + r, term].
                # 
                # For term (k, c):
                #   U[r, term] = 1 if r == k, else 0  (U has shape (n*m, rank) = (8, rank), indexed by r*m + k_a)
                #   Wait, U is indexed by (r, k_a) where r is row of A (0,1) and k_a is col of A (0,1,2,3)
                #   For the half h, the cols are h*2 and h*2+1
                #   So U[r*m + (h*2 + k), term] = 1 if r == k, else 0? No, that's wrong.
                # 
                # I think the issue is that for 2x2x8, the "2" in the middle is the k index, and
                # A[r, k] for r in {0,1}, k in {0,1}. But in our problem, A is 2x4, so for half h,
                # A[r, h*2+k] for k in {0,1}.
                # 
                # Let me just directly construct the 32 terms.
                pass
    
    # Direct construction: 32 terms, each corresponding to (h, k, c) where h in {0,1}, k in {0,1}, c in {0..7}
    # But that's 2*2*8 = 32 terms. For each:
    #   U[r, term] = 1 if r == k, else 0  -- but r is 0 or 1, and k is 0 or 1, so this works
    #   But U is indexed by (r, col_of_A) where col_of_A = h*2 + k
    #   So U[r * 4 + (h*2 + k), term] = 1 if r == k, else 0
    #   V[k_row, c, term] = 1 if k_row == h*2 + k, else 0  (V indexed by (k_row, c))
    #   W[c, r, term] = 1  (W indexed by (c, r), i.e., W[c*2 + r, term])
    
    U = [[0] * rank for _ in range(8)]   # 8 x 32
    V = [[0] * rank for _ in range(32)]  # 32 x 32
    W = [[0] * rank for _ in range(16)]  # 16 x 32
    
    term = 0
    for h in range(2):
        for k in range(2):
            for c in range(8):
                # U: for r in {0,1}, U[r*4 + (h*2+k), term] = 1 if r == k
                for r in range(2):
                    if r == k:
                        U[r * 4 + (h * 2 + k)][term] = 1
                # V: V[(h*2+k)*8 + c, term] = 1
                V[(h * 2 + k) * 8 + c][term] = 1
                # W: W[c*2 + r, term] for the r that matches... 
                # Actually, W[c*2 + r, term] should be 1 for the r that this term contributes to.
                # This term contributes to C[r, c] for r == k.
                # So W[c*2 + k, term] = 1
                W[c * 2 + k][term] = 1
                term += 1
    
    return {"U": U, "V": V, "W": W}

