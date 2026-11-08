import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 16
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strategy: 7 Strassen terms for B[0:2,0:2] -> C[:,0:2], 
    # plus 9 terms for the rest. Total = 16.
    # 
    # Actually, let me construct it as:
    # Terms 0-6: Strassen for the 2x2x2 subproblem (A cols 0,1; B rows 0,1; C cols 0,1)
    # Terms 7-15: Handle remaining contributions
    #
    # For the Strassen part, the 2x2x2 tensor T[i,j,k]=1 for i,j,k in {0,1}
    # maps to: U[i*4+j, t], V[j*4+k, t], W[k*2+i, t]
    # where j is in {0,1} (B rows 0,1), k is in {0,1} (C cols 0,1)
    
    # Strassen 7 terms for 2x2x2:
    # M1 = (X00+X11)(Y00+Y11) -> contributes to C00, C01, C10, C11
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # In our indexing:
    # U row index = i*4 + j where i in {0,1}, j in {0,1} (A[i,j])
    # V row index = j*4 + k where j in {0,1} (B row), k in {0,1} (B col / C col)
    # W row index = k*2 + i where k in {0,1} (C col), i in {0,1} (C row)
    
    strassen = [
        # (U coeffs, V coeffs, W coeffs)
        # M1: (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6: (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Apply Strassen terms 0-6
    for t_idx, (u_dict, v_dict, w_dict) in enumerate(strassen):
        t = t_idx
        for u_idx, coeff in u_dict.items():
            # u_idx is local 2x2 index: row i, col j_local (0 or 1)
            i = u_idx // 2
            j_local = u_idx % 2
            j_global = j_local  # B rows 0,1
            U[i * 4 + j_global, t] += coeff
        for v_idx, coeff in v_dict.items():
            j_local = v_idx // 2
            k_local = v_idx % 2
            j_global = j_local  # B rows 0,1
            k_global = k_local  # C cols 0,1
            V[j_global * 4 + k_global, t] += coeff
        for w_idx, coeff in w_dict.items():
            k_local = w_idx // 2
            i_local = w_idx % 2
            k_global = k_local  # C cols 0,1
            i_global = i_local  # C rows 0,1
            W[k_global * 2 + i_global, t] += coeff
    
    # Now handle remaining contributions:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Already handled: j in {0,1}, k in {0,1} via Strassen
    # Remaining:
    #   j in {0,1}, k in {2,3}: A[i,0]*B[0,k] + A[i,1]*B[1,k] for k=2,3
    #   j in {2,3}, k in {0,1}: A[i,2]*B[2,k] + A[i,3]*B[3,k] for k=0,1
    #   j in {2,3}, k in {2,3}: A[i,2]*B[2,k] + A[i,3]*B[3,k] for k=2,3
    #
    # That's 2*2 + 2*2 + 2*2 = 12 sub-problems, each a 2x1x1 (or similar)
    # We have 9 terms left (terms 7-15).
    #
    # Let me use 4 terms for j in {2,3}, k in {0,1} (2x2x2 subproblem -> 4 terms direct)
    # 4 terms for j in {2,3}, k in {2,3} (2x2x2 subproblem -> 4 terms direct)  
    # 1 term for... hmm, that's only 8. We need to also handle j in {0,1}, k in {2,3}.
    #
    # Wait, I'm overcomplicating. Let me use 9 terms:
    # Terms 7-10: Handle j in {2,3}, k in {0,3} with 4 direct terms
    #   Term 7: A[i,2]*B[2,0] -> C[i,0]
    #   Term 8: A[i,2]*B[2,1] -> C[i,1]  
    #   Term 9: A[i,3]*B[3,0] -> C[i,0]
    #   Term 10: A[i,3]*B[3,1] -> C[i,1]
    #   But these need to work for both i=0 and i=1...
    #
    # A rank-1 term: U[i*4+j, t] * V[j*4+k, t] * W[k*2+i, t] = 1 for specific (i,j,k)
    # If U[i*4+j, t] = 1, V[j*4+k, t] = 1, W[k*2+i, t] = 1, then only that (i,j,k) gets 1.
    # But we need ALL (i,j,k) in the remaining set to get 1.
    #
    # For j=2, k=0: we need C[i,0] += A[i,2]*B[2,0] for i=0,1
    # So we need: U[i*4+2, t]*V[2*4+0, t]*W[0*2+i, t] = 1 for i=0,1
    # If V[8, t] = 1, then U[i*4+2, t]*W[i, t] = 1 for i=0,1
    # Set U[2, t] = 1, U[6, t] = 1, W[0, t] = 1, W[1, t] = 1
    # That's one term handling both i values for (j=2, k=0)!
    #
    # So each (j,k) pair can be handled by ONE term that covers both i values!
    # Number of remaining (j,k) pairs: j in {0,1,2,3} x k in {0,1,2,3} minus the 4 handled by Strassen
    # Strassen handles j in {0,1}, k in {0,1} = 4 pairs
    # Remaining: 16 - 4 = 12 pairs
    # But we only have 9 terms. So this approach uses 12 terms, total = 7+12 = 19. Too many.
    #
    # Alternative: Can we share terms across multiple (j,k) pairs?
    # If V[j*4+k, t] has multiple non-zero entries, one term can handle multiple (j,k) pairs.
    #
    # Let me try: use 4 terms for j in {2,3} (all k), and 5 terms for j in {0,1}, k in {2,3}
    # 
    # For j=2: V[8+k, t] for k=0,1,2,3. If I set V[8, t]=V[9, t]=V[10, t]=V[11, t]=1,
    # then U[i*4+2, t]*W[k*2+i, t] must be 1 for all i,k. 
    # That requires W[k*2+i, t] = 1 for all i,k, and U[i*4+2, t] = 1 for all i.
    # So: U[2, t]=1, U[6, t]=1, V[8,t]=V[9,t]=V[10,t]=V[11,t]=1, W[0,t]=W[1,t]=W[2,t]=W[3,t]=1
    # This gives contribution 1 for all (i, j=2, k) pairs. 
    # Similarly for j=3: another term.
    #
    # So 2 terms handle j in {2,3} for all k. 
    # For j in {0,1}, k in {2,3}: 2x2 = 4 (j,k) pairs.
    # Use 2 terms: one for j=0 (k=2,3), one for j=1 (k=2,3)
    # Term: U[i*4+0, t]=1 for i=0,1; V[0*4+2, t]=V[0*4+3, t]=1; W[2*2+i, t]=W[3*2+i, t]=1 for i=0,1
    # That's W[4,t]=W[5,t]=W[6,t]=W[7,t]=1
    #
    # So: 2 terms for j=2 (all k), 2 terms for j=3 (all k), 2 terms for j=0 (k=2,3), 2 terms for j=1 (k=2,3)
    # Total remaining: 8 terms. Grand total: 7 + 8 = 15. We have 1 term left!
    #
    # Wait, let me recount. The 8 terms handle:
    # - j=2, k=0,1,2,3 (1 term)
    # - j=3, k=0,1,2,3 (1 term)
    # - j=0, k=2,3 (1 term)
    # - j=1, k=2,3 (1 term)
    # That's only 4 terms! Not 8.
    #
    # So total = 7 + 4 = 11 terms. We have 5 extra terms for rank 16.
    # But we can just use rank 11! Let me verify this works.
    
    # Let me rebuild with rank 11:
    rank = 11
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Terms 0-6: Strassen for j in {0,1}, k in {0,1}
    for t_idx, (u_dict, v_dict, w_dict) in enumerate(strassen):
        t = t_idx
        for u_idx, coeff in u_dict.items():
            i = u_idx // 2
            j_local = u_idx % 2
            j_global = j_local
            U[i * 4 + j_global, t] += coeff
        for v_idx, coeff in v_dict.items():
            j_local = v_idx // 2
            k_local = v_idx % 2
            j_global = j_local
            k_global = k_local
            V[j_global * 4 + k_global, t] += coeff
        for w_idx, coeff in w_dict.items():
            k_local = w_idx // 2
            i_local = w_idx % 2
            k_global = k_local
            i_global = i_local
            W[k_global * 2 + i_global, t] += coeff
    
    # Term 7: j=2, all k
    t = 7
    U[2, t] = 1   # i=0, j=2
    U[6, t] = 1   # i=1, j=2
    V[8, t] = 1   # j=2, k=0
    V[9, t] = 1   # j=2, k=1
    V[10, t] = 1  # j=2, k=2
    V[11, t] = 1  # j=2, k=3
    W[0, t] = 1   # k=0, i=0
    W[1, t] = 1   # k=0, i=1
    W[2, t] = 1   # k=1, i=0
    W[3, t] = 1   # k=1, i=1
    W[4, t] = 1   # k=2, i=0
    W[5, t] = 1   # k=2, i=1
    W[6, t] = 1   # k=3, i=0
    W[7, t] = 1   # k=3, i=1
    
    # Term 8: j=3, all k
    t = 8
    U[3, t] = 1   # i=0, j=3
    U[7, t] = 1   # i=1, j=3
    V[12, t] = 1  # j=3, k=0
    V[13, t] = 1  # j=3, k=1
    V[14, t] = 1  # j=3, k=2
    V[15, t] = 1  # j=3, k=3
    W[0, t] = 1
    W[1, t] = 1
    W[2, t] = 1
    W[3, t] = 1
    W[4, t] = 1
    W[5, t] = 1
    W[6, t] = 1
    W[7, t] = 1
    
    # Term 9: j=0, k=2,3
    t = 9
    U[0, t] = 1   # i=0, j=0
    U[4, t] = 1   # i=1, j=0
    V[2, t] = 1   # j=0, k=2
    V[3, t] = 1   # j=0, k=3
    W[4, t] = 1   # k=2, i=0
    W[5, t] = 1   # k=2, i=1
    W[6, t] = 1   # k=3, i=0
    W[7, t] = 1   # k=3, i=1
    
    # Term 10: j=1, k=2,3
    t = 10
    U[1, t] = 1   # i=0, j=1
    U[5, t] = 1   # i=1, j=1
    V[6, t] = 1   # j=1, k=2
    V[7, t] = 1   # j=1, k=3
    W[4, t] = 1
    W[5, t] = 1
    W[6, t] = 1
    W[7, t] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
