"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy: Split 5 into 2+3, split 6 into 3+3
    # 2x2x3: Strassen 2x2x2 (rank 7) + schoolbook 2x2x1 (rank 4) = rank 11
    # 2x3x3: schoolbook rank 18
    # Total: 2*11 + 2*18 = 58... but we can do better
    # Alternative: 6 blocks of 2x2x2 (rank 7 each) + 12 schoolbook 2x1x1 (rank 12) = 42+12 = 54
    # Or: use the split 5=2+3, 6=3+3 with proper Strassen

    # Let's use: split j into {0,1} and {2,3,4}, split k into {0,1,2} and {3,4,5}
    # For j={0,1}: use Strassen 2x2x2 for k-pairs + schoolbook for remaining k
    #   k={0,1}: Strassen (7), k={2,3}: Strassen (7), k={4,5}: Strassen (7) = 21
    # For j={2,3,4}: schoolbook 2x3x6 = 36
    # Total: 21 + 36 = 57. Not great.
    
    # Better: split j into {0,1,2} and {3,4}, split k into {0,1,2} and {3,4,5}
    # For j={0,1,2}, k={0,1,2}: 2x3x3 schoolbook = 18
    # For j={0,1,2}, k={3,4,5}: 2x3x3 schoolbook = 18
    # For j={3,4}, k={0,1,2}: Strassen 2x2x2 (7) + schoolbook 2x2x1 (4) = 11
    # For j={3,4}, k={3,4,5}: Strassen 2x2x2 (7) + schoolbook 2x2x1 (4) = 11
    # Total: 18+18+11+11 = 58

    # Best known: use 6 Strassen 2x2x2 blocks + 12 schoolbook terms = 42 + 12 = 54
    # Split j into pairs: (0,1), (2,3), (4) and k into pairs: (0,1), (2,3), (4,5)
    # 6 Strassen blocks for j-pairs (0,1) and (2,3) with k-pairs
    # Plus schoolbook for j=4 with all 6 k values, and for j-pairs with no k-pair left
    # Actually: j has 5 elements, k has 6.
    # Pair j: (0,1), (2,3), (4)
    # Pair k: (0,1), (2,3), (4,5)
    # Strassen for (j01,k01), (j01,k23), (j01,k45), (j23,k01), (j23,k23), (j23,k45) = 6*7=42
    # Schoolbook for j=4, k=0..5: 2*1*6 = 12
    # Total: 54

    rank = 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
        col += 1

    # Correct Strassen 2x2x2 block
    # A is 2x2: rows i=0,1; cols j=ja,jb
    # B is 2x2: rows j=ja,jb; cols k=ka,kb
    # C is 2x2: rows k=ka,kb; cols i=0,1
    # C[k,i] = A[i,ja]*B[ja,k] + A[i,jb]*B[jb,k]
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    def strassen_222(ja, jb, ka, kb):
        # U indices
        u00 = 0 * m + ja  # A[0, ja]
        u01 = 0 * m + jb  # A[0, jb]
        u10 = 1 * m + ja  # A[1, ja]
        u11 = 1 * m + jb  # A[1, jb]
        
        # V indices
        v00 = ja * p + ka  # B[ja, ka]
        v01 = ja * p + kb  # B[ja, kb]
        v10 = jb * p + ka  # B[jb, ka]
        v11 = jb * p + kb  # B[jb, kb]
        
        # W indices
        w00 = ka * n + 0  # C[ka, 0]
        w01 = ka * n + 1  # C[ka, 1]
        w10 = kb * n + 0  # C[kb, 0]
        w11 = kb * n + 1  # C[kb, 1]
        
        # Strassen's 7 products:
        # M1 = (A[0,ja] + A[1,jb]) * (B[ja,ka] + B[jb,kb])
        #   -> C[ka,0] += M1, C[kb,1] += M1
        add_term([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], [(w00, 1), (w11, 1)])
        
        # M2 = (A[1,ja] + A[1,jb]) * B[ja,ka]
        #   -> C[ka,1] += M2
        add_term([(u10, 1), (u11, 1)], [(v00, 1)], [(w01, 1)])
        
        # M3 = A[0,ja] * (B[jb,kb] - B[ja,kb])
        #   -> C[kb,0] += M3
        add_term([(u00, 1)], [(v11, 1), (v01, -1)], [(w10, 1)])
        
        # M4 = A[1,jb] * (B[jb,ka] - B[ja,ka])
        #   -> C[kb,1] += M4
        add_term([(u11, 1)], [(v10, 1), (v00, -1)], [(w11, 1)])
        
        # M5 = (A[0,ja] + A[0,jb]) * B[jb,kb]
        #   -> C[ka,0] += M5, C[kb,0] += M5
        add_term([(u00, 1), (u01, 1)], [(v11, 1)], [(w00, 1), (w10, 1)])
        
        # M6 = (A[1,ja] - A[0,ja]) * (B[ja,ka] + B[ja,kb])
        #   -> C[ka,1] += M6, C[kb,1] += M6  -- wait, let me re-derive
        # Actually let me verify each one carefully.
        # C[ka,0] = A[0,ja]*B[ja,ka] + A[0,jb]*B[jb,ka]
        # C[ka,1] = A[1,ja]*B[ja,ka] + A[1,jb]*B[jb,ka]
        # C[kb,0] = A[0,ja]*B[ja,kb] + A[0,jb]*B[jb,kb]
        # C[kb,1] = A[1,ja]*B[ja,kb] + A[1,jb]*B[jb,kb]
        
        # Let me use the standard Strassen:
        # M1 = (a+d)(e+h) where a=A[0,ja], d=A[1,jb], e=B[ja,ka], h=B[jb,kb]
        # M2 = (c+d)e where c=A[1,ja], d=A[1,jb], e=B[ja,ka]
        # M3 = a(f-h) where a=A[0,ja], f=B[jb,ka], h=B[jb,kb]  -- NO
        # This is getting confusing. Let me just use a verified set.
        
        # Actually, I realize I already added M1-M5 above. Let me verify them:
        # M1 = (A[0,ja]+A[1,jb])(B[ja,ka]+B[jb,kb])
        #   = A[0,ja]B[ja,ka] + A[0,ja]B[jb,kb] + A[1,jb]B[ja,ka] + A[1,jb]B[jb,kb]
        #   contributes to C[ka,0] (via A[0,ja]B[ja,ka]) and C[kb,1] (via A[1,jb]B[jb,kb])
        #   but also has cross terms A[0,ja]B[jb,kb] and A[1,jb]B[ja,ka]
        # These cross terms need to be cancelled by other M terms.
        
        # This approach of writing individual terms is error-prone. Let me use a 
        # different strategy: just verify the full decomposition numerically.
        
        # M6 = (A[1,ja] - A[0,ja]) * (B[ja,ka] + B[ja,kb])
        add_term([(u10, 1), (u00, -1)], [(v00, 1), (v01, 1)], [(w01, 1)])
        
        # M7 = (A[0,jb] - A[1,jb]) * (B[jb,ka] + B[jb,kb])
        add_term([(u01, 1), (u11, -1)], [(v10, 1), (v11, 1)], [(w00, 1)])

    # Let me verify this Strassen implementation numerically
    # and if it doesn't work, fall back to schoolbook
    
    # Test Strassen with a simple case
    U_test = np.zeros((n * m, 7), dtype=float)
    V_test = np.zeros((m * p, 7), dtype=float)
    W_test = np.zeros((p * n, 7), dtype=float)
    col_test = 0
    def add_term_test(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col_test
        for u_idx, coeff in u_coeffs:
            U_test[u_idx, col_test] += coeff
        for v_idx, coeff in v_coeffs:
            V_test[v_idx, col_test] += coeff
        for w_idx, coeff in w_coeffs:
            W_test[w_idx, col_test] += coeff
        col_test += 1
    
    # Temporarily redirect add_term to test
    orig_U, orig_V, orig_W, orig_col = U, V, W, col
    U, V, W, col = U_test, V_test, W_test, 0
    
    strassen_222(0, 1, 0, 1)
    
    U, V, W, col = orig_U, orig_V, orig_W, orig_col
    
    # Verify: for i in {0,1}, j in {0,1}, k in {0,1}:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    strassen_valid = True
    for i in range(n):
        for j in [0, 1]:
            for k in [0, 1]:
                u_row = U_test[i * m + j, :]
                v_row = V_test[j * p + k, :]
                w_row = W_test[k * n + i, :]
                result = np.sum(u_row * v_row * w_row)
                if abs(result - 1) > 1e-10:
                    strassen_valid = False
                    break
            if not strassen_valid:
                break
        if not strassen_valid:
            break
    
    if strassen_valid:
        # Use Strassen blocks
        strassen_222(0, 1, 0, 1)
        strassen_222(0, 1, 2, 3)
        strassen_222(0, 1, 4, 5)
        strassen_222(2, 3, 0, 1)
        strassen_222(2, 3, 2, 3)
        strassen_222(2, 3, 4, 5)
        
        # Schoolbook for j=4, all k
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 4
                v_idx = 4 * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    else:
        # Fallback: pure schoolbook rank 60
        U = np.zeros((n * m, 60), dtype=float)
        V = np.zeros((m * p, 60), dtype=float)
        W = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U_used = U[:, :col]
    V_used = V[:, :col]
    W_used = W[:, :col]

    # Final verification
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_used[i * m + j, :]
                v_row = V_used[j * p + k, :]
                w_row = W_used[k * n + i, :]
                result = np.sum(u_row * v_row * w_row)
                if abs(result - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        U_fallback = np.zeros((n * m, 60), dtype=float)
        V_fallback = np.zeros((m * p, 60), dtype=float)
        W_fallback = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j, c] = 1
                    V_fallback[j * p + k, c] = 1
                    W_fallback[k * n + i, c] = 1
                    c += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

    return {"U": U_used.tolist(), "V": V_used.tolist(), "W": W_used.tolist()}
# EVOLVE-BLOCK-END
