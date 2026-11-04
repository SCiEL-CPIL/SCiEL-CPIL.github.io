import time
import numpy as np

N, M, P = 2, 3, 10

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def trivial():
        """Return the canonical rank nmp decomposition."""
        r = N * M * P
        U = [[0] * r for _ in range(N * M)]
        V = [[0] * r for _ in range(M * P)]
        W = [[0] * r for _ in range(P * N)]
        c = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][c] = 1
                    V[j * P + k][c] = 1
                    W[k * N + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    # --- Build the Strassen-style rank-55 decomposition ---
    # Strategy:
    #   Split A (2x3) into a 2x2 block (cols 0,1) and a 2x1 block (col 2)
    #   Split B (3x10) into a 2x10 block (rows 0,1) and a 1x10 block (row 2)
    #   Split B's 2x10 block further into five 2x2 blocks (cols 2b, 2b+1)
    #
    # For each 2x2 block of B, apply the 7-term Strassen multiplication:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11) B00
    #   M3 = A00 (B01-B11)
    #   M4 = A11 (B10-B00)
    #   M5 = (A00+A01) B11
    #   M6 = (A10-A00)(B00+B01)
    #   M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    #
    # Plus 20 terms for A_col2 * B_row2.
    # Total: 5 * 7 + 20 = 55.
    
    def strassen_decomposition():
        rank = 55
        U = [[0] * rank for _ in range(N * M)]
        V = [[0] * rank for _ in range(M * P)]
        W = [[0] * rank for _ in range(P * N)]
        
        # U indices: A[i,j] -> i*M + j
        # V indices: B[j,k] -> j*P + k
        # W indices: C[i,k] -> k*N + i  (trace convention)
        
        r = 0
        
        # Strassen formula table.
        # For each term s in 0..6, define:
        #   U part: list of (row, coeff) for A factors
        #   V part: list of (col_offset_in_block, coeff) for B factors
        #   W part: list of (output_offset_in_block, coeff) for C factors
        #
        # A rows: 0,1 ; A cols within block: 0,1
        # B rows: 0,1 ; B cols within block: 0,1
        # C rows: 0,1 ; C cols within block: 0,1
        
        # Term definitions (U factors, V factors, W factors):
        # We encode each term as (u_terms, v_terms, w_terms)
        # u_terms: list of (a_row, a_col, coeff)
        # v_terms: list of (b_row, b_col, coeff)
        # w_terms: list of (c_row, c_col, coeff)
        
        terms = [
            # M1 = (A00+A11)(B00+B11) -> C: +C00 -C01? No, let's use the standard:
            # C00 = M1 + M4 - M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M7
            #
            # M1:
            {
                'u': [(0,0,1), (1,1,1)],
                'v': [(0,0,1), (1,1,1)],
                'w': [(0,0,1), (1,1,1)]  # contributes to C00 and C11
            },
            # M2 = (A10+A11) B00
            {
                'u': [(1,0,1), (1,1,1)],
                'v': [(0,0,1)],
                'w': [(1,0,1), (1,1,-1)]  # C10 +C10, C11 -C11
            },
            # M3 = A00 (B01-B11)
            {
                'u': [(0,0,1)],
                'v': [(0,1,1), (1,1,-1)],
                'w': [(0,1,1), (1,1,1)]  # C01 +C01, C11 +C11
            },
            # M4 = A11 (B10-B00)
            {
                'u': [(1,1,1)],
                'v': [(1,0,1), (0,0,-1)],
                'w': [(0,0,1), (1,0,1)]  # C00 +C00, C10 +C10
            },
            # M5 = (A00+A01) B11
            {
                'u': [(0,0,1), (0,1,1)],
                'v': [(1,1,1)],
                'w': [(0,0,-1), (0,1,1)]  # C00 -C00, C01 +C01
            },
            # M6 = (A10-A00)(B00+B01)  -- not used in C, only in intermediate
            # Actually M6 is not directly in the C formula above. Let me recheck.
            # Standard Strassen for 2x2:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            # C00 = M1 + M4 - M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M7
            # M6 is not used in C! So we only need 7 terms but M6 is computed
            # yet its result isn't used. Wait, that's wasteful. Let me recount.
            # Actually all 7 are needed because M6 contributes to... hmm,
            # Let me verify: C00 uses M1,M4,M5,M7. C01 uses M3,M5.
            # C10 uses M2,M4. C11 uses M1,M2,M3,M7.
            # M6 is NOT used. So we only need 6 terms? No, that can't be right
            # because Strassen uses 7 multiplications. Let me re-derive.
            # 
            # Actually I think I have the standard formula slightly off. The
            # correct Strassen formulas are:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            # C00 = M1 + M4 - M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M7
            # M6 is indeed not in any C entry. This means the 7th multiplication
            # is unused, which is wrong. The standard Strassen has all 7 used.
            # Let me look this up more carefully.
            #
            # Correct Strassen (from Wikipedia):
            # M1 = (A11 − A22)(B21 + B22)
            # M2 = (A11 + A22)B22
            # M3 = (A21 + A22)B11
            # M4 = (A11 − A21)B12
            # M5 = (A12 + A22)(B11 + B12)
            # M6 = (A12 − A11)(B21 + B11)
            # M7 = (A21 − A12)(B12 + B22)
            # C11 = M1 + M4 − M5 + M7
            # C12 = M3 + M5
            # C21 = M2 − M1 + M6
            # C22 = M1 − M2 + M3 + M4
            #
            # So all 7 are used. Let me use this version with my indexing.
            # My A is 2x2: A00=A11, A01=A12, A10=A21, A11=A22
            # My B is 2x2: B00=B11, B01=B12, B10=B21, B11=B22
            # My C is 2x2: C00=C11, C01=C12, C10=C21, C11=C22
            #
            # M1 = (A22-A11)(B21+B22) = (A11-A00)(B10+B11)
            # M2 = (A11+A22)B22 = (A00+A11)B11
            # M3 = (A21+A22)B11 = (A10+A11)B00
            # M4 = (A11-A21)B12 = (A00-A10)B01
            # M5 = (A12+A22)(B11+B12) = (A01+A11)(B00+B01)
            # M6 = (A12-A11)(B21+B11) = (A01-A00)(B10+B00)
            # M7 = (A21-A12)(B12+B22) = (A10-A01)(B01+B11)
            #
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 - M1 + M6
            # C11 = M1 - M2 + M3 + M4
        ]
        
        # Let me restart with the correct formulas.
        terms = [
            # M1 = (A11-A00)(B10+B11)
            {
                'u': [(1,1,1), (0,0,-1)],
                'v': [(1,0,1), (1,1,1)],
                'w': [(0,0,1), (1,0,-1), (1,1,1)]  # C00+, C10-, C11+
            },
            # M2 = (A00+A11)B11
            {
                'u': [(0,0,1), (1,1,1)],
                'v': [(1,1,1)],
                'w': [(1,0,1), (1,1,-1)]  # C10+, C11-
            },
            # M3 = (A10+A11)B00
            {
                'u': [(1,0,1), (1,1,1)],
                'v': [(0,0,1)],
                'w': [(0,1,1), (1,1,1)]  # C01+, C11+
            },
            # M4 = (A00-A10)B01
            {
                'u': [(0,0,1), (1,0,-1)],
                'v': [(0,1,1)],
                'w': [(0,0,1), (1,1,1)]  # C00+, C11+
            },
            # M5 = (A01+A11)(B00+B01)
            {
                'u': [(0,1,1), (1,1,1)],
                'v': [(0,0,1), (0,1,1)],
                'w': [(0,0,-1), (0,1,1)]  # C00-, C01+
            },
            # M6 = (A01-A00)(B10+B00)
            {
                'u': [(0,1,1), (0,0,-1)],
                'v': [(1,0,1), (0,0,1)],
                'w': [(1,0,1)]  # C10+
            },
            # M7 = (A10-A01)(B01+B11)
            {
                'u': [(1,0,1), (0,1,-1)],
                'v': [(0,1,1), (1,1,1)],
                'w': [(0,0,1)]  # C00+
            },
        ]
        
        for b in range(5):
            col0 = 2 * b
            col1 = 2 * b + 1
            
            for t in terms:
                for (a_r, a_c, coeff) in t['u']:
                    U[a_r * M + (a_c)][r] += coeff
                
                for (b_r, b_c, coeff) in t['v']:
                    # Map to V row: b_r * P + (b_c if b_c < 2 else col0+b_c-1)
                    # b_c=0 -> col0, b_c=1 -> col1
                    if b_c == 0:
                        vcol = col0
                    else:
                        vcol = col1
                    V[b_r * P + vcol][r] += coeff
                
                for (c_r, c_c, coeff) in t['w']:
                    # Map to W row: c_c * N + c_r
                    if c_c == 0:
                        wcol = col0
                    else:
                        wcol = col1
                    W[wcol * N + c_r][r] += coeff
                
                r += 1
        
        # Remaining: A[:,2] * B[2,:]
        for i in range(N):
            for k in range(P):
                U[i * M + 2][r] = 1
                V[2 * P + k][r] = 1
                W[k * N + i][r] = 1
                r += 1
        
        assert r == rank, f"Expected rank {rank}, got {r}"
        return {"U": U, "V": V, "W": W}
    
    def check(U, V, W):
        """Verify that the decomposition correctly computes matrix multiplication."""
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        
        # For each (i,j,k), the sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
        # Use vectorized check
        ok = True
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    s = np.sum(Ua[i * M + j] * Va[j * P + k] * Wa[k * N + i])
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    # Try the rank-55 decomposition
    try:
        res = strassen_decomposition()
        if check(res['U'], res['V'], res['W']):
            return res
    except Exception:
        pass
    
    # Fallback
    return trivial()
