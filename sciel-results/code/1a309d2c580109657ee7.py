import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _build_2x2_rank6():
    """
    Build a rank-6 decomposition of the 2x2 matrix multiplication tensor.
    Uses a known valid construction with half-integer entries.
    Returns (U6, V6, W6) each of shape (4, 6) in the 2x2 local index convention:
      A indices: 0=A00, 1=A01, 2=A10, 3=A11
      B indices: 0=B00, 1=B01, 2=B10, 3=B11
      C indices: 0=C00, 1=C01, 2=C10, 3=C11
    such that sum_r U6[a,r]*V6[b,r]*W6[c,r] = 1 iff (a,b,c) form a valid term, else 0.
    Returns None if no valid half-integer solution found.
    """
    # Try multiple candidate constructions
    candidates = [
        # Candidate 1: Based on known rank-6 2x2 MM tensor decomposition
        # Each product term: (linear combo of A) * (linear combo of B) -> contributes to C
        # We need 6 terms such that the tensor decomposes correctly.
        
        # Construction using the fact that 2x2 MM tensor has border rank 6
        # Try: products defined by pairs of indices with coefficients
        [
            # (U_coeffs for A[0..3], V_coeffs for B[0..3], W_coeffs for C[0..3])
            # Term r: sum_a U[a,r]*A[a] * sum_b V[b,r]*B[b] -> sum_c W[c,r]*C[c]
            
            # P0 = (A00+A10)(B00+B10) -> C00 + C01 + C10 + C11 (but we need to subtract)
            # This approach is tricky. Let me use a verified construction.
            
            # Known rank-6 construction for 2x2x2 tensor:
            # Using the identity from tensor decomposition literature
            # Let's try a systematic approach with the 8 equations
            
            # Actually, let me just hardcode a known valid one if I can find one.
            # The 2x2 matrix multiplication tensor has rank 7 (Strassen), not 6.
            # So rank 6 is impossible for exact decomposition!
            
            # Wait - the problem says "rank-minimal exact tensor decomposition"
            # For 2x2x2, the rank is 7 (Strassen's algorithm).
            # So we CANNOT do rank 6 for 2x2!
            
            # This means the current approach of trying rank 6 for 2x2 blocks
            # is fundamentally flawed. We should use rank 7 (Strassen) instead.
            # But the current code already has _strassen_rank28 which uses rank 7 per block.
            
            # Let me return None to indicate rank 6 is not achievable.
        ]
    ]
    
    # The 2x2 matrix multiplication tensor has rank 7 (proven by Strassen),
    # so rank 6 is impossible. Return None.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # The 2x2 matrix multiplication tensor has rank 7 (Strassen), so rank 6 is impossible.
    # Use the rank-28 Strassen-based decomposition directly.
    result = _strassen_rank28(n, m, p)
    
    # Verify the result
    if result is not None:
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        rank = U.shape[1]
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = sum(U[idx_u, ri] * V[idx_v, ri] * W[idx_w, ri] for ri in range(rank))
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return result
    
    # Ultimate fallback: schoolbook
    return _schoolbook(n, m, p)

def _strassen_rank28(n, m, p):
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # Part 2: A1 B13 (j=0,1, k=4) - 2 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    # Part 3: Strassen for (j=0,1, k=0,1) - 7 terms
    # A indices: A00=U[0], A01=U[1], A10=U[3], A11=U[4]
    # B indices: B00=V[0], B01=V[1], B10=V[5], B11=V[6]
    # C indices: C00=W[0], C01=W[1], C10=W[2], C11=W[3]
    
    # M1 = (A00+A10)(B00+B10) -> contributes to C00, C01, C10, C11
    # M2 = (A11)(B00) -> contributes to C01, C11 (with sign)
    # M3 = (A00)(B01-B11) -> contributes to C10, C11
    # M4 = (A11)(B10-B00) -> contributes to C00, C01
    # M5 = (A00+A01)(B11) -> contributes to C00, C10
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A01-A11)(B10+B11) -> contributes to C00
    
    # Standard Strassen 7 multiplications for 2x2:
    # P1 = (A00+A10)(B00+B10)
    # P2 = (A10+A11)(B00)
    # P3 = (A00)(B01-B11)
    # P4 = (A11)(B10-B00)
    # P5 = (A00+A01)(B11)
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # P1: (A00+A10)(B00+B10)
    U[0][r] = 1; U[3][r] = 1; V[0][r] = 1; V[5][r] = 1; W[0][r] = 1; W[3][r] = 1; r += 1
    # P2: (A10+A11)(B00)
    U[3][r] = 1; U[4][r] = 1; V[0][r] = 1; W[2][r] = 1; W[3][r] = -1; r += 1
    # P3: (A00)(B01-B11)
    U[0][r] = 1; V[1][r] = 1; V[6][r] = -1; W[1][r] = 1; W[3][r] = 1; r += 1
    # P4: (A11)(B10-B00)
    U[4][r] = 1; V[5][r] = 1; V[0][r] = -1; W[0][r] = 1; W[2][r] = 1; r += 1
    # P5: (A00+A01)(B11)
    U[0][r] = 1; U[1][r] = 1; V[6][r] = 1; W[0][r] = -1; W[1][r] = 1; r += 1
    # P6: (A10-A00)(B00+B01)
    U[3][r] = 1; U[0][r] = -1; V[0][r] = 1; V[1][r] = 1; W[3][r] = 1; r += 1
    # P7: (A01-A11)(B10+B11)
    U[1][r] = 1; U[4][r] = -1; V[5][r] = 1; V[6][r] = 1; W[0][r] = 1; r += 1
    
    # Part 4: Strassen for (j=0,1, k=2,3) - 7 terms
    # A indices: same as Part 3
    # B indices: B00=V[2], B01=V[3], B10=V[7], B11=V[8]
    # C indices: C00=W[4], C01=W[5], C10=W[6], C11=W[7]
    
    # P1: (A00+A10)(B00+B10)
    U[0][r] = 1; U[3][r] = 1; V[2][r] = 1; V[7][r] = 1; W[4][r] = 1; W[7][r] = 1; r += 1
    # P2: (A10+A11)(B00)
    U[3][r] = 1; U[4][r] = 1; V[2][r] = 1; W[6][r] = 1; W[7][r] = -1; r += 1
    # P3: (A00)(B01-B11)
    U[0][r] = 1; V[3][r] = 1; V[8][r] = -1; W[5][r] = 1; W[7][r] = 1; r += 1
    # P4: (A11)(B10-B00)
    U[4][r] = 1; V[7][r] = 1; V[2][r] = -1; W[4][r] = 1; W[6][r] = 1; r += 1
    # P5: (A00+A01)(B11)
    U[0][r] = 1; U[1][r] = 1; V[8][r] = 1; W[4][r] = -1; W[5][r] = 1; r += 1
    # P6: (A10-A00)(B00+B01)
    U[3][r] = 1; U[0][r] = -1; V[2][r] = 1; V[3][r] = 1; W[7][r] = 1; r += 1
    # P7: (A01-A11)(B10+B11)
    U[1][r] = 1; U[4][r] = -1; V[7][r] = 1; V[8][r] = 1; W[4][r] = 1; r += 1
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_arr[i*m+j, ri] * V_arr[j*p+k, ri] * W_arr[k*n+i, ri] for ri in range(rank))
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    if ok:
        return {"U": U, "V": V, "W": W}
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
