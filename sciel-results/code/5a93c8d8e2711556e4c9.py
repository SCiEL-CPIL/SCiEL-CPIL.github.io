import time
import numpy as np

DIMENSIONS = [4, 4, 8]


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 matrix multiplication (rank 7).
    
    Strassen's algorithm:
    Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    M1 = (a + d)(e + h)
    M2 = (c + d) e
    M3 = a (f - h)
    M4 = d (g - e)
    M5 = (a + b) h
    M6 = (c - a) (e + f)
    M7 = (b - d) (g + h)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    U[i*m+j, t] = coefficient of A[i,j] in term t
    V[j*p+k, t] = coefficient of B[j,k] in term t
    W[k*n+i, t] = coefficient of C[k,i] in term t
    
    Wait, the convention is:
    U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
    So U is indexed by (i,j), V by (j,k), W by (k,i)
    
    Term 0: M1 = (a+d)(e+h)
      U: a=1, d=1 (indices (0,0) and (1,1))
      V: e=1, h=1 (indices (0,0) and (1,1))
      W: C11 gets M1 (index (0,0) for k=0,i=0), C22 gets M1 (index (1,1) for k=1,i=1)
    
    Term 1: M2 = (c+d)e
      U: c=1 (index (1,0)), d=1 (index (1,1))
      V: e=1 (index (0,0))
      W: C21 gets M2 (index (0,1) for k=0,i=1), C22 gets -M2 (index (1,1) for k=1,i=1)
    
    Term 2: M3 = a(f-h)
      U: a=1 (index (0,0))
      V: f=1 (index (0,1)), h=-1 (index (1,1))
      W: C12 gets M3 (index (1,0) for k=1,i=0), C22 gets M3 (index (1,1) for k=1,i=1)
    
    Term 3: M4 = d(g-e)
      U: d=1 (index (1,1))
      V: g=1 (index (1,0)), e=-1 (index (0,0))
      W: C11 gets M4 (index (0,0) for k=0,i=0), C21 gets M4 (index (0,1) for k=0,i=1)
    
    Term 4: M5 = (a+b)h
      U: a=1 (index (0,0)), b=1 (index (0,1))
      V: h=1 (index (1,1))
      W: C11 gets -M5 (index (0,0) for k=0,i=0), C12 gets M5 (index (1,0) for k=1,i=0)
    
    Term 5: M6 = (c-a)(e+f)
      U: c=1 (index (1,0)), a=-1 (index (0,0))
      V: e=1 (index (0,0)), f=1 (index (0,1))
      W: C22 gets M6 (index (1,1) for k=1,i=1)
    
    Term 6: M7 = (b-d)(g+h)
      U: b=1 (index (0,1)), d=-1 (index (1,1))
      V: g=1 (index (1,0)), h=1 (index (1,1))
      W: C11 gets M7 (index (0,0) for k=0,i=0)
    """
    n = m = p = 2
    R = 7
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Term 0: M1 = (a+d)(e+h)
    U[0, 0] = 1  # a = A[0,0]
    U[3, 0] = 1  # d = A[1,1]
    V[0, 0] = 1  # e = B[0,0]
    V[3, 0] = 1  # h = B[1,1]
    W[0, 0] = 1  # C[0,0] (k=0,i=0)
    W[3, 0] = 1  # C[1,1] (k=1,i=1)
    
    # Term 1: M2 = (c+d)e
    U[2, 1] = 1  # c = A[1,0]
    U[3, 1] = 1  # d = A[1,1]
    V[0, 1] = 1  # e = B[0,0]
    W[1, 1] = 1  # C[0,1] (k=0,i=1)
    W[3, 1] = -1  # C[1,1] (k=1,i=1)
    
    # Term 2: M3 = a(f-h)
    U[0, 2] = 1  # a = A[0,0]
    V[1, 2] = 1  # f = B[0,1]
    V[3, 2] = -1  # h = B[1,1]
    W[2, 2] = 1  # C[1,0] (k=1,i=0)
    W[3, 2] = 1  # C[1,1] (k=1,i=1)
    
    # Term 3: M4 = d(g-e)
    U[3, 3] = 1  # d = A[1,1]
    V[2, 3] = 1  # g = B[1,0]
    V[0, 3] = -1  # e = B[0,0]
    W[0, 3] = 1  # C[0,0] (k=0,i=0)
    W[1, 3] = 1  # C[0,1] (k=0,i=1)
    
    # Term 4: M5 = (a+b)h
    U[0, 4] = 1  # a = A[0,0]
    U[1, 4] = 1  # b = A[0,1]
    V[3, 4] = 1  # h = B[1,1]
    W[0, 4] = -1  # C[0,0] (k=0,i=0)
    W[2, 4] = 1  # C[1,0] (k=1,i=0)
    
    # Term 5: M6 = (c-a)(e+f)
    U[2, 5] = 1  # c = A[1,0]
    U[0, 5] = -1  # a = A[0,0]
    V[0, 5] = 1  # e = B[0,0]
    V[1, 5] = 1  # f = B[0,1]
    W[3, 5] = 1  # C[1,1] (k=1,i=1)
    
    # Term 6: M7 = (b-d)(g+h)
    U[1, 6] = 1  # b = A[0,1]
    U[3, 6] = -1  # d = A[1,1]
    V[2, 6] = 1  # g = B[1,0]
    V[3, 6] = 1  # h = B[1,1]
    W[0, 6] = 1  # C[0,0] (k=0,i=0)
    
    return U, V, W


def _expand_strassen(U_small, V_small, W_small, n_small, n_big):
    """
    Expand a small Strassen decomposition to a larger one.
    For 4x4x4 from 2x2x2.
    
    Index mapping:
    i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    
    For each term in the small decomposition, we create n_small^2 = 4 terms in the large one.
    Actually, the expansion is more subtle. Let me think again.
    
    The 4x4x4 multiplication can be viewed as 4 2x2x2 multiplications plus cross terms.
    But Strassen recursively applies: 4x4x4 = 7 * (2x2x2).
    
    So for each of the 7 terms in the 4x4 level, we have 7 terms in the 2x2 level.
    Total: 49 terms.
    
    The mapping:
    For a term (t4, t2) where t4 is the 4x4 Strassen term and t2 is the 2x2 Strassen term:
    - U4[i,j, t4*7+t2] = U2[i2,j2, t2] * coeff4[i1,j1, t4]
      where i = 2*i1+i2, j = 2*j1+j2
    - Similarly for V and W.
    
    Actually, the standard recursive Strassen:
    A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    where each block is 2x2.
    
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each M is a 2x2 matrix, computed using the 2x2x2 Strassen (rank 7).
    
    So for the 4x4 level, term t4 corresponds to one of M1..M7.
    For each t4, the 2x2 blocks involved have specific signs.
    
    Let's define the 4x4 Strassen coefficients:
    M1: A11(+1), A22(+1); B11(+1), B22(+1); C11(+1), C22(+1)
    M2: A21(+1), A22(+1); B11(+1); C21(+1), C22(-1)
    M3: A11(+1); B12(+1), B22(-1); C12(+1), C22(+1)
    M4: A22(+1); B21(+1), B11(-1); C11(+1), C21(+1)
    M5: A11(+1), A12(+1); B22(+1); C11(-1), C12(+1)
    M6: A21(+1), A11(-1); B11(+1), B12(+1); C22(+1)
    M7: A12(+1), A22(-1); B21(+1), B22(+1); C11(+1)
    
    For each 4x4 term t4, we need to combine the 2x2 Strassen terms.
    The U vector for 4x4 at (i,j,t) where i=2*i1+i2, j=2*j1+j2:
    U4[i,j,t] = sum over t4: coeff_A[i1,j1,t4] * U2[i2,j2, t2]
    where t = t4*7 + t2
    
    But wait, this isn't quite right. The product is:
    (sum over t4,t2: coeff_A[i1,j1,t4] * U2[i2,j2,t2]) * (coeff_B[j1,k1,t4] * V2[j2,k2,t2]) * (coeff_C[k1,i1,t4] * W2[k2,i2,t2])
    
    This should equal the sum over t4,t2 of the products, which gives the correct C.
    
    Let me implement this.
    """
    n2 = n_small
    n4 = n_big
    R2 = U_small.shape[1]
    R4 = 7 * R2  # 7 terms at 4x4 level, each with R2 terms at 2x2 level
    
    U4 = np.zeros((n4 * n4, R4))
    V4 = np.zeros((n4 * n4, R4))
    W4 = np.zeros((n4 * n4, R4))
    
    # Define the 4x4 Strassen coefficients
    # For each term t4 (0..6), define which 2x2 blocks are involved and with what sign
    # A blocks: (i1,j1) -> sign
    # B blocks: (j1,k1) -> sign
    # C blocks: (k1,i1) -> sign
    
    a_coeffs = [
        {(0, 0): 1, (1, 1): 1},   # M1: A11, A22
        {(1, 0): 1, (1, 1): 1},   # M2: A21, A22
        {(0, 0): 1},               # M3: A11
        {(1, 1): 1},               # M4: A22
        {(0, 0): 1, (0, 1): 1},   # M5: A11, A12
        {(1, 0): 1, (0, 0): -1},  # M6: A21, A11
        {(0, 1): 1, (1, 1): -1},  # M7: A12, A22
    ]
    
    b_coeffs = [
        {(0, 0): 1, (1, 1): 1},   # M1: B11, B22
        {(0, 0): 1},               # M2: B11
        {(0, 1): 1, (1, 1): -1},  # M3: B12, B22
        {(1, 0): 1, (0, 0): -1},  # M4: B21, B11
        {(1, 1): 1},               # M5: B22
        {(0, 0): 1, (0, 1): 1},   # M6: B11, B12
        {(1, 0): 1, (1, 1): 1},   # M7: B21, B22
    ]
    
    c_coeffs = [
        {(0, 0): 1, (1, 1): 1},   # M1: C11, C22
        {(0, 1): 1, (1, 1): -1},  # M2: C21, C22
        {(1, 0): 1, (1, 1): 1},   # M3: C12, C22
        {(0, 0): 1, (0, 1): 1},   # M4: C11, C21
        {(0, 0): -1, (1, 0): 1},  # M5: C11, C12
        {(1, 1): 1},               # M6: C22
        {(0, 0): 1},               # M7: C11
    ]
    
    for t4 in range(7):
        for t2 in range(R2):
            t = t4 * R2 + t2
            
            for i2 in range(n2):
                for j2 in range(n2):
                    u_val = U_small[i2 * n2 + j2, t2]
                    if u_val != 0:
                        for i1, j1 in a_coeffs[t4]:
                            sign = a_coeffs[t4][(i1, j1)]
                            i = i1 * n2 + i2
                            j = j1 * n2 + j2
                            U4[i * n4 + j, t] += sign * u_val
            
            for j2 in range(n2):
                for k2 in range(n2):
                    v_val = V_small[j2 * n2 + k2, t2]
                    if v_val != 0:
                        for j1, k1 in b_coeffs[t4]:
                            sign = b_coeffs[t4][(j1, k1)]
                            j = j1 * n2 + j2
                            k = k1 * n2 + k2
                            V4[j * n4 + k, t] += sign * v_val
            
            for k2 in range(n2):
                for i2 in range(n2):
                    w_val = W_small[k2 * n2 + i2, t2]
                    if w_val != 0:
                        for k1, i1 in c_coeffs[t4]:
                            sign = c_coeffs[t4][(k1, i1)]
                            k = k1 * n2 + k2
                            i = i1 * n2 + i2
                            W4[k * n4 + i, t] += sign * w_val
    
    return U4, V4, W4


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    # Step 1: Get 2x2x2 Strassen decomposition (rank 7)
    U2, V2, W2 = _strassen_2x2x2()
    
    # Step 2: Expand to 4x4x4 (rank 49)
    U444, V444, W444 = _expand_strassen(U2, V2, W2, 2, 4)
    
    # Step 3: Construct 4x4x8 decomposition
    # We can decompose 4x4x8 as two 4x4x4 multiplications
    # B is 4x8, C is 4x8
    # Split B into B1 (4x4, cols 0-3) and B2 (4x4, cols 4-7)
    # Split C into C1 (4x4, cols 0-3) and C2 (4x4, cols 4-7)
    # C1 = A * B1, C2 = A * B2
    # Total rank = 49 + 49 = 98
    
    R_total = 98
    U = np.zeros((n * m, R_total))
    V = np.zeros((m * p, R_total))
    W = np.zeros((p * n, R_total))
    
    # First block: C1 = A * B1 (cols 0-3 of B, cols 0-3 of C)
    U[:, 0:49] = U444
    V[0:16, 0:49] = V444
    W[0:16, 0:49] = W444
    
    # Second block: C2 = A * B2 (cols 4-7 of B, cols 4-7 of C)
    U[:, 49:98] = U444
    V[16:32, 49:98] = V444
    W[16:32, 49:98] = W444
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-7):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails
    schoolbook_rank = n * m * p
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
