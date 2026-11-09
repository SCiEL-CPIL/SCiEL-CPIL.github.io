import numpy as np
import time

DIMENSIONS = [3, 4, 5]
DEADLINE = 150.0

def _schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists for the trivial rank-60 decomposition."""
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
    return U, V, W

def _strassen_block_2x2x2(n, m, p, i0, j0, k0):
    """
    For a 2x2x2 sub-block starting at (i0, j0, k0), return the 7 Strassen rank-1 tensors
    as a list of (u_vec, v_vec, w_vec) where each vec is a dict mapping index -> coefficient.
    """
    # The 7 Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B11-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # U index = i*m + j, V index = j*p + k, W index = k*n + i
    
    # M1: u = A00+A11, v = B00+B11
    u1 = {(i0,j0):1, (i0+1,j0+1):1}
    v1 = {(j0,k0):1, (j0+1,k0+1):1}
    w1 = {}
    # C00 += M1, C01 += M1, C10 -= M1, C11 += M1
    # W[k*n+i]
    w1[(k0, i0)] = 1
    w1[(k0+1, i0)] = 1
    w1[(k0, i0+1)] = -1
    w1[(k0+1, i0+1)] = 1
    
    # M2: u = A10+A11, v = B00
    u2 = {(i0+1,j0):1, (i0+1,j0+1):1}
    v2 = {(j0,k0):1}
    w2 = {}
    # C10 += M2, C11 -= M2
    w2[(k0, i0+1)] = 1
    w2[(k0+1, i0+1)] = -1
    
    # M3: u = A00, v = B10-B00
    u3 = {(i0,j0):1}
    v3 = {(j0+1,k0):1, (j0,k0):-1}
    w3 = {}
    # C01 += M3, C11 += M3
    w3[(k0+1, i0)] = 1
    w3[(k0+1, i0+1)] = 1
    
    # M4: u = A11, v = B11-B00
    u4 = {(i0+1,j0+1):1}
    v4 = {(j0+1,k0+1):1, (j0,k0):-1}
    w4 = {}
    # C00 += M4, C10 += M4
    w4[(k0, i0)] = 1
    w4[(k0, i0+1)] = 1
    
    # M5: u = A00+A01, v = B11
    u5 = {(i0,j0):1, (i0,j0+1):1}
    v5 = {(j0+1,k0+1):1}
    w5 = {}
    # C00 -= M5, C01 += M5
    w5[(k0, i0)] = -1
    w5[(k0+1, i0)] = 1
    
    # M6: u = A10-A00, v = B00+B10
    u6 = {(i0+1,j0):1, (i0,j0):-1}
    v6 = {(j0,k0):1, (j0+1,k0):1}
    w6 = {}
    # C11 += M6
    w6[(k0+1, i0+1)] = 1
    
    # M7: u = A01-A11, v = B01+B11
    u7 = {(i0,j0+1):1, (i0+1,j0+1):-1}
    v7 = {(j0,k0+1):1, (j0+1,k0+1):1}
    w7 = {}
    # C00 += M7, C11 -= M7
    w7[(k0, i0)] = 1
    w7[(k0+1, i0+1)] = -1
    
    # Convert to vectors
    def to_vec(d, size):
        v = [0]*size
        for idx, coef in d.items():
            v[idx[0]*size//idx[1] + idx[1]] if False else None
        return v
    
    # Actually, let's just return the dicts and build the matrices later
    return [(u1,v1,w1), (u2,v2,w2), (u3,v3,w3), (u4,v4,w4), (u5,v5,w5), (u6,v6,w6), (u7,v7,w7)]

def _build_from_strassen(n, m, p, i0, j0, k0):
    """Build U, V, W for the full decomposition using one 2x2x2 Strassen block."""
    rank = n * m * p - 1  # 59
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # First, add all the "normal" terms that are NOT in the 2x2x2 block
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this is in the 2x2x2 block
                if (i0 <= i <= i0+1) and (j0 <= j <= j0+1) and (k0 <= k <= k0+1):
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Now add the 7 Strassen terms
    strassen_terms = _strassen_block_2x2x2(n, m, p, i0, j0, k0)
    for u_dict, v_dict, w_dict in strassen_terms:
        for (i, j), coef in u_dict.items():
            U[i * m + j][col] = coef
        for (j, k), coef in v_dict.items():
            V[j * p + k][col] = coef
        for (k, i), coef in w_dict.items():
            W[k * n + i][col] = coef
        col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # The tensor T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But wait, the convention is: T is the matrix multiplication tensor
    # T[(i,j), (j,k), (k,i)] = 1 if i,j,k are valid, else 0
    # And T = sum_r u_r ⊗ v_r ⊗ w_r where (u_r ⊗ v_r ⊗ w_r)[a,b,c] = u_r[a] v_r[b] w_r[c]
    
    # Let's check a few random entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected = 1
                actual = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if abs(actual - expected) > 1e-9:
                    return False
    
    # Also check that entries outside the support are 0
    # The support is {(i*m+j, j*p+k, k*n+i) for i in range(n), j in range(m), k in range(p)}
    # For any other triple (a, b, c), we need sum_r U[a,r] V[b,r] W[c,r] = 0
    # This is harder to check exhaustively, but we can check some random ones
    for _ in range(100):
        a = np.random.randint(0, n * m)
        b = np.random.randint(0, m * p)
        c = np.random.randint(0, p * n)
        
        # Check if (a, b, c) is in the support
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        
        in_support = (j == j2) and (k == k2) and (i == i2)
        
        actual = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
        if in_support:
            if abs(actual - 1) > 1e-9:
                return False
        else:
            if abs(actual) > 1e-9:
                return False
    
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try Strassen block at (0,0,0)
    U, V, W = _build_from_strassen(n, m, p, 0, 0, 0)
    
    if time.time() - start_time > DEADLINE - 10:
        return {"U": U, "V": W, "W": W}  # placeholder, shouldn't happen
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
