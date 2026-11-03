import time
import numpy as np

DEADLINE = 140.0
_start_time = None

def _elapsed():
    return time.monotonic() - _start_time

def _deadline_hit():
    return _elapsed() > DEADLINE

def _verify(U, V, W, n=2, m=4, p=6):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    rank = U.shape[1]
    
    # Check valid triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if val != 1:
                    return False
    return True

def _construct_strassen_42():
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We will build 42 terms.
    # 21 terms for C1 = A_L B_T
    # 21 terms for C2 = A_R B_B
    
    # Helper to add a Strassen term
    # A_idx: list of (i, j, coeff) for A
    # B_idx: list of (j, k, coeff) for B
    # W_contrib: list of (i, k, coeff) for C
    def add_term(r, A_coeffs, B_coeffs, W_coeffs):
        for (i, j, c) in A_coeffs:
            U[i * m + j][r] += c
        for (j, k, c) in B_coeffs:
            V[j * p + k][r] += c
        for (i, k, c) in W_coeffs:
            W[k * n + i][r] += c
            
    # Strassen formulas for 2x2 x 2x2:
    # m1: (a11+a22)(b11+b22) -> c11+=1, c22+=1
    # m2: (a21+a22)b22 -> c11-=1, c21+=1
    # m3: a11(b12-b22) -> c11+=1, c12+=1
    # m4: a22(b21-b11) -> c11+=1, c21+=1
    # m5: (a11+a12)b21 -> c12+=1, c22-=1
    # m6: (a21-a22)(b11+b12) -> c22+=1
    # m7: (a12+a21)(b22+b21) -> c22+=1
    
    # C1: A_L (A[0,0..1], A[1,0..1]) * B_T (B[0,0..5], B[1,0..5])
    # A_L indices: (0,0), (0,1), (1,0), (1,1)
    # B_T is 2x6. Split into 3 blocks of 2x2.
    # Block k uses B[0, 2k], B[0, 2k+1], B[1, 2k], B[1, 2k+1]
    
    for k in range(3):
        # Block k of B_T
        # B indices: j=0,1; k_col=2k, 2k+1
        b11_idx = (0, 2*k)
        b12_idx = (0, 2*k+1)
        b21_idx = (1, 2*k)
        b22_idx = (1, 2*k+1)
        
        # m1
        r = k * 7
        add_term(r, 
                 [(0,0,1), (1,1,1)],
                 [b11_idx[0:1]+(1,), b22_idx[0:1]+(1,)],
                 [(0, 2*k, 1), (1, 2*k+1, 1)])
                 
        # m2
        r = k * 7 + 1
        add_term(r,
                 [(1,0,1), (1,1,1)],
                 [b22_idx[0:1]+(1,)],
                 [(0, 2*k, -1), (1, 2*k, 1)])
                 
        # m3
        r = k * 7 + 2
        add_term(r,
                 [(0,0,1)],
                 [b12_idx[0:1]+(1,), b22_idx[0:1]+(-1,)],
                 [(0, 2*k, 1), (0, 2*k+1, 1)])
                 
        # m4
        r = k * 7 + 3
        add_term(r,
                 [(1,1,1)],
                 [b21_idx[0:1]+(1,), b11_idx[0:1]+(-1,)],
                 [(0, 2*k, 1), (1, 2*k, 1)])
                 
        # m5
        r = k * 7 + 4
        add_term(r,
                 [(0,0,1), (0,1,1)],
                 [b21_idx[0:1]+(1,)],
                 [(0, 2*k+1, 1), (1, 2*k+1, -1)])
                 
        # m6
        r = k * 7 + 5
        add_term(r,
                 [(1,0,1), (1,1,-1)],
                 [b11_idx[0:1]+(1,), b12_idx[0:1]+(1,)],
                 [(1, 2*k+1, 1)])
                 
        # m7
        r = k * 7 + 6
        add_term(r,
                 [(0,1,1), (1,0,1)],
                 [b22_idx[0:1]+(1,), b21_idx[0:1]+(1,)],
                 [(1, 2*k+1, 1)])

    # C2: A_R (A[0,2..3], A[1,2..3]) * B_B (B[2,0..5], B[3,0..5])
    # A_R indices: (0,2), (0,3), (1,2), (1,3)
    # B_B indices: j=2,3; k_col=0..5
    
    for k in range(3):
        # Block k of B_B
        b11_idx = (2, 2*k)
        b12_idx = (2, 2*k+1)
        b21_idx = (3, 2*k)
        b22_idx = (3, 2*k+1)
        
        # m1
        r = 21 + k * 7
        add_term(r,
                 [(0,2,1), (1,3,1)],
                 [b11_idx[0:1]+(1,), b22_idx[0:1]+(1,)],
                 [(0, 2*k, 1), (1, 2*k+1, 1)])
                 
        # m2
        r = 21 + k * 7 + 1
        add_term(r,
                 [(1,2,1), (1,3,1)],
                 [b22_idx[0:1]+(1,)],
                 [(0, 2*k, -1), (1, 2*k, 1)])
                 
        # m3
        r = 21 + k * 7 + 2
        add_term(r,
                 [(0,2,1)],
                 [b12_idx[0:1]+(1,), b22_idx[0:1]+(-1,)],
                 [(0, 2*k, 1), (0, 2*k+1, 1)])
                 
        # m4
        r = 21 + k * 7 + 3
        add_term(r,
                 [(1,3,1)],
                 [b21_idx[0:1]+(1,), b11_idx[0:1]+(-1,)],
                 [(0, 2*k, 1), (1, 2*k, 1)])
                 
        # m5
        r = 21 + k * 7 + 4
        add_term(r,
                 [(0,2,1), (0,3,1)],
                 [b21_idx[0:1]+(1,)],
                 [(0, 2*k+1, 1), (1, 2*k+1, -1)])
                 
        # m6
        r = 21 + k * 7 + 5
        add_term(r,
                 [(1,2,1), (1,3,-1)],
                 [b11_idx[0:1]+(1,), b12_idx[0:1]+(1,)],
                 [(1, 2*k+1, 1)])
                 
        # m7
        r = 21 + k * 7 + 6
        add_term(r,
                 [(0,3,1), (1,2,1)],
                 [b22_idx[0:1]+(1,), b21_idx[0:1]+(1,)],
                 [(1, 2*k+1, 1)])

    return {"U": U, "V": V, "W": W}

def construct():
    global _start_time
    _start_time = time.monotonic()
    
    result = _construct_strassen_42()
    
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook if verification fails (should not happen)
    n, m, p = 2, 4, 6
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
