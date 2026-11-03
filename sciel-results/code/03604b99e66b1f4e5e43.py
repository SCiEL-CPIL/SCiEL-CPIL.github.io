import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Constructs the matrix multiplication tensor decomposition for 2x3 by 3x5.
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        # Build the tensor from the decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The (i,j,k) entry should be 1
                    val = 0.0
                    for t in range(r):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check off-diagonal entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    # Check if (a,b,c) is a valid (i,j,k) triple
                    i = a // m
                    j = a % m
                    jj = b // p
                    k = b % p
                    kk = c // n
                    ii = c % n
                    if i == ii and j == jj and k == kk:
                        continue  # This is a diagonal entry, already checked
                    val = 0.0
                    for t in range(r):
                        val += U[a, t] * V[b, t] * W[c, t]
                    if abs(val) > 1e-9:
                        return False
        return True

    def try_rank29():
        """
        Construct a rank-29 decomposition using Strassen for the 2x2x2 block
        (i in {0,1}, j in {0,1}, k in {0,1}) and schoolbook for the rest.
        """
        if time.time() > deadline:
            return None
            
        r = 29
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # 1. Strassen terms for block i in {0,1}, j in {0,1}, k in {0,1}
        # Indices in U: i*3 + j
        # Indices in V: j*5 + k
        # Indices in W: k*2 + i
        
        # Term 1: M1 = (A00 + A11) * (B00 + B11)
        # U: A00, A11 -> 1
        # V: B00, B11 -> 1
        # W: C00, C11 -> 1
        U[0*3+0, 0] = 1.0; U[1*3+1, 0] = 1.0
        V[0*5+0, 0] = 1.0; V[1*5+1, 0] = 1.0
        W[0*2+0, 0] = 1.0; W[1*2+1, 0] = 1.0
        
        # Term 2: M2 = (A10 + A11) * B00
        # U: A10, A11 -> 1
        # V: B00 -> 1
        # W: C10 -> 1, C11 -> -1
        U[1*3+0, 1] = 1.0; U[1*3+1, 1] = 1.0
        V[0*5+0, 1] = 1.0
        W[0*2+1, 1] = 1.0; W[1*2+1, 1] = -1.0
        
        # Term 3: M3 = A00 * (B01 - B11)
        # U: A00 -> 1
        # V: B01 -> 1, B11 -> -1
        # W: C01 -> 1, C11 -> 1
        U[0*3+0, 2] = 1.0
        V[0*5+1, 2] = 1.0; V[1*5+1, 2] = -1.0
        W[0*2+0, 2] = 1.0; W[1*2+1, 2] = 1.0
        
        # Term 4: M4 = A11 * (B10 - B00)
        # U: A11 -> 1
        # V: B10 -> 1, B00 -> -1
        # W: C00 -> 1, C10 -> 1
        U[1*3+1, 3] = 1.0
        V[1*5+0, 3] = 1.0; V[0*5+0, 3] = -1.0
        W[0*2+0, 3] = 1.0; W[0*2+1, 3] = 1.0
        
        # Term 5: M5 = (A00 + A01) * B11
        # U: A00, A01 -> 1
        # V: B11 -> 1
        # W: C00 -> -1, C01 -> 1
        U[0*3+0, 4] = 1.0; U[0*3+1, 4] = 1.0
        V[1*5+1, 4] = 1.0
        W[0*2+0, 4] = -1.0; W[0*2+0, 4] = -1.0 # Wait, C00 is k=0, i=0. W[0,0].
        # Correction: W index is k*n+i.
        # C00: k=0, i=0 -> W[0,0]
        # C01: k=1, i=0 -> W[1,0]
        W[0*2+0, 4] = -1.0
        W[1*2+0, 4] = 1.0
        
        # Term 6: M6 = (A10 - A00) * (B00 + B01)
        # U: A10 -> 1, A00 -> -1
        # V: B00 -> 1, B01 -> 1
        # W: C11 -> 1
        U[1*3+0, 5] = 1.0; U[0*3+0, 5] = -1.0
        V[0*5+0, 5] = 1.0; V[0*5+1, 5] = 1.0
        W[1*2+1, 5] = 1.0
        
        # Term 7: M7 = (A01 - A11) * (B10 + B11)
        # U: A01 -> 1, A11 -> -1
        # V: B10 -> 1, B11 -> 1
        # W: C00 -> 1
        U[0*3+1, 6] = 1.0; U[1*3+1, 6] = -1.0
        V[1*5+0, 6] = 1.0; V[1*5+1, 6] = 1.0
        W[0*2+0, 6] = 1.0
        
        # 2. Schoolbook terms for J=2 (row 2 of B, col 2 of A)
        # i in {0,1}, j=2, k in {0,1,2,3,4}
        col = 7
        for i in range(n):
            for k in range(p):
                U[i*3+2, col] = 1.0
                V[2*5+k, col] = 1.0
                W[k*2+i, col] = 1.0
                col += 1
        
        # 3. Schoolbook terms for J in {0,1} and K in {2,3,4}
        # i in {0,1}, j in {0,1}, k in {2,3,4}
        for i in range(n):
            for j in range(2):
                for k in range(2, p):
                    U[i*3+j, col] = 1.0
                    V[j*5+k, col] = 1.0
                    W[k*2+i, col] = 1.0
                    col += 1
        
        if col != r:
            print(f"Error: rank mismatch, expected {r}, got {col}")
            return None
            
        if verify(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        return None

    result29 = try_rank29()
    if result29 is not None:
        best_result = result29
        
    return best_result
