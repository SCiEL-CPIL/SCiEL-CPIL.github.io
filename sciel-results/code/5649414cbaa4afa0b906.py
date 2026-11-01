"""
Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication.
Uses block Strassen algorithm to achieve rank 28.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16

def verify(U, V, W, n=N, m=M, p=P):
    """
    Verify that U, V, W form an exact decomposition.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    The tensor T(a,b,c) = sum_r U[a,r] V[b,r] W[c,r]
    Should be 1 if (a,b,c) corresponds to a valid multiplication term, 0 otherwise.
    Valid terms: A[i,j] * B[j,k] -> C[i,k]
    Indices:
    a = i*m + j
    b = j*p + k
    c = k*n + i
    """
    nm, mp, pn = n*m, m*p, p*n
    if not U or not V or not W:
        return False
    
    rank = len(U[0])
    
    # Check shapes
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False

    # Build a set of valid (a,b,c) indices
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                valid.add((a, b, c))
    
    # Check all entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                expected = 1 if (a, b, c) in valid else 0
                if s != expected:
                    return False
    return True

def schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    n, m, p = N, M, P
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

def strassen_decomposition_228():
    """
    Construct a rank-28 decomposition for M_{2,2,8} using 4 blocks of Strassen M_{2,2,2}.
    """
    n, m, p = N, M, P
    num_blocks = p // 2  # 4 blocks of 2x2
    rank = num_blocks * 7  # 28
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # Strassen coefficients for 2x2 x 2x2
    # M1 = (A[0,0]+A[1,1]) * (B[0,2b]+B[1,2b+1])
    # M2 = (A[1,0]+A[1,1]) * B[0,2b]
    # M3 = A[0,0] * (B[0,2b+1]-B[1,2b+1])
    # M4 = A[1,1] * (B[1,2b]-B[0,2b])
    # M5 = (A[0,0]+A[0,1]) * B[1,2b+1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,2b]+B[0,2b+1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,2b]+B[1,2b+1])
    
    # Reconstruction:
    # C[0,2b]   = M1 + M4 - M5 + M7
    # C[0,2b+1] = M3 + M5
    # C[1,2b]   = M2 + M4
    # C[1,2b+1] = M1 - M2 + M3 + M6
    
    for b in range(num_blocks):
        base = b * 7
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,2b]+B[1,2b+1])
        U[0][base + 0] += 1  # A[0,0]
        U[3][base + 0] += 1  # A[1,1]
        V[0 * p + 2 * b][base + 0] += 1  # B[0,2b]
        V[1 * p + 2 * b + 1][base + 0] += 1  # B[1,2b+1]
        W[2 * b * n + 0][base + 0] += 1  # C[0,2b]
        W[(2 * b + 1) * n + 1][base + 0] += 1  # C[1,2b+1]
        
        # M2 = (A[1,0]+A[1,1]) * B[0,2b]
        U[2][base + 1] += 1  # A[1,0]
        U[3][base + 1] += 1  # A[1,1]
        V[0 * p + 2 * b][base + 1] += 1  # B[0,2b]
        W[2 * b * n + 1][base + 1] += 1  # C[1,2b]
        W[(2 * b + 1) * n + 1][base + 1] -= 1  # C[1,2b+1]
        
        # M3 = A[0,0] * (B[0,2b+1]-B[1,2b+1])
        U[0][base + 2] += 1  # A[0,0]
        V[0 * p + 2 * b + 1][base + 2] += 1  # B[0,2b+1]
        V[1 * p + 2 * b + 1][base + 2] -= 1  # B[1,2b+1]
        W[(2 * b + 1) * n + 0][base + 2] += 1  # C[0,2b+1]
        W[(2 * b + 1) * n + 1][base + 2] += 1  # C[1,2b+1]
        
        # M4 = A[1,1] * (B[1,2b]-B[0,2b])
        U[3][base + 3] += 1  # A[1,1]
        V[1 * p + 2 * b][base + 3] += 1  # B[1,2b]
        V[0 * p + 2 * b][base + 3] -= 1  # B[0,2b]
        W[2 * b * n + 0][base + 3] += 1  # C[0,2b]
        W[2 * b * n + 1][base + 3] += 1  # C[1,2b]
        
        # M5 = (A[0,0]+A[0,1]) * B[1,2b+1]
        U[0][base + 4] += 1  # A[0,0]
        U[1][base + 4] += 1  # A[0,1]
        V[1 * p + 2 * b + 1][base + 4] += 1  # B[1,2b+1]
        W[2 * b * n + 0][base + 4] -= 1  # C[0,2b]
        W[(2 * b + 1) * n + 0][base + 4] += 1  # C[1,2b]
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,2b]+B[0,2b+1])
        U[2][base + 5] += 1  # A[1,0]
        U[0][base + 5] -= 1  # A[0,0]
        V[0 * p + 2 * b][base + 5] += 1  # B[0,2b]
        V[0 * p + 2 * b + 1][base + 5] += 1  # B[0,2b+1]
        W[(2 * b + 1) * n + 1][base + 5] += 1  # C[1,2b+1]
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,2b]+B[1,2b+1])
        U[1][base + 6] += 1  # A[0,1]
        U[3][base + 6] -= 1  # A[1,1]
        V[1 * p + 2 * b][base + 6] += 1  # B[1,2b]
        V[1 * p + 2 * b + 1][base + 6] += 1  # B[1,2b+1]
        W[2 * b * n + 0][base + 6] += 1  # C[0,2b]
    
    return U, V, W

def construct():
    """
    Entry point: return the best decomposition found.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    best_decomp = None
    best_rank = 32

    # Try rank 28 (Strassen)
    try:
        U, V, W = strassen_decomposition_228()
        if verify(U, V, W):
            best_rank = 28
            best_decomp = {"U": U, "V": V, "W": W}
        else:
            # If verification fails, try to debug by checking specific entries
            # But for now, fall back to schoolbook
            pass
    except Exception as e:
        print(f"Strassen decomposition failed: {e}")

    # Fallback to schoolbook if rank 28 failed
    if best_decomp is None:
        try:
            U, V, W = schoolbook()
            if verify(U, V, W):
                best_decomp = {"U": U, "V": V, "W": W}
            else:
                # Should not happen, but return something valid
                best_decomp = {"U": U, "V": V, "W": W}
        except Exception as e:
            print(f"Schoolbook decomposition failed: {e}")

    if time.time() > deadline:
        return best_decomp

    return best_decomp
