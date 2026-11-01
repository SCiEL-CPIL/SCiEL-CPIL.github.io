"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check that no spurious terms exist
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in valid:
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if abs(val) > 1e-9:
                        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _strassen_block(n, m, p):
    """
    For multiplying (n x m) by (m x p) where n=m=2, p=16.
    The 2x16 matrix can be seen as 8 independent 2x2 blocks.
    Each 2x2x2 multiplication has rank 7 (Strassen).
    Total rank = 8 * 7 = 56.
    
    A is 2x2: indices a = i*2+j, i,j in {0,1}
    B is 2x16: indices b = j*16+k, j in {0,1}, k in {0..15}
    C is 2x16: indices c = k*2+i, k in {0..15}, i in {0,1}
    
    For block t (k in {2t, 2t+1}):
    C[4t]   = A[0,0]*B[0,2t] + A[0,1]*B[1,2t]
    C[4t+1] = A[1,0]*B[0,2t] + A[1,1]*B[1,2t]
    C[4t+2] = A[0,0]*B[0,2t+1] + A[0,1]*B[1,2t+1]
    C[4t+3] = A[1,0]*B[0,2t+1] + A[1,1]*B[1,2t+1]
    """
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7 intermediates for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B11-B00)
    # M4 = A11*(B00-B11)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    col = 0
    for t in range(num_blocks):
        # B indices for this block:
        # B[0, 2t] = index 0*16 + 2t = 2t
        # B[0, 2t+1] = index 0*16 + 2t+1 = 2t+1
        # B[1, 2t] = index 1*16 + 2t = 16+2t
        # B[1, 2t+1] = index 1*16 + 2t+1 = 16+2t+1
        
        b00 = 2*t      # B[0, 2t]
        b01 = 2*t + 1  # B[0, 2t+1]
        b10 = 16 + 2*t      # B[1, 2t]
        b11 = 16 + 2*t + 1  # B[1, 2t+1]
        
        # W indices for this block:
        # C[4t] = C[k=2t, i=0]
        # C[4t+1] = C[k=2t, i=1]
        # C[4t+2] = C[k=2t+1, i=0]
        # C[4t+3] = C[k=2t+1, i=1]
        
        w00 = 4*t      # C[2t, 0]
        w01 = 4*t + 1  # C[2t, 1]
        w10 = 4*t + 2  # C[2t+1, 0]
        w11 = 4*t + 3  # C[2t+1, 1]
        
        # U indices:
        # A[0,0] = 0, A[0,1] = 1, A[1,0] = 2, A[1,1] = 3
        u00, u01, u10, u11 = 0, 1, 2, 3
        
        # M1 = (A00+A11)*(B00+B11)
        U[u00][col] += 1; U[u11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        # M2 = (A10+A11)*B00
        U[u10][col] += 1; U[u11][col] += 1
        V[b00][col] += 1
        W[w10][col] += 1; W[w11][col] -= 1
        col += 1
        
        # M3 = A00*(B11-B00)
        U[u00][col] += 1
        V[b11][col] += 1; V[b00][col] -= 1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        
        # M4 = A11*(B00-B11)
        U[u11][col] += 1
        V[b00][col] += 1; V[b11][col] -= 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        
        # M5 = (A00+A01)*B11
        U[u00][col] += 1; U[u01][col] += 1
        V[b11][col] += 1
        W[w00][col] -= 1; W[w01][col] += 1
        col += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[u10][col] += 1; U[u00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[u01][col] += 1; U[u11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[w00][col] += 1
        col += 1
    
    return U, V, W

def _rank48_direct(n, m, p):
    """
    Rank-48 decomposition: 16 independent 2x2x1 multiplications, each of rank 3.
    
    For each k in {0..15}:
    C[k*2+0] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[k*2+1] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    This is a 2x2x1 tensor with rank 3.
    Rank-3 decomposition for 2x2x1:
    M1 = (A[0,0]+A[1,0]) * B[0,k]
    M2 = A[1,0] * (B[0,k]+B[1,k])
    M3 = A[0,1] * B[1,k]
    
    C[k*2+0] = M1 - M2 + M3 + A[0,0]*B[0,k]... 
    
    Let me verify: We need C0 = A00*B0 + A01*B1 and C1 = A10*B0 + A11*B1.
    
    Try:
    M1 = A00*B0
    M2 = A01*B1
    M3 = (A10+A11)*B0 -- no, this mixes.
    
    Correct approach for 2x2x1 (which is just two dot products sharing B):
    The tensor T[a,b,c] where a in {0,1,2,3}, b in {0,1}, c in {0,1}:
    T[0,0,0]=1, T[1,1,0]=1, T[2,0,1]=1, T[3,1,1]=1
    
    This is rank 3 (can be shown by explicit decomposition):
    M1 = (A00 + A10) * B0
    M2 = A10 * (B0 + B1)  
    M3 = A01 * B1
    
    C0 = M1 - M2 + M3 + A00*B0... 
    
    Actually, simpler: since 2x2x1 has only 2 B-entries, the rank is at most 2*2=4,
    but it's known to be 3. Let me use:
    
    M1 = A00 * B0
    M2 = A01 * B1
    M3 = (A10 + A11) * B0 -- still mixes.
    
    The cleanest rank-3 for this specific structure:
    M1 = (A00 + A10) * B0
    M2 = A10 * B1
    M3 = A01 * B1
    
    C0 = M1 + M3 - A10*B0... no.
    
    Let me just use rank 4 per k (schoolbook for 2x2x1) for a total of 64,
    or find the correct rank-3.
    
    Actually for C[i] = sum_j A[i,j]*B[j] with i,j in {0,1}:
    This is just computing two linear combinations. The tensor is:
    T = A00*B0*e0 + A01*B1*e0 + A10*B0*e1 + A11*B1*e1
    
    = B0*(A00*e0 + A10*e1) + B1*(A01*e0 + A11*e1)
    
    This can be written with 2 terms if we allow vector-valued, but for scalar rank:
    We need sum_r U[a,r]*V[b,r]*W[c,r].
    
    Try rank 3:
    r=0: U = [1,0,1,0], V = [1,0], W = [1,0]  -> A00*B0*C0
    r=1: U = [0,1,0,0], V = [0,1], W = [1,0]  -> A01*B1*C0
    r=2: U = [0,0,1,1], V = [1,0], W = [0,1]  -> (A10+A11)*B0*C1
    
    C1 = (A10+A11)*B0, but we need A10*B0 + A11*B1.
    
    Let me try:
    r=0: U=[1,0,1,0], V=[1,0], W=[1,0]
    r=1: U=[0,1,0,0], V=[0,1], W=[1,0]
    r=2: U=[0,0,1,0], V=[1,1], W=[0,1]
    
    C0 = A00*B0 + A01*B1 ✓
    C1 = A10*(B0+B1) ✗ (need A10*B0 + A11*B1)
    
    Try:
    r=0: U=[1,0,1,0], V=[1,0], W=[1,0]  -> A00*B0 contributes to C0
    r=1: U=[0,1,0,0], V=[0,1], W=[1,0]  -> A01*B1 contributes to C0
    r=2: U=[0,0,1,1], V=[1,-1], W=[0,1] -> (A10+A11)*(B0-B1) contributes to C1
    r=3: U=[0,0,1,0], V=[0,1], W=[0,1]  -> A10*B1 contributes to C1
    
    C1 = (A10+A11)*(B0-B1) + A10*B1 = A10*B0 - A10*B1 + A11*B0 - A11*B1 + A10*B1
         = A10*B0 + A11*B0 - A11*B1 ✗
    
    I'll use rank 4 per k for simplicity: total rank 64. Not better than schoolbook.
    
    Let me just return the Strassen 56-rank solution.
    """
    return _strassen_block(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Try rank-32 circulant construction
    try:
        U, V, W = _rank32_circulant(n, m, p)
        if time.time() < deadline and _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Try Strassen-based construction (rank 56)
    try:
        U, V, W = _strassen_block(n, m, p)
        if time.time() < deadline and _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fall back to schoolbook (rank 64)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
