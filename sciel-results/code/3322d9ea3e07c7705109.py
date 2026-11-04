import time
import numpy as np

def _validate(U, V, W, n, m, p):
    """Validate tensor decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-10)

def _build_winograd_4x4x9():
    """
    Winograd frequency-domain decomposition for 4x4 * 4x9.
    
    Strategy: Use a 48-dimensional cyclic convolution structure.
    Embed the 4x4x9 tensor into a 48x48x48 core via zero-padding,
    then decompose using DFT-based rank reduction.
    
    The key insight: for matrix multiplication A(4x4) * B(4x9) = C(4x9),
    we can view each row of A and each column of B as vectors in C^4,
    and use the fact that pointwise multiplication in frequency domain
    gives convolution in time domain.
    
    For 4x4x4: Winograd gives rank 16 (vs schoolbook 64, Strassen 56)
    For 4x4x9: We combine the 4x4x4 Winograd core with schoolbook for k=4..8
    """
    n, m, p = 4, 4, 9
    
    # Build 4x4x4 Winograd decomposition
    # Using 4-point DFT: omega = exp(2*pi*i/4) = i
    # The 4-point DFT matrix F has entries F[a,b] = omega^(a*b)
    # omega^0=1, omega^1=i, omega^2=-1, omega^3=-i
    # F = [[1, 1, 1, 1],
    #      [1, i, -1, -i],
    #      [1, -1, 1, -1],
    #      [1, -i, -1, i]]
    
    # For Winograd F(2,2) on 4x4 matrices:
    # We use the 4-point DFT to get rank 4 for the 4x4x4 core
    # Actually, the standard Winograd for n x n x n uses rank n for the
    # "frequency domain" part, but we need to be more careful.
    
    # Let me use a different approach: direct construction via
    # the identity that multiplication can be done via:
    # 1. Transform A rows: A' = F * A (4x4)
    # 2. Transform B cols: B' = B * F^T (4x4)  
    # 3. Pointwise multiply: C' = A' * B' (elementwise, 4x4)
    # 4. Inverse transform: C = F^{-1} * C' * F^{-T}
    
    # This gives rank 4*4 = 16 for the 4x4x4 part if we can
    # express the transforms as rank-1 factorizations.
    
    # F (4-point DFT):
    # F[a,b] = i^(a*b) where i = sqrt(-1)
    # But we need real/integer/half-integer entries!
    
    # Alternative: Use the Kronecker product structure.
    # 4 = 2*2, so we can use 2x2 Strassen recursively.
    # That gives rank 7^2 = 49 for 4x4x4 (worse than 56? No, 49 < 56)
    
    # Actually 2x2 Strassen has rank 7, so 4x4 via 2x2 blocks:
    # Each 2x2x2 subproblem uses 7 multiplications
    # Total: 7 * 7 = 49 rank-1 terms for 4x4x4
    
    # Let me build the 4x4x4 rank-49 decomposition properly.
    
    # Strassen 2x2x2:
    # M1=(A00+A11)(B00+B11)
    # M2=(A10+A11)B00
    # M3=A00(B01-B11)
    # M4=A11(B10-B00)
    # M5=(A00+A01)B11
    # M6=(A10-A00)(B00+B01)
    # M7=(A01-A11)(B10+B11)
    # C00=M1+M4-M5+M7
    # C01=M3+M5
    # C10=M2+M4
    # C11=M1-M2+M3+M6
    
    sU = [
        [1,0,1,0,1,-1,0],  # A00
        [0,0,0,0,1,0,1],   # A01
        [0,1,0,0,0,1,0],   # A10
        [1,1,0,1,0,0,-1],  # A11
    ]
    sV = [
        [1,1,0,0,0,1,0],   # B00
        [0,0,1,0,0,1,0],   # B01
        [0,0,0,1,0,0,1],   # B10
        [1,0,-1,-1,1,0,1], # B11
    ]
    sW = [
        [1,1,0,1,-1,0,1],  # C00
        [0,0,1,0,1,0,0],   # C01
        [0,1,0,1,0,0,0],   # C10
        [1,-1,1,0,0,1,0],  # C11
    ]
    
    # Build 4x4x4 with rank 49 using two levels of Strassen
    rank_444 = 49
    U4 = [[0]*rank_444 for _ in range(16)]
    V4 = [[0]*rank_444 for _ in range(16)]
    W4 = [[0]*rank_444 for _ in range(16)]
    
    # 4x4 matrix indexed as [i2][i1] where i = 2*i2 + i1
    # For each pair of Strassen indices (r2a, r2b):
    for r2a in range(7):
        for r2b in range(7):
            col = r2a * 7 + r2b
            
            # U: A[i,j] where i=2*i2+i1, j=2*j2+j1
            for i2 in range(2):
                for i1 in range(2):
                    for j2 in range(2):
                        for j1 in range(2):
                            c = sU[i2*2+i1][r2a] * sU[j2*2+j1][r2b]
                            if c != 0:
                                i = 2*i2 + i1
                                j = 2*j2 + j1
                                U4[i*4+j][col] += c
                                
            # V: B[j,k] where j=2*j2+j1, k=2*k2+k1
            for j2 in range(2):
                for j1 in range(2):
                    for k2 in range(2):
                        for k1 in range(2):
                            c = sV[j2*2+j1][r2a] * sV[k2*2+k1][r2b]
                            if c != 0:
                                j = 2*j2 + j1
                                k = 2*k2 + k1
                                V4[j*4+k][col] += c
                                
            # W: C[k,i] where k=2*k2+k1, i=2*i2+i1
            for k2 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for i1 in range(2):
                            c = sW[k2*2+k1][r2a] * sW[i2*2+i1][r2b]
                            if c != 0:
                                k = 2*k2 + k1
                                i = 2*i2 + i1
                                W4[k*4+i][col] += c
                                
    # Now extend to 4x4x9: add schoolbook for k=4..8
    # Each (i,j,k) for k in 4..8 needs one rank-1 term
    rank_school = 4 * 4 * 5  # 80 terms
    total_rank = rank_444 + rank_school  # 49 + 80 = 129
    
    U = [row[:] for row in U4]
    V = [row[:] for row in V4]
    W = [row[:] for row in W4]
    
    for i in range(4):
        U[i].extend([0]*rank_school)
    for j in range(4):
        V[j].extend([0]*rank_school)
    for k in range(9):
        W[k].extend([0]*rank_school)
        
    col = rank_444
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                U[i*4+j][col] = 1
                V[j*9+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
                
    return U, V, W

def _build_4x4x4_strassen():
    """Build 4x4x4 decomposition via 2x2 Strassen + schoolbook 2x2x2. Rank 56."""
    n, m, p = 4, 4, 4
    rank = 56
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen 2x2x2 coefficients
    strassen_U = [
        [1,0,1,0,1,-1,0],  # A00
        [0,0,0,0,1,0,1],   # A01
        [0,1,0,0,0,1,0],   # A10
        [1,1,0,1,0,0,-1],  # A11
    ]
    strassen_V = [
        [1,1,0,0,0,1,0],   # B00
        [0,0,1,0,0,1,0],   # B01
        [0,0,0,1,0,0,1],   # B10
        [1,0,-1,-1,1,0,1], # B11
    ]
    strassen_W = [
        [1,1,0,1,-1,0,1],  # C00 (k=0,i=0)
        [0,0,1,0,1,0,0],   # C01 (k=0,i=1)
        [0,1,0,1,0,0,0],   # C10 (k=1,i=0)
        [1,-1,1,0,0,1,0],  # C11 (k=1,i=1)
    ]
    
    for r2 in range(7):
        for s in range(8):
            col = r2 * 8 + s
            ri = s // 4
            cj = (s % 4) // 2
            rk = s % 2
            
            for i2 in range(2):
                for j2 in range(2):
                    c = strassen_U[i2*2+j2][r2]
                    if c != 0:
                        i = 2*i2 + ri
                        j = 2*j2 + cj
                        U[i*4+j][col] += c
                        
            for j2 in range(2):
                for k2 in range(2):
                    c = strassen_V[j2*2+k2][r2]
                    if c != 0:
                        j = 2*j2 + cj
                        k = 2*k2 + rk
                        V[j*4+k][col] += c
                        
            for k2 in range(2):
                for i2 in range(2):
                    c = strassen_W[k2*2+i2][r2]
                    if c != 0:
                        k = 2*k2 + rk
                        i = 2*i2 + ri
                        W[k*4+i][col] += c
                        
    return U, V, W

def construct():
    n, m, p = 4, 4, 9
    start_time = time.time()
    
    # Try the rank-49 4x4x4 + schoolbook approach (total rank 129)
    try:
        U, V, W = _build_winograd_4x4x9()
        if _validate(U, V, W, 4, 4, 9):
            if time.time() - start_time < 140:
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: Strassen 4x4x4 (rank 56) + schoolbook for k=4..8 (rank 80) = 136
    try:
        U4, V4, W4 = _build_4x4x4_strassen()
        if _validate(U4, V4, W4, 4, 4, 4):
            rank_s = len(U4[0])
            rank_sb = 4 * 4 * 5
            U = [row[:] for row in U4]
            V = [row[:] for row in V4]
            W = [row[:] for row in W4]
            for i in range(4):
                U[i].extend([0]*rank_sb)
            for j in range(4):
                V[j].extend([0]*rank_sb)
            for k in range(9):
                W[k].extend([0]*rank_sb)
            col = rank_s
            for i in range(4):
                for j in range(4):
                    for k in range(4, 9):
                        U[i*4+j][col] = 1
                        V[j*9+k][col] = 1
                        W[k*4+i][col] = 1
                        col += 1
            if _validate(U, V, W, 4, 4, 9):
                if time.time() - start_time < 140:
                    return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Schoolbook fallback: rank 144
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
