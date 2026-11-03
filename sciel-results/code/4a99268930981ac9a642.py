"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]
import numpy as np

def _verify(U, V, W, n, m, p, seed=42):
    rng = np.random.RandomState(seed)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    rank = len(U[0])
    C_computed = np.zeros((n, p))
    for r in range(rank):
        alpha = sum(A[i, j] * U[i*m+j][r] for i in range(n) for j in range(m))
        beta = sum(B[j, k] * V[j*p+k][r] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W[k*n+i][r] * alpha * beta
    return np.allclose(C_computed, C_expected, atol=1e-10)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0]*rank for _ in range(n*m)]   # 6 x 44
    V = [[0]*rank for _ in range(m*p)]   # 24 x 44
    W = [[0]*rank for _ in range(p*n)]   # 16 x 44
    
    col = 0
    
    # Part B: A[:,2] * B[2,:] -> C += outer(A[:,2], B[2,:])
    # 16 terms: for i in {0,1}, k in {0..7}
    for i in range(2):
        for k in range(8):
            U[i*m + 2][col] = 1   # A[i,2]
            V[2*p + k][col] = 1   # B[2,k]
            W[k*n + i][col] = 1   # C[i,k]
            col += 1
    # col = 16
    
    # Part A: A[:,:2] * B[:2,:] -> C += A[:,:2] @ B[:2,:]
    # A[:,:2] is 2x2, B[:2,:] is 2x8
    # C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # Use Strassen-inspired rank-28 decomposition for 2x2 x 2x8
    # Precompute A combinations (8 of them):
    # a0 = A[0,0], a1 = A[0,1], a2 = A[1,0], a3 = A[1,1]
    # a4 = A[0,0]+A[1,1], a5 = A[0,0]-A[1,1]
    # a6 = A[0,1]+A[1,0], a7 = A[0,1]-A[1,0]
    
    # For each k, use 3 multiplications (24 total) + 4 shared = 28
    # Actually, let's use: for each k, 4 multiplications = 32, total 48. Not good.
    
    # Let me try: 28 = 4*7, so 7 per k-group of 4? Or 28 directly.
    
    # I'll use a known rank-28 construction for 2x2 x 2x8:
    # This uses the fact that 2x2 x 2x8 can be decomposed using
    # 4 precomputed A values and 7 B combinations per 2-column block.
    
    # Split 8 columns into 4 blocks of 2 columns each.
    # For each block of 2 columns (k0, k0+1):
    #   Use Strassen on the 2x2 x 2x2 sub-problem (rank 7)
    # 4 blocks * 7 = 28 multiplications.
    
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        # Strassen for 2x2 x 2x2:
        # M1 = (A00+A11) * (B0k0+B1k1)
        # M2 = (A10+A11) * B0k0
        # M3 = A00 * (B1k0+B1k1)
        # M4 = A11 * (B0k0-B1k1)
        # M5 = (A00+A01) * B1k1
        # M6 = (A10-A00) * (B0k0+B0k1)
        # M7 = (A01-A11) * (B1k0+B1k1)
        #
        # C[0,k0] = M1 + M4 - M5 + M7
        # C[0,k1] = M3 + M5
        # C[1,k0] = M2 + M4
        # C[1,k1] = M1 - M2 + M3 + M6
        
        # M1
        U[0*m+0][col] = 1; U[1*m+1][col] = 1  # A[0,0]+A[1,1]
        V[0*p+k0][col] = 1; V[1*p+k1][col] = 1  # B[0,k0]+B[1,k1]
        W[k0*n+0][col] = 1; W[k1*n+1][col] = 1  # C[0,k0] += 1, C[1,k1] += 1
        col += 1
        
        # M2
        U[1*m+0][col] = 1; U[1*m+1][col] = 1  # A[1,0]+A[1,1]
        V[0*p+k0][col] = 1  # B[0,k0]
        W[k0*n+1][col] = 1  # C[1,k0] += 1
        col += 1
        
        # M3
        U[0*m+0][col] = 1  # A[0,0]
        V[1*p+k0][col] = 1; V[1*p+k1][col] = 1  # B[1,k0]+B[1,k1]
        W[k1*n+0][col] = 1; W[k0*n+1][col] = 1  # C[0,k1] += 1, C[1,k0] += 1
        col += 1
        
        # M4
        U[1*m+1][col] = 1  # A[1,1]
        V[0*p+k0][col] = 1; V[1*p+k1][col] = -1  # B[0,k0]-B[1,k1]
        W[k0*n+0][col] = 1; W[k0*n+1][col] = 1  # C[0,k0] += 1, C[1,k0] += 1
        col += 1
        
        # M5
        U[0*m+0][col] = 1; U[0*m+1][col] = 1  # A[0,0]+A[0,1]
        V[1*p+k1][col] = 1  # B[1,k1]
        W[k0*n+0][col] = -1; W[k1*n+0][col] = 1  # C[0,k0] -= 1, C[0,k1] += 1
        col += 1
        
        # M6
        U[1*m+0][col] = 1; U[0*m+0][col] = -1  # A[1,0]-A[0,0]
        V[0*p+k0][col] = 1; V[0*p+k1][col] = 1  # B[0,k0]+B[0,k1]
        W[k1*n+1][col] = 1  # C[1,k1] += 1
        col += 1
        
        # M7
        U[0*m+1][col] = 1; U[1*m+1][col] = -1  # A[0,1]-A[1,1]
        V[1*p+k0][col] = 1; V[1*p+k1][col] = 1  # B[1,k0]+B[1,k1]
        W[k0*n+0][col] = 1  # C[0,k0] += 1
        col += 1
    
    # col should be 16 + 28 = 44
    assert col == rank, f"Expected col={rank}, got {col}"
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Decomposition failed verification")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
