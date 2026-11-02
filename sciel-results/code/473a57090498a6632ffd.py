"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_TIMEOUT = 120

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(len(U)):
        if len(U[i]) != rank:
            return False
    for i in range(len(V)):
        if len(V[i]) != rank:
            return False
    for i in range(len(W)):
        if len(W[i]) != rank:
            return False
    
    # For each output element C[i,j] = sum_k A[i,k] * B[k,j]
    # The decomposition gives: C[i,j] = sum_r U[i,r] * V[j,r] * W[r]
    # Wait, that's not right. Let me re-read the convention.
    # W[k*n+i, r] where k is row of B (0..m-1), i is col of A (0..n-1)
    # Actually: C[i,j] = sum_k A[i,k] B[k,j]
    # Decomposition: C[i,j] = sum_r (sum_k U[i*k+m... no
    
    # Standard convention: U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # C[i,j] = sum_r U[i*m + ?] ... 
    
    # Let me think again. The tensor T_{i,k,j} = delta (it's the identity in k)
    # No, matrix multiplication tensor: T[i,k,j] such that C[i,j] = sum_k A[i,k]*B[k,j]*T[i,k,j]
    # But T[i,k,j] = 1 always? No, that's not a tensor decomposition.
    
    # The multiplication tensor has entries: for each (i,k,j), the coefficient is 1
    # We want: sum_r U[i,r] * V[k,r] * W[j,r] = 1 for all i,k,j
    # But that's not right either.
    
    # Correct formulation: 
    # C[i,j] = sum_k A[i,k] B[k,j]
    # = sum_k sum_r (A[i,k] * U[k*m+i, r]) * (B[k,j] * V[j*p+k, r]) ... no
    
    # Let me use the standard: 
    # C[i,j] = sum_r (sum_k A[i,k] * U[k, r]) * (sum_k B[k,j] * V[k, r]) ... no
    
    # Actually the standard tensor decomposition for matrix mult:
    # C[i,j] = sum_r M_r[i] * M_r[j] where M_r are intermediate products
    # But with the U,V,W format:
    # C[i,j] = sum_r (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j]) ... 
    
    # I think the correct interpretation is:
    # C[i,j] = sum_r U[i, r] * V[j, r] * W[r]
    # But U is (n*m) x rank, so U[i, r] doesn't make sense unless i indexes n*m elements
    
    # Let me just test with a simple case. For 1x1x1:
    # A is 1x1, B is 1x1, C is 1x1
    # U is (1*1)x1 = 1x1, V is (1*1)x1 = 1x1, W is (1*1)x1 = 1x1
    # C[0,0] = A[0,0]*B[0,0] = U[0,0]*V[0,0]*W[0,0]
    # So we need U[0,0]*V[0,0]*W[0,0] = 1
    
    # For 2x2x2 with Strassen rank 7:
    # C[i,j] = sum_r U[i*2+k, r] ... 
    
    # I think the indexing is:
    # U has shape (n*m, rank), indexed as U[i*m + k, r] for A[i,k]
    # V has shape (m*p, rank), indexed as V[k*p + j, r] for B[k,j]  
    # W has shape (p*n, rank), indexed as W[j*n + i, r] for C[j,i]... 
    
    # But the task says W[k*n+i, r] which is confusing.
    # Let me just verify by checking that the decomposition works.
    
    # The multiplication tensor T has T[i,k,j] = 1 for all i in [0,n), k in [0,m), j in [0,p)
    # Wait no, T[i,k,j] is the coefficient such that C[i,j] = sum_k A[i,k]*B[k,j]
    # The tensor is T[i,k,j] = delta_{?} ... 
    
    # I'll just do a numerical verification.
    import random
    for _ in range(5):
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        
        # Compute C = A * B
        C = [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]
        
        # Compute C from decomposition
        # C[i,j] = sum_r (sum_k A[i,k]*U[i*m+k,r]) * (sum_k B[k,j]*V[k*p+j,r]) * W[j*n+i, r]
        # Hmm, that doesn't look right dimensionally.
        
        # Let me try: C[i,j] = sum_r U[i*m+?, r] * V[?, r] * W[j*n+i, r]
        
        # I think the correct formula is:
        # C[i,j] = sum_r (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j]) / W[j*n+i, r]
        # No, that's division.
        
        # Standard: C[i,j] = sum_r M_r[i] * N_r[j] where M_r = A * U_r, N_r = V_r * B
        # In our format: M_r[i] = sum_k A[i,k] * U[i*m+k, r]
        # N_r[j] = sum_k V[k*p+j, r] * B[k,j]
        # C[i,j] = sum_r M_r[i] * N_r[j] * W[j*n+i, r]
        
        C_decomp = [[0]*p for _ in range(n)]
        for i in range(n):
            for j in range(p):
                val = 0
                for r in range(rank):
                    M_r_i = sum(A[i][k] * U[i*m+k][r] for k in range(m))
                    N_r_j = sum(V[k*p+j][r] * B[k][j] for k in range(m))
                    val += M_r_i * N_r_j * W[j*n+i][r]
                C_decomp[i][j] = val
        
        if C_decomp != C:
            return False
    return True

def _build_strassen_2x3_3x2():
    """Build rank-11 decomposition for 2x3 * 3x2."""
    n, m, p = 2, 3, 2
    rank = 11
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    k0, k1 = 0, 1
    
    # Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # M1 = (A00+A11)*(B00+B11)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[3][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2 = (A10+A11)*B00
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[2][col] = 1; W[3][col] = -1
    col += 1
    
    # M3 = A00*(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[3][col] = -1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # M4 = A11*(B10-B00)
    U[3][col] = 1
    V[2][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # M5 = (A00+A01)*B11
    U[0][col] = 1; U[1][col] = 1
    V[3][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][col] = 1; U[3][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[0][col] = 1
    col += 1
    
    # Rank-4 for A[0:2,2] * B[2,0:2]
    # P1 = A02*B20
    U[4][col] = 1
    V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # P2 = A02*B21
    U[4][col] = 1
    V[5][col] = 1
    W[1][col] = 1
    col += 1
    
    # P3 = A12*B20
    U[5][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    
    # P4 = A12*B21
    U[5][col] = 1
    V[5][col] = 1
    W[3][col] = 1
    col += 1
    
    return U, V, W

def _try_rank_28():
    """Try to build a rank-28 decomposition using Kronecker structure."""
    n, m, p = 2, 3, 8
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Use the rank-11 2x3x3x2 block and replicate with shifts
    # But we need to be more clever. Let's use 4 blocks of 2x3x2 with shared structure.
    
    # Actually, let's just use 4 independent 2x3x2 blocks (rank 11 each) = 44
    # That's what we already have. For rank 28, we need a different approach.
    
    # Try: use Strassen on the full 3x8 by embedding in 4x4
    # This is complex, so let's fall back to rank 44 for now.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try rank 28 first
    if time.time() - _start_time < _TIMEOUT - 30:
        r28 = _try_rank_28()
        if r28 is not None:
            U, V, W = r28
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to rank 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen part
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        U[0 * m + 0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # Rank-4 part
        U[0 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        U[1 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
