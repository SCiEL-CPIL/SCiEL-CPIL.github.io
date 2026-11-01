"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    np.random.seed(42)
    for _ in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(C[i, k] - val) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin
    
    # Build the Strassen-based rank-39 decomposition
    # 5 blocks of 2x2 (Strassen, rank 7 each) = 35
    # 1 block of 2x1 (Schoolbook, rank 4) = 4
    # Total rank = 39
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # 5 Strassen blocks for columns 0-9 (5 pairs of columns)
    for b in range(5):
        # B cols: 2*b, 2*b+1
        v00 = 2*b
        v01 = 2*b + 1
        v10 = 11 + 2*b
        v11 = 12 + 2*b
        
        # W indices for C block:
        # C[k*n+i] where k is column index, i is row index
        # For columns 2*b and 2*b+1:
        w00 = (2*b)*n + 0      # C[2*b, 0]
        w01 = (2*b+1)*n + 0    # C[2*b+1, 0]
        w10 = (2*b)*n + 1      # C[2*b, 1]
        w11 = (2*b+1)*n + 1    # C[2*b+1, 1]
        
        # U indices for A:
        # A[i*m+j]: u00=0, u01=1, u10=2, u11=3
        
        # P1: (a00+a11)(b00+b11)
        U[0][r] += 1; U[3][r] += 1
        V[v00][r] += 1; V[v11][r] += 1
        W[w00][r] += 1; W[w11][r] += 1
        r += 1
        
        # P2: (a10+a11)b00
        U[2][r] += 1; U[3][r] += 1
        V[v00][r] += 1
        W[w10][r] += 1; W[w11][r] -= 1
        r += 1
        
        # P3: a00(b01-b11)
        U[0][r] += 1
        V[v01][r] += 1; V[v11][r] -= 1
        W[w01][r] += 1; W[w11][r] += 1
        r += 1
        
        # P4: a11(b10-b00)
        U[3][r] += 1
        V[v10][r] += 1; V[v00][r] -= 1
        W[w00][r] += 1; W[w10][r] += 1
        r += 1
        
        # P5: (a00+a01)b11
        U[0][r] += 1; U[1][r] += 1
        V[v11][r] += 1
        W[w00][r] -= 1; W[w01][r] += 1
        r += 1
        
        # P6: (a10-a00)(b00+b01)
        U[2][r] += 1; U[0][r] -= 1
        V[v00][r] += 1; V[v01][r] += 1
        W[w11][r] += 1
        r += 1
        
        # P7: (a01-a11)(b10+b11)
        U[1][r] += 1; U[3][r] -= 1
        V[v10][r] += 1; V[v11][r] += 1
        W[w00][r] += 1
        r += 1

    # 1 block of 2x1 at col 10
    # B[0, 10] -> V[10]
    # B[1, 10] -> V[21]
    # C[0, 10] -> W[20] (10*2+0)
    # C[1, 10] -> W[21] (10*2+1)
    
    # Q1: a00 * B[0,10] -> C[0,10]
    U[0][r] += 1
    V[10][r] += 1
    W[20][r] += 1
    r += 1
    
    # Q2: a01 * B[1,10] -> C[0,10]
    U[1][r] += 1
    V[21][r] += 1
    W[20][r] += 1
    r += 1
    
    # Q3: a10 * B[0,10] -> C[1,10]
    U[2][r] += 1
    V[10][r] += 1
    W[21][r] += 1
    r += 1
    
    # Q4: a11 * B[1,10] -> C[1,10]
    U[3][r] += 1
    V[21][r] += 1
    W[21][r] += 1
    r += 1
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col] = 1
                    V_s[j*p+k][col] = 1
                    W_s[k*n+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    # Now attempt to reduce rank by finding linear dependencies
    # Convert to numpy for SVD-based rank analysis
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to find columns that are linearly dependent
    # The key insight: if we can express one rank-1 term as a linear combination
    # of others, we can eliminate it
    
    # Build the tensor and check for rank reduction opportunities
    # For each column c, the term is U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # We want to find if any column can be removed while preserving the tensor
    
    # Strategy: try removing each column one at a time and check if the tensor
    # can still be represented by the remaining columns
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try greedy removal: for each column, check if it's redundant
    # This is O(rank^2) in the worst case, but rank is small (39)
    
    # First, let's try a simpler approach: check if the last 4 columns (schoolbook part)
    # can be merged with Strassen columns
    
    # Actually, let's try a different structural approach:
    # Use a 2x2 by 2x3 Strassen-like decomposition for the first 6 columns
    # and handle the remaining 5 columns separately
    
    # But first, let's try to see if we can reduce the rank by checking
    # linear dependencies in the U, V, W matrices
    
    # The rank of the tensor is at most the minimum of:
    # - rank of U matrix
    # - rank of V matrix  
    # - rank of W matrix
    # But this is a lower bound, not the tensor rank
    
    # Let's try a different approach: use the fact that for 2x2 by 2x11,
    # we might be able to use a more efficient decomposition
    
    # Try: 2x2 by 2x11 = 2x2 by (2x3 + 2x3 + 2x3 + 2x1)
    # If we have a rank-9 decomposition for 2x2 by 2x3, then:
    # 3 * 9 + 4 = 31
    
    # But I don't know a rank-9 decomposition for 2x2 by 2x3
    # Let's stick with the rank-39 decomposition for now
    
    # Try to verify and return
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
