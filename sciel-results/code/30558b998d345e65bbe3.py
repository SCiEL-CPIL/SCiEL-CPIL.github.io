import time
import numpy as np

def _build_strassen_2x2x2_block(k0, k1, col_start):
    """Build 7 Strassen terms for a 2x2x2 block with k indices k0, k1.
    Returns list of (U_entries, V_entries, W_entries) tuples.
    U entries: list of (row, val)
    V entries: list of (row, val)  
    W entries: list of (row, val)
    """
    terms = []
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M7
    
    # Our A is 2x2: A[i,j] where i in {0,1} (rows), j in {1,2} (cols, but we index as 0,1 locally)
    # Our B is 2x2: B[j,k] where j in {1,2}, k in {k0,k1}
    # Our C is 2x2: C[i,k] where i in {0,1}, k in {k0,k1}
    
    # M1 = (A[0,1]+A[1,2]) * (B[1,k0]+B[2,k1])
    terms.append((
        [(0*m+1, 1), (1*m+2, 1)],
        [(1*p+k0, 1), (2*p+k1, 1)],
        [(k0*n+0, 1), (k1*n+1, 1)]
    ))
    
    # M2 = (A[1,1]+A[1,2]) * B[1,k0]
    terms.append((
        [(1*m+1, 1), (1*m+2, 1)],
        [(1*p+k0, 1)],
        [(k0*n+1, 1), (k1*n+1, -1)]
    ))
    
    # M3 = A[0,1] * (B[1,k1]-B[2,k1])
    terms.append((
        [(0*m+1, 1)],
        [(1*p+k1, 1), (2*p+k1, -1)],
        [(k1*n+0, 1), (k1*n+1, 1)]
    ))
    
    # M4 = A[1,2] * (B[2,k0]-B[1,k0])
    terms.append((
        [(1*m+2, 1)],
        [(2*p+k0, 1), (1*p+k0, -1)],
        [(k0*n+0, 1), (k0*n+1, 1)]
    ))
    
    # M5 = (A[0,1]+A[0,2]) * B[2,k1]
    terms.append((
        [(0*m+1, 1), (0*m+2, 1)],
        [(2*p+k1, 1)],
        [(k0*n+0, -1), (k1*n+0, 1)]
    ))
    
    # M6 = (A[1,1]-A[0,1]) * (B[1,k0]+B[1,k1])
    terms.append((
        [(1*m+1, 1), (0*m+1, -1)],
        [(1*p+k0, 1), (1*p+k1, 1)],
        [(k1*n+1, 1)]
    ))
    
    # M7 = (A[0,2]-A[1,2]) * (B[2,k0]+B[2,k1])
    terms.append((
        [(0*m+2, 1), (1*m+2, -1)],
        [(2*p+k0, 1), (2*p+k1, 1)],
        [(k0*n+0, 1)]
    ))
    
    return terms

def _try_rank_40():
    """Try to find a rank-40 decomposition.
    Strategy: Use 16 terms for j=0 (schoolbook) and 24 terms for j=1,2.
    For j=1,2: 4 blocks of 2x2x2, each using 6 terms (one Strassen term merged).
    """
    n, m, p = 2, 3, 8
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    rank = 40
    
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    col = 0
    
    # Part 1: 2x1x8 part (j=0) - 16 terms
    for k in range(p):
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
    
    # Part 2: 2x2x8 part (j=1,2) - try 6 terms per block = 24 terms
    # This is a hypothesis; may not work. Return None if verification fails.
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        terms = _build_strassen_2x2x2_block(k0, k1, col)
        # Use only 6 of the 7 terms - this won't work in general
        # So this approach is invalid. Return None.
        pass
    
    # This approach doesn't work cleanly. Return None to indicate failure.
    return None

def _verify_fast(U, V, W):
    """Fast verification using numpy."""
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check all (a,b,c) triples
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j2 = b // p
            k = b % p
            for c in range(p * n):
                i2 = c % n
                k2 = c // n
                expected = 1 if (i == i2 and j == j2 and k == k2) else 0
                
                val = 0
                for r in range(R):
                    val += U_np[a, r] * V_np[b, r] * W_np[c, r]
                
                if abs(val - expected) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 2, 3, 8
    
    # Dimensions of factors
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    # Start with known rank-44 solution
    rank = 44
    
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    col = 0
    
    # Part 1: 2x1x8 part (j=0)
    # A[:, 0] * B[0, :]
    # For each k in 0..7
    for k in range(p):
        # Term 1: A[0,0] * B[0,k] -> C[0,k]
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 2: A[1,0] * B[0,k] -> C[1,k]
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    # Part 2: 2x2x8 part (j=1,2)
    # Split into 4 blocks of 2x2x2
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # P1: (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P2: (A[1,1] + A[1,2]) * B[1,k0]
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # P3: A[0,1] * (B[1,k1] - B[2,k1])
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P4: A[1,2] * (B[2,k0] - B[1,k0])
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P5: (A[0,1] + A[0,2]) * B[2,k1]
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # P6: (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P7: (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check that the decomposition is exact."""
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    
    # Build the target tensor T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) for same (i,j,k), 0 otherwise
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Check all (a,b,c) triples
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j2 = b // p
            k = b % p
            for c in range(p * n):
                i2 = c % n
                k2 = c // n
                # Target: 1 if (i,j,k) == (i2,j2,k2), i.e., i==i2, j==j2, k==k2
                expected = 1 if (i == i2 and j == j2 and k == k2) else 0
                
                val = 0
                for r in range(R):
                    val += U[a][r] * V[b][r] * W[c][r]
                
                if val != expected:
                    return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
