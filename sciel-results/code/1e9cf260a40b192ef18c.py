import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _construct_rank_40():
    """
    Rank-40 decomposition for 2x3x10 matrix multiplication.
    Each (i,j,k) triple gets its own rank-1 term: A[i,j]*B[j,k] -> C[i,k].
    Total: 2*3*10 = 60 terms, but we group by column pairs using 8 per pair.
    Actually this is just the naive 60-term decomposition compressed... 
    Wait, 5 pairs * 8 = 40, but we have 60 (i,j,k) triples.
    This construction is WRONG - it only covers 40 of the 60 required entries.
    """
    n, m, p = DIMENSIONS
    rank = 60
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _construct_rank_36():
    """
    Attempt rank-36: use Strassen-like 7-rank for 2x2x2 blocks.
    For each pair of columns (k0,k1), decompose the 2x3x2 sub-problem.
    Split j into {0,1} and {2}. The {2} part is rank 4 (2 rows x 2 cols).
    The {0,1} part is 2x2x2, which has Strassen rank 7.
    Total per pair: 7 + 4 = 11. For 5 pairs: 55. Not better.
    
    Alternative: try to find a rank-7 decomposition for 2x3x2 directly.
    """
    n, m, p = DIMENSIONS
    # 2x3x2 tensor: entries (i,j,k) for i in {0,1}, j in {0,1,2}, k in {0,1}
    # That's 12 entries. Naive rank 12. Can we do better?
    # Try: use 7 ranks
    rank = 35
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # For the 2x2x2 part (j=0,1), use Strassen with 7 ranks
        # Strassen for 2x2x2:
        # m1 = (A00+A11)(B00+B11)
        # m2 = (A10+A11)(B00)
        # m3 = (A00)(B01-B11)
        # m4 = (A11)(B10-B00)
        # m5 = (A00+A01)(B11)
        # m6 = (A10-A00)(B00+B01)
        # m7 = (A01-A11)(B10+B11)
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # But our A is 2x3 and B is 3x2, not 2x2 x 2x2.
        # We need A[i,0], A[i,1] for i in {0,1} and B[0,k], B[1,k] for k in {k0,k1}
        # This IS a 2x2x2 problem if we ignore j=2!
        
        # Strassen terms for j=0,1 part:
        # m1: (A00+A11)*(B0k0+B1k0) -> contributes to C0k0 and C1k0
        # But wait, Strassen computes C = A*B where A,B are 2x2.
        # Here A = [[A00, A01], [A10, A11]] and B = [[B0k0, B0k1], [B1k0, B1k1]]
        # C = A*B gives C[0,k0], C[0,k1], C[1,k0], C[1,k1]
        
        # Strassen rank-7 for 2x2x2:
        # s1 = (A00+A11), t1 = (B00+B11), m1 = s1*t1
        # s2 = (A10+A11), t2 = B00, m2 = s2*t2
        # s3 = A00, t3 = (B01-B11), m3 = s3*t3
        # s4 = A11, t4 = (B10-B00), m4 = s4*t4
        # s5 = (A00+A01), t5 = B11, m5 = s5*t5
        # s6 = (A10-A00), t6 = (B00+B01), m6 = s6*t6
        # s7 = (A01-A11), t7 = (B10+B11), m7 = s7*t7
        
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # In our notation: A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
        # B00=B[0,k0], B01=B[0,k1], B10=B[1,k0], B11=B[1,k1]
        # C00=C[0,k0], C01=C[0,k1], C10=C[1,k0], C11=C[1,k1]
        
        # m1: (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k0])
        U[0*3+0, col] = 1; U[1*3+1, col] = 1
        V[0*10+k0, col] = 1; V[1*10+k0, col] = 1
        W[k0*2+0, col] = 1; W[k0*2+1, col] = 1
        col += 1
        
        # m2: (A[1,0]+A[1,1]) * B[0,k0]
        U[1*3+0, col] = 1; U[1*3+1, col] = 1
        V[0*10+k0, col] = 1
        W[k0*2+1, col] = 1
        col += 1
        
        # m3: A[0,0] * (B[0,k1]-B[1,k1])
        U[0*3+0, col] = 1
        V[0*10+k1, col] = 1; V[1*10+k1, col] = -1
        W[k1*2+0, col] = 1
        col += 1
        
        # m4: A[1,1] * (B[1,k0]-B[0,k0])
        U[1*3+1, col] = 1
        V[1*10+k0, col] = 1; V[0*10+k0, col] = -1
        W[k0*2+1, col] = 1
        col += 1
        
        # m5: (A[0,0]+A[0,1]) * B[1,k0]
        U[0*3+0, col] = 1; U[0*3+1, col] = 1
        V[1*10+k0, col] = 1
        W[k0*2+0, col] = -1; W[k0*2+1, col] = 1
        col += 1
        
        # m6: (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[1*3+0, col] = 1; U[0*3+0, col] = -1
        V[0*10+k0, col] = 1; V[0*10+k1, col] = 1
        W[k1*2+1, col] = 1
        col += 1
        
        # m7: (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[0*3+1, col] = 1; U[1*3+1, col] = -1
        V[1*10+k0, col] = 1; V[1*10+k1, col] = 1
        W[k0*2+0, col] = 1
        col += 1
        
        # Now handle j=2 part: C[i,k] += A[i,2]*B[2,k]
        # This needs 4 more ranks (2 rows x 2 cols)
        # A[0,2]*B[2,k0] -> C[0,k0]
        U[0*3+2, col] = 1
        V[2*10+k0, col] = 1
        W[k0*2+0, col] = 1
        col += 1
        
        # A[0,2]*B[2,k1] -> C[0,k1]
        U[0*3+2, col] = 1
        V[2*10+k1, col] = 1
        W[k1*2+0, col] = 1
        col += 1
        
        # A[1,2]*B[2,k0] -> C[1,k0]
        U[1*3+2, col] = 1
        V[2*10+k0, col] = 1
        W[k0*2+1, col] = 1
        col += 1
        
        # A[1,2]*B[2,k1] -> C[1,k1]
        U[1*3+2, col] = 1
        V[2*10+k1, col] = 1
        W[k1*2+1, col] = 1
        col += 1
    
    return U, V, W

def _construct_rank_35():
    """
    Rank-35 decomposition attempt using Strassen for 2x2x2 + naive for j=2.
    Per pair: 7 (Strassen) + 4 (j=2) = 11 ranks. Total: 5*11 = 55.
    This actually gives rank 55, not 35. Renamed for clarity.
    """
    n, m, p = DIMENSIONS
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen rank-7 for the 2x2x2 sub-problem (j=0,1)
        # m1: (A[0,0]+A[1,1])*(B[0,k0]+B[1,k0])
        U[0, col] = 1; U[4, col] = 1
        V[k0, col] = 1; V[10+k0, col] = 1
        W[k0*2, col] = 1; W[k0*2+1, col] = 1
        col += 1
        # m2: (A[1,0]+A[1,1])*B[0,k0]
        U[3, col] = 1; U[4, col] = 1
        V[k0, col] = 1
        W[k0*2+1, col] = 1
        col += 1
        # m3: A[0,0]*(B[0,k1]-B[1,k1])
        U[0, col] = 1
        V[k1, col] = 1; V[10+k1, col] = -1
        W[k1*2, col] = 1
        col += 1
        # m4: A[1,1]*(B[1,k0]-B[0,k0])
        U[4, col] = 1
        V[10+k0, col] = 1; V[k0, col] = -1
        W[k0*2+1, col] = 1
        col += 1
        # m5: (A[0,0]+A[0,1])*B[1,k0]
        U[0, col] = 1; U[1, col] = 1
        V[10+k0, col] = 1
        W[k0*2, col] = -1; W[k0*2+1, col] = 1
        col += 1
        # m6: (A[1,0]-A[0,0])*(B[0,k0]+B[0,k1])
        U[3, col] = 1; U[0, col] = -1
        V[k0, col] = 1; V[k1, col] = 1
        W[k1*2+1, col] = 1
        col += 1
        # m7: (A[0,1]-A[1,1])*(B[1,k0]+B[1,k1])
        U[1, col] = 1; U[4, col] = -1
        V[10+k0, col] = 1; V[10+k1, col] = 1
        W[k0*2, col] = 1
        col += 1
        
        # j=2 part: 4 naive terms
        U[2, col] = 1; V[20+k0, col] = 1; W[k0*2, col] = 1; col += 1
        U[2, col] = 1; V[20+k1, col] = 1; W[k1*2, col] = 1; col += 1
        U[5, col] = 1; V[20+k0, col] = 1; W[k0*2+1, col] = 1; col += 1
        U[5, col] = 1; V[20+k1, col] = 1; W[k1*2+1, col] = 1; col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=3, p=10, tol=1e-9):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i*m+j, j*p+k, k*n+i] - 1.0) > tol:
                    return False
    # Also check no spurious entries
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i = a // m; j = a % m
                jj = b // p; kk = b % p
                ii = c % n; kk2 = c // n
                if j != jj or kk != kk2:
                    if abs(T[a,b,c]) > tol:
                        return False
    return True

def _construct_naive_60():
    """Naive rank-60: one term per (i,j,k) triple."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _construct_rank_40():
    """
    Rank-40: For each pair of columns, use 8 rank-1 terms.
    This is the naive per-term decomposition grouped by column pairs.
    Total: 5 pairs * 8 = 40. But we have 60 (i,j,k) triples!
    This is WRONG - we need 60 terms minimum with this approach.
    Actually, each term covers exactly one (i,j,k), so we need 60 terms.
    This function is intentionally left as a placeholder that returns rank 60.
    """
    return _construct_naive_60()

def _construct_strassen_55():
    """
    Rank-55: Strassen (rank 7) for 2x2x2 sub-problem + naive (rank 4) for j=2.
    Per column pair: 7 + 4 = 11 ranks. Total: 5 * 11 = 55.
    """
    n, m, p = DIMENSIONS
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        # Indices: A[i,j] at index i*m+j, B[j,k] at j*p+k, C[i,k] at k*n+i
        
        # Strassen rank-7 for A[:2,:2] * B[:2,:] (2x2 times 2x2)
        # m1 = (A00+A11)*(B00+B10) where B00=B[0,k0], B10=B[1,k0]
        U[0, col] = 1; U[4, col] = 1
        V[k0, col] = 1; V[10+k0, col] = 1
        W[2*k0, col] = 1; W[2*k0+1, col] = 1
        col += 1
        # m2 = (A10+A11)*B[0,k0]
        U[3, col] = 1; U[4, col] = 1
        V[k0, col] = 1
        W[2*k0+1, col] = 1
        col += 1
        # m3 = A[0,0]*(B[0,k1]-B[1,k1])
        U[0, col] = 1
        V[k1, col] = 1; V[10+k1, col] = -1
        W[2*k1, col] = 1
        col += 1
        # m4 = A[1,1]*(B[1,k0]-B[0,k0])
        U[4, col] = 1
        V[10+k0, col] = 1; V[k0, col] = -1
        W[2*k0+1, col] = 1
        col += 1
        # m5 = (A[0,0]+A[0,1])*B[1,k0]
        U[0, col] = 1; U[1, col] = 1
        V[10+k0, col] = 1
        W[2*k0, col] = -1; W[2*k0+1, col] = 1
        col += 1
        # m6 = (A[1,0]-A[0,0])*(B[0,k0]+B[0,k1])
        U[3, col] = 1; U[0, col] = -1
        V[k0, col] = 1; V[k1, col] = 1
        W[2*k1+1, col] = 1
        col += 1
        # m7 = (A[0,1]-A[1,1])*(B[1,k0]+B[1,k1])
        U[1, col] = 1; U[4, col] = -1
        V[10+k0, col] = 1; V[10+k1, col] = 1
        W[2*k0, col] = 1
        col += 1
        
        # j=2 part: A[i,2]*B[2,k] for i in {0,1}, k in {k0,k1}
        U[2, col] = 1; V[20+k0, col] = 1; W[2*k0, col] = 1; col += 1
        U[2, col] = 1; V[20+k1, col] = 1; W[2*k1, col] = 1; col += 1
        U[5, col] = 1; V[20+k0, col] = 1; W[2*k0+1, col] = 1; col += 1
        U[5, col] = 1; V[20+k1, col] = 1; W[2*k1+1, col] = 1; col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 140.0
    
    # Try rank-55 Strassen construction
    U, V, W = _construct_strassen_55()
    if _verify_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to naive rank-60
    U, V, W = _construct_naive_60()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
