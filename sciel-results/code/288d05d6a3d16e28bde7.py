"""Matrix multiplication 2,2,15: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 145.0  # Safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the target tensor T[i,j,k] = delta_{i,k} (for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k])
    # Actually, the tensor is M[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k]
    # In the trace convention: W[k*n + i, r] corresponds to output C[i,k]
    # The tensor entry for (i,j,k) is 1 if this term contributes to C[i,k] via A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 for all i,j,k (it's the identity in the j-sum sense)
    # Wait, let me think again.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear form is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # This must equal sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k? No.
    # The tensor T is defined such that T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[?,?]
    # Actually, the standard formulation: the matrix multiplication tensor has entries
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # In the trace convention with W indexed by (k,i), the condition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's verify by checking random instances
    for _ in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for r in range(len(W[0])):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_computed[i, k] = total
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # We use a block decomposition.
    # 7 pairs of columns (0-1, 2-3, ..., 12-13) using Strassen (rank 7 each)
    # 1 single column (14) using trivial (rank 4)
    # Total rank = 7 * 7 + 4 = 53
    
    rank = 53
    U = np.zeros((n * m, rank))  # 4 x 53
    V = np.zeros((m * p, rank))  # 30 x 53
    W = np.zeros((p * n, rank))  # 30 x 53
    
    # A indices in U:
    # 0: A00, 1: A01, 2: A10, 3: A11
    
    # B indices in V:
    # j*p + k. j=0 -> 0..14, j=1 -> 15..29
    
    # C indices in W:
    # k*n + i. k=0 -> 0,1; k=1 -> 2,3; ... k=14 -> 28,29
    
    # Helper to set Strassen block for column pair (k0, k1)
    def add_strassen_block(base_r, k0, k1):
        # e = B0k0, f = B0k1, g = B1k0, h = B1k1
        # P1 = (A00+A11)(B0k0+B1k1) -> (a+d)(e+h)
        # P2 = (A10+A11)B0k0 -> (c+d)e
        # P3 = A00(B0k1-B1k1) -> a(f-h)
        # P4 = A11(B1k0-B0k0) -> d(g-e)
        # P5 = (A00+A01)B1k1 -> (a+b)h
        # P6 = (A10-A00)(B0k0+B0k1) -> (c-a)(e+f)
        # P7 = (A01-A11)(B1k0+B1k1) -> (b-d)(g+h)
        
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        # P1: (A00+A11)(B0k0+B1k1)
        U[0, r1] = 1  # A00
        U[3, r1] = 1  # A11
        V[0 * p + k0, r1] = 1  # B0k0
        V[1 * p + k1, r1] = 1  # B1k1
        
        # P2: (A10+A11)B0k0
        U[2, r2] = 1  # A10
        U[3, r2] = 1  # A11
        V[0 * p + k0, r2] = 1  # B0k0
        
        # P3: A00(B0k1-B1k1)
        U[0, r3] = 1  # A00
        V[0 * p + k1, r3] = 1  # B0k1
        V[1 * p + k1, r3] = -1  # B1k1
        
        # P4: A11(B1k0-B0k0)
        U[3, r4] = 1  # A11
        V[1 * p + k0, r4] = 1  # B1k0
        V[0 * p + k0, r4] = -1  # B0k0
        
        # P5: (A00+A01)B1k1
        U[0, r5] = 1  # A00
        U[1, r5] = 1  # A01
        V[1 * p + k1, r5] = 1  # B1k1
        
        # P6: (A10-A00)(B0k0+B0k1)
        U[2, r6] = 1  # A10
        U[0, r6] = -1  # A00
        V[0 * p + k0, r6] = 1  # B0k0
        V[0 * p + k1, r6] = 1  # B0k1
        
        # P7: (A01-A11)(B1k0+B1k1)
        U[1, r7] = 1  # A01
        U[3, r7] = -1  # A11
        V[1 * p + k0, r7] = 1  # B1k0
        V[1 * p + k1, r7] = 1  # B1k1
        
        # Reconstruction W
        # C0k0 = P1 + P4 - P5 + P7
        # C1k0 = P2 + P4
        # C0k1 = P3 + P5
        # C1k1 = P1 - P2 + P3 + P6
        
        # C0k0 (i=0, k=k0) -> W[2*k0]
        W[2 * k0, r1] = 1
        W[2 * k0, r4] = 1
        W[2 * k0, r5] = -1
        W[2 * k0, r7] = 1
        
        # C1k0 (i=1, k=k0) -> W[2*k0+1]
        W[2 * k0 + 1, r2] = 1
        W[2 * k0 + 1, r4] = 1
        
        # C0k1 (i=0, k=k1) -> W[2*k1]
        W[2 * k1, r3] = 1
        W[2 * k1, r5] = 1
        
        # C1k1 (i=1, k=k1) -> W[2*k1+1]
        W[2 * k1 + 1, r1] = 1
        W[2 * k1 + 1, r2] = -1
        W[2 * k1 + 1, r3] = 1
        W[2 * k1 + 1, r6] = 1
    
    # Add 7 Strassen blocks for pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)
    
    # Add trivial block for column 14
    base_r = 49
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3
    
    # Q1 = A00 B014
    U[0, r1] = 1
    V[0 * p + 14, r1] = 1
    W[2 * 14, r1] = 1  # C014
    
    # Q2 = A01 B114
    U[1, r2] = 1
    V[1 * p + 14, r2] = 1
    W[2 * 14, r2] = 1  # C014
    
    # Q3 = A10 B014
    U[2, r3] = 1
    V[0 * p + 14, r3] = 1
    W[2 * 14 + 1, r3] = 1  # C114
    
    # Q4 = A11 B114
    U[3, r4] = 1
    V[1 * p + 14, r4] = 1
    W[2 * 14 + 1, r4] = 1  # C114
    
    # Verify the decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
