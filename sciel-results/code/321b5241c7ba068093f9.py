import time
import numpy as np

def construct():
    """
    Return a rank-28 exact decomposition for 2x3 * 3x5 matrix multiplication.
    
    The decomposition is based on splitting the middle dimension (j) into j=0 and j in {1,2}.
    - Block 1 (j=0): 2x1x5 schoolbook, rank 10.
    - Block 2 (j in {1,2}): 2x2x5.
      - Split k into {0,1}, {2}, {3,4}.
      - k in {0,1}: Strassen 2x2x2, rank 7.
      - k=2: schoolbook 2x2x1, rank 4.
      - k in {3,4}: Strassen 2x2x2, rank 7.
    Total rank: 28.
    
    Indexing:
    - U: (n*m, rank) = (6, 28), A[i,j] -> U[i*m+j, r]
    - V: (m*p, rank) = (15, 28), B[j,k] -> V[j*p+k, r]
    - W: (p*n, rank) = (10, 28), C[i,k] -> W[k*n+i, r]
    """
    n, m, p = 2, 3, 5
    rank = 28
    
    # Use integer arrays for exactness
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Block 1: j=0, all k in {0,1,2,3,4}
    # T[i, 0, k] = 1
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    # col = 10
    
    # Block 2: j in {1,2}
    # We use Strassen for 2x2x2 blocks and schoolbook for 2x2x1.
    
    def add_strassen_2x2x2(base_j, base_k):
        """Add Strassen decomposition for 2x2x2 block at j=base_j..base_j+1, k=base_k..base_k+1"""
        nonlocal col
        # Indices:
        # A rows: i=0,1; cols: j=base_j, base_j+1
        # B rows: j=base_j, base_j+1; cols: k=base_k, base_k+1
        # C rows: k=base_k, base_k+1; cols: i=0,1
        
        j0 = base_j
        j1 = base_j + 1
        k0 = base_k
        k1 = base_k + 1
        
        # M1: (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
        U[0 * m + j0, col] = 1
        U[1 * m + j1, col] = 1
        V[j0 * p + k0, col] = 1
        V[j1 * p + k1, col] = 1
        W[k0 * n + 0, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1
        
        # M2: (A[1,j0]+A[1,j1]) * B[j0,k0]
        U[1 * m + j0, col] = 1
        U[1 * m + j1, col] = 1
        V[j0 * p + k0, col] = 1
        W[k0 * n + 1, col] = 1
        W[k1 * n + 1, col] = -1
        col += 1
        
        # M3: A[0,j0] * (B[j0,k1]-B[j1,k1])
        U[0 * m + j0, col] = 1
        V[j0 * p + k1, col] = 1
        V[j1 * p + k1, col] = -1
        W[k1 * n + 0, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1
        
        # M4: A[1,j1] * (B[j1,k0]-B[j0,k0])
        U[1 * m + j1, col] = 1
        V[j1 * p + k0, col] = 1
        V[j0 * p + k0, col] = -1
        W[k0 * n + 0, col] = 1
        W[k0 * n + 1, col] = 1
        col += 1
        
        # M5: (A[0,j0]+A[0,j1]) * B[j1,k1]
        U[0 * m + j0, col] = 1
        U[0 * m + j1, col] = 1
        V[j1 * p + k1, col] = 1
        W[k0 * n + 0, col] = -1
        W[k1 * n + 0, col] = 1
        col += 1
        
        # M6: (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
        U[1 * m + j0, col] = 1
        U[0 * m + j0, col] = -1
        V[j0 * p + k0, col] = 1
        V[j0 * p + k1, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1
        
        # M7: (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
        U[0 * m + j1, col] = 1
        U[1 * m + j1, col] = -1
        V[j1 * p + k0, col] = 1
        V[j1 * p + k1, col] = 1
        W[k0 * n + 0, col] = 1
        col += 1

    add_strassen_2x2x2(1, 0)  # j=1,2; k=0,1
    # col = 17
    
    # Block 2b: j in {1,2}, k=2 — schoolbook 2x2x1
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
    # col = 21
    
    add_strassen_2x2x2(1, 3)  # j=1,2; k=3,4
    # col = 28
    
    assert col == rank, f"Expected col={rank}, got {col}"
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if val != 1:
                    raise ValueError(f"Decomposition failed at i={i}, j={j}, k={k}: got {val}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
