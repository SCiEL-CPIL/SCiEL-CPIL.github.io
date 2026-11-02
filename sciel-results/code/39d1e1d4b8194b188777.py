"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # p = 15, partition into 7 pairs of 2 columns + 1 singleton
    # Each 2x2x2 block uses Strassen (rank 7), singleton uses schoolbook (rank 4)
    # Total rank = 7*7 + 4 = 53
    
    # Strassen decomposition for 2x2x2:
    # U factors (A side): [A00+A11, A10+A11, A00, A11, A00+A01, A10-A00, A01-A11]
    # V factors (B side): [B00+B10, B00, B01-B11, B10-B00, B11, B00+B01, B10+B11]
    # W combinations for C00: [1, 0, -1, 0, 1, 0, 0]
    # W combinations for C01: [0, 0, 1, 0, 0, 1, 0]
    # W combinations for C10: [0, 1, 0, -1, 0, 0, 1]
    # W combinations for C11: [1, 0, -1, 0, 1, 0, 1]
    
    # Actually, let me use the standard Strassen with proper indexing.
    # For 2x2 matrices A, B, C = A*B:
    # M1 = (A00+A11)*(B00+B11)  -- wait, B is 2x2 so B00,B01,B10,B11
    # Standard Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B00)
    # P4 = A11(B01-B11)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B10)
    # P7 = (A01-A11)(B01+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 - P4 + P6
    # C11 = P1 - P2 + P3 + P6
    
    # Wait, I need to be careful with B's indexing. B is 2x2, B[j,k].
    # A[i,j], B[j,k], C[i,k].
    # A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
    # B00=B[0,0], B01=B[0,1], B10=B[1,0], B11=B[1,1]
    # C00=C[0,0], C01=C[0,1], C10=C[1,0], C11=C[1,1]
    
    # Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B00)
    # P4 = A11(B01-B11)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B10)
    # P7 = (A01-A11)(B01+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 - P4 + P6
    # C11 = P1 - P2 + P3 + P6
    
    # U factors (A-side), indexed by [i*m+j, col] where m=2:
    # For A: row index i*m+j = i*2+j
    # A00: i=0,j=0 -> 0
    # A01: i=0,j=1 -> 1
    # A10: i=1,j=0 -> 2
    # A11: i=1,j=1 -> 3
    
    # V factors (B-side), indexed by [j*p+k, col]:
    # For a 2x2 block at columns k0, k0+1:
    # B00: j=0, k=k0 -> 0*2+0 = 0 (within block) -> global: 0*15+0? No, V is m*p = 2*15=30 rows.
    # V[j*p + k] where j in [0,1], k in [0,14]
    # For block starting at col k0:
    # B[0,k0]: j=0, k=k0 -> 0*15+k0
    # B[0,k0+1]: j=0, k=k0+1 -> 0*15+k0+1
    # B[1,k0]: j=1, k=k0 -> 1*15+k0
    # B[1,k0+1]: j=1, k=k0+1 -> 1*15+k0+1
    
    # W factors (C-side), indexed by [k*n+i, col] where n=2:
    # C[i,k]: k in [0,14], i in [0,1]
    # W[k*n+i] = W[2*k+i]
    # C[0,k0]: 2*k0+0
    # C[0,k0+1]: 2*k0+1
    # C[1,k0]: 2*k0+2
    # C[1,k0+1]: 2*k0+3
    
    # Let's build the decomposition
    
    rank = 53
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # 7 pairs: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)
    for t in range(7):
        k0 = 2 * t
        # Strassen for block at columns k0, k0+1
        # P1 = (A00+A11)(B[0,k0]+B[1,k0+1])  -- wait, B00+B11 means B[0,k0]+B[1,k0+1]
        # Actually B00 = B[0,k0], B11 = B[1,k0+1]
        
        # P1: A: A00+A11 -> U[0]+U[3], V: B[0,k0]+B[1,k0+1] -> V[k0]+V[15+k0+1]
        # W: C00 += 1, C01 += 0, C10 += 0, C11 += 1
        U[0, col] += 1
        U[3, col] += 1
        V[k0, col] += 1
        V[15 + k0 + 1, col] += 1
        W[2*k0+0, col] += 1  # C[0,k0]
        W[2*k0+3, col] += 1  # C[1,k0+1]
        col += 1
        
        # P2 = (A10+A11)B[0,k0]
        U[2, col] += 1
        U[3, col] += 1
        V[k0, col] += 1
        W[2*k0+2, col] += 1  # C[1,k0]
        W[2*k0+3, col] += -1  # C[1,k0+1]
        col += 1
        
        # P3 = A00(B[1,k0]-B[0,k0])
        U[0, col] += 1
        V[15+k0, col] += 1
        V[k0, col] += -1
        W[2*k0+1, col] += 1  # C[0,k0+1]
        W[2*k0+3, col] += 1  # C[1,k0+1]
        col += 1
        
        # P4 = A11(B[0,k0+1]-B[1,k0+1])
        U[3, col] += 1
        V[k0+1, col] += 1
        V[15+k0+1, col] += -1
        W[2*k0+0, col] += 1  # C[0,k0]
        W[2*k0+2, col] += -1  # C[1,k0]
        col += 1
        
        # P5 = (A00+A01)B[1,k0+1]
        U[0, col] += 1
        U[1, col] += 1
        V[15+k0+1, col] += 1
        W[2*k0+0, col] += -1  # C[0,k0]
        W[2*k0+1, col] += 1  # C[0,k0+1]
        col += 1
        
        # P6 = (A10-A00)(B[0,k0]+B[1,k0])
        U[2, col] += 1
        U[0, col] += -1
        V[k0, col] += 1
        V[15+k0, col] += 1
        W[2*k0+2, col] += 1  # C[1,k0]
        W[2*k0+3, col] += 1  # C[1,k0+1]
        col += 1
        
        # P7 = (A01-A11)(B[0,k0+1]+B[1,k0+1])
        U[1, col] += 1
        U[3, col] += -1
        V[k0+1, col] += 1
        V[15+k0+1, col] += 1
        W[2*k0+0, col] += 1  # C[0,k0]
        col += 1
    
    # Singleton column 14: schoolbook 2x2x1 (rank 4)
    # C[0,14] = A[0,0]*B[0,14] + A[0,1]*B[1,14]
    # C[1,14] = A[1,0]*B[0,14] + A[1,1]*B[1,14]
    k0 = 14
    
    # Term 1: A00 * B[0,14]
    U[0, col] = 1
    V[k0, col] = 1
    W[2*k0+0, col] = 1  # C[0,14]
    col += 1
    
    # Term 2: A01 * B[1,14]
    U[1, col] = 1
    V[15+k0, col] = 1
    W[2*k0+0, col] = 1  # C[0,14]
    col += 1
    
    # Term 3: A10 * B[0,14]
    U[2, col] = 1
    V[k0, col] = 1
    W[2*k0+1, col] = 1  # C[1,14]
    col += 1
    
    # Term 4: A11 * B[1,14]
    U[3, col] = 1
    V[15+k0, col] = 1
    W[2*k0+1, col] = 1  # C[1,14]
    col += 1
    
    # Verify with random matrices
    if time.time() < deadline - 5:
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        # Compute using decomposition
        C_comp = np.zeros((n, p))
        for c in range(rank):
            u = U[:, c]
            v = V[:, c]
            w = W[:, c]
            scalar = np.dot(u, A.flatten()) * np.dot(v, B.flatten())
            # Add to C
            for ki in range(p):
                for ii in range(n):
                    idx = ki * n + ii
                    C_comp[ii, ki] += w[idx] * scalar
        
        err = np.max(np.abs(C_comp - C_true))
        if err > 1e-10:
            # Fall back to schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, column] = 1
                        V[j * p + k, column] = 1
                        W[k * n + i, column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
