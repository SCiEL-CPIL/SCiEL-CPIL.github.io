"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    deadline = time.time() + 145
    
    # Target tensor
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook rank 30
    rank0 = 30
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((n*p, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = rank0
    
    # Explicit rank-29 construction using Strassen on 2x2x2 sub-block
    # Sub-block: i in {0,1}, j in {0,1}, k in {0,1}
    # Strassen terms for X(2x2) * Y(2x2):
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10-y11)
    # M4 = x11(y01-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y10)
    # M7 = (x01-x11)(y01+y11)
    
    # In our tensor: U[i*m+j], V[j*p+k], W[k*n+i]
    # For sub-block i in {0,1}, j in {0,1}, k in {0,1}:
    # U indices: 0,1,3,4 (i*m+j for i in {0,1}, j in {0,1})
    # V indices: 0,1,5,6 (j*p+k for j in {0,1}, k in {0,1})
    # W indices: 0,1,2,3 (k*n+i for k in {0,1}, i in {0,1})
    
    # Strassen M1: (a00+a11)(b00+b11) where a_ij = A[i,j], b_jk = B[j,k]
    # U: a00 at U[0], a11 at U[4]
    # V: b00 at V[0], b11 at V[6]
    # W: c00 at W[0], c11 at W[3]
    
    # Let me build the rank-29 solution
    U29 = np.zeros((n*m, 29))
    V29 = np.zeros((m*p, 29))
    W29 = np.zeros((n*p, 29))
    
    # Strassen terms (7 terms) for sub-block i in {0,1}, j in {0,1}, k in {0,1}
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    r = 0
    U29[0, r] = 1; U29[4, r] = 1  # A[0,0], A[1,1]
    V29[0, r] = 1; V29[6, r] = 1  # B[0,0], B[1,1]
    W29[0, r] = 1; W29[3, r] = 1  # C[0,0], C[1,1]
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    r = 1
    U29[3, r] = 1; U29[4, r] = 1  # A[1,0], A[1,1]
    V29[0, r] = 1  # B[0,0]
    W29[2, r] = 1  # C[1,0]
    
    # M3 = A[0,0] * (B[1,0]-B[1,1])
    r = 2
    U29[0, r] = 1  # A[0,0]
    V29[5, r] = 1; V29[6, r] = -1  # B[1,0], B[1,1]
    W29[1, r] = 1  # C[0,1]
    
    # M4 = A[1,1] * (B[0,1]-B[0,0])
    r = 3
    U29[4, r] = 1  # A[1,1]
    V29[1, r] = 1; V29[0, r] = -1  # B[0,1], B[0,0]
    W29[2, r] = 1  # C[1,0]
    
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    r = 4
    U29[0, r] = 1; U29[1, r] = 1  # A[0,0], A[0,1]
    V29[6, r] = 1  # B[1,1]
    W29[1, r] = 1  # C[0,1]
    
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[1,0])
    r = 5
    U29[3, r] = 1; U29[0, r] = -1  # A[1,0], A[0,0]
    V29[0, r] = 1; V29[5, r] = 1  # B[0,0], B[1,0]
    W29[2, r] = 1  # C[1,0]
    
    # M7 = (A[0,1]-A[1,1]) * (B[0,1]+B[1,1])
    r = 6
    U29[1, r] = 1; U29[4, r] = -1  # A[0,1], A[1,1]
    V29[1, r] = 1; V29[6, r] = 1  # B[0,1], B[1,1]
    W29[3, r] = 1  # C[1,1]
    
    # Now add the remaining 22 schoolbook terms (30 - 8 = 22, plus 7 Strassen = 29)
    # The 8 removed terms were for i in {0,1}, j in {0,1}, k in {0,1}
    # Remaining: all (i,j,k) except those 8
    
    r = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in [0,1] and j in [0,1] and k in [0,1]:
                    continue  # Already handled by Strassen
                U29[i*m+j, r] = 1
                V29[j*p+k, r] = 1
                W29[k*n+i, r] = 1
                r += 1
    
    # Verify
    recon = np.zeros((n*m, m*p, n*p))
    for rr in range(29):
        recon += np.einsum('i,j,k->ijk', U29[:,rr], V29[:,rr], W29[:,rr])
    
    if np.allclose(recon, T, atol=1e-9):
        return {"U": U29.tolist(), "V": V29.tolist(), "W": W29.tolist()}
    
    # Fallback to schoolbook
    return best_result
# EVOLVE-BLOCK-END
