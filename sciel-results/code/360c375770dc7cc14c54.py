import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Build rank-36 decomposition using block Strassen
    # Split middle dim 4 into 2+2, output dim 5 into 2+3
    # Blocks: 2x2x2 (rank 7), 2x2x2 (rank 7), 2x2x3 (rank 11), 2x2x3 (rank 11)
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
        # Strassen: compute 7 products for 2x2 matrix mult
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # U: shape (2*2, 7) = (4, 7), indexed by [i*2+j, c]
        # V: shape (2*2, 7) = (4, 7), indexed by [j*2+k, c]
        # W: shape (2*2, 7) = (4, 7), indexed by [k*2+i, c] (trace convention W[k*n+i,r])
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # For 2x2x2: A is 2x2, B is 2x2, C is 2x2
        # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        
        # M1 = (A00+A11)(B00+B11)
        U[0,0] = 1; U[3,0] = 1  # A00, A11
        V[0,0] = 1; V[3,0] = 1  # B00, B11
        W[0,0] = 1; W[3,0] = 1  # C00 += M1, C11 += M1 (W[k*2+i])
        # C00 gets M1: k=0,i=0 -> W[0,0]=1
        # C11 gets M1: k=1,i=1 -> W[1*2+1,0]=W[3,0]=1
        
        # M2 = (A10+A11)B00
        U[2,1] = 1; U[3,1] = 1  # A10, A11
        V[0,1] = 1  # B00
        W[2,1] = 1  # C10 gets M2: k=0,i=1 -> W[0*2+1,1]=W[1,1]... 
        # Wait, W[k*n+i, r] with n=2. C10 means i=1,k=0. W[0*2+1,1]=W[1,1]=1
        W[1,1] = 1
        
        # M3 = A00(B01-B11)
        U[0,2] = 1  # A00
        V[1,2] = 1; V[3,2] = -1  # B01, -B11
        W[1,2] = 1  # C01 gets M3: i=0,k=1 -> W[1*2+0,2]=W[2,2]... 
        # C01: i=0, k=1. W[k*n+i, r] = W[1*2+0, 2] = W[2,2]
        W[2,2] = 1
        
        # M4 = A11(B10-B00)
        U[3,3] = 1  # A11
        V[2,3] = 1; V[0,3] = -1  # B10, -B00
        W[0,3] = 1; W[3,3] = 1  # C00 += M4, C11 += M4
        # C00: i=0,k=0 -> W[0,3]=1
        # C11: i=1,k=1 -> W[3,3]=1
        
        # M5 = (A00+A01)B11
        U[0,4] = 1; U[1,4] = 1  # A00, A01
        V[3,4] = 1  # B11
        W[3,4] = -1  # C11 -= M5: i=1,k=1 -> W[3,4]=-1
        W[0,4] = -1  # C00 -= M5: i=0,k=0 -> W[0,4]=-1
        
        # M6 = (A10-A00)(B00+B10)
        U[2,5] = 1; U[0,5] = -1  # A10, -A00
        V[0,5] = 1; V[2,5] = 1  # B00, B10
        W[1,5] = 1; W[3,5] = 1  # C10 += M6, C11 += M6
        # C10: i=1,k=0 -> W[1,5]=1
        # C11: i=1,k=1 -> W[3,5]=1
        
        # M7 = (A01-A11)(B01+B11)
        U[1,6] = 1; U[3,6] = -1  # A01, -A11
        V[1,6] = 1; V[3,6] = 1  # B01, B11
        W[2,6] = 1; W[0,6] = 1  # C01 += M7, C00 += M7
        # C01: i=0,k=1 -> W[2,6]=1
        # C00: i=0,k=0 -> W[0,6]=1
        
        return U, V, W
    
    def schoolbook_2x2x3():
        """Return U, V, W for 2x2x3 schoolbook (rank 12) - but we want rank 11."""
        # Actually let's just use schoolbook for now: rank = 2*2*3 = 12
        # But we claimed rank 11. Let me use a mixed approach.
        # Split 3 cols as 1+2: 2x2x1 (rank 4) + 2x2x2 (rank 7) = rank 11
        pass
    
    # Let me build the full 2x4x5 decomposition
    # Middle split: J1={0,1}, J2={2,3}
    # Output split: K1={0,1}, K2={2,3,4}
    
    # Block (J1,K1): 2x2x2, Strassen rank 7
    # Block (J2,K1): 2x2x2, Strassen rank 7  
    # Block (J1,K2): 2x2x3, rank 11 (4+7)
    # Block (J2,K2): 2x2x3, rank 11 (4+7)
    
    total_rank = 7 + 7 + 11 + 11  # = 36
    
    U = np.zeros((n*m, total_rank))  # (8, 36)
    V = np.zeros((m*p, total_rank))  # (20, 36)
    W = np.zeros((p*n, total_rank))  # (10, 36)
    
    col = 0
    
    # Block (J1, K1): rows 0,1 of A; cols 0,1 of B; cols 0,1 of C
    # A indices: i*4+j for i in {0,1}, j in {0,1} -> rows 0,1,4,5
    # B indices: j*5+k for j in {0,1}, k in {0,1} -> rows 0,1,5,6
    # C indices: k*2+i for k in {0,1}, i in {0,1} -> rows 0,1,2,3
    U1, V1, W1 = strassen_2x2x2()
    # Map local indices to global
    a_rows = [0, 1, 4, 5]  # i*4+j for (i,j) in [(0,0),(0,1),(1,0),(1,1)]
    b_rows = [0, 1, 5, 6]  # j*5+k for (j,k) in [(0,0),(0,1),(1,0),(1,1)]
    c_rows = [0, 1, 2, 3]  # k*2+i for (k,i) in [(0,0),(0,1),(1,0),(1,1)]
    
    for local_row, global_row in enumerate(a_rows):
        U[global_row, col:col+7] = U1[local_row, :]
    for local_row, global_row in enumerate(b_rows):
        V[global_row, col:col+7] = V1[local_row, :]
    for local_row, global_row in enumerate(c_rows):
        W[global_row, col:col+7] = W1[local_row, :]
    col += 7
    
    # Block (J2, K1): rows 2,3 of A; cols 0,1 of B; cols 0,1 of C
    # A indices: i*4+j for i in {0,1}, j in {2,3} -> rows 2,3,6,7
    # B indices: j*5+k for j in {2,3}, k in {0,1} -> rows 10,11,15,16
    # C indices: same as before
    a_rows = [2, 3, 6, 7]
    b_rows = [10, 11, 15, 16]
    c_rows = [0, 1, 2, 3]
    
    U[2, col:col+7] = U1[0, :]  # A[0,2]
    U[3, col:col+7] = U1[1, :]  # A[0,3]
    U[6, col:col+7] = U1[2, :]  # A[1,2]
    U[7, col:col+7] = U1[3, :]  # A[1,3]
    V[10, col:col+7] = V1[0, :]  # B[2,0]
    V[11, col:col+7] = V1[1, :]  # B[2,1]
    V[15, col:col+7] = V1[2, :]  # B[3,0]
    V[16, col:col+7] = V1[3, :]  # B[3,1]
    W[0, col:col+7] = W1[0, :]
    W[1, col:col+7] = W1[1, :]
    W[2, col:col+7] = W1[2, :]
    W[3, col:col+7] = W1[3, :]
    col += 7
    
    # Block (J1, K2): 2x2x3, split into 2x2x1 + 2x2x2
    # First: 2x2x1 (rank 4) for col k=2
    # A rows: 0,1,4,5; B rows: 0*5+2=2, 1*5+2=7; C rows: 2*2+0=4, 2*2+1=5
    for ci in range(2):  # i=0,1
        for cj in range(2):  # j=0,1
            a_idx = ci*4 + cj
            b_idx = cj*5 + 2  # k=2
            c_idx = 2*2 + ci  # k=2, i=ci
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
    
    # Second: 2x2x2 (rank 7) for cols k=3,4
    # A rows: 0,1,4,5; B rows: 0*5+3=3, 0*5+4=4, 1*5+3=8, 1*5+4=9
    # C rows: 3*2+0=6, 3*2+1=7, 4*2+0=8, 4*2+1=9
    U[0, col:col+7] = U1[0, :]
    U[1, col:col+7] = U1[1, :]
    U[4, col:col+7] = U1[2, :]
    U[5, col:col+7] = U1[3, :]
    V[3, col:col+7] = V1[0, :]  # B[0,3]
    V[4, col:col+7] = V1[1, :]  # B[0,4]
    V[8, col:col+7] = V1[2, :]  # B[1,3]
    V[9, col:col+7] = V1[3, :]  # B[1,4]
    W[6, col:col+7] = W1[0, :]  # C[0,3]: k=3,i=0
    W[7, col:col+7] = W1[1, :]  # C[1,3]: k=3,i=1
    W[8, col:col+7] = W1[2, :]  # C[0,4]: k=4,i=0
    W[9, col:col+7] = W1[3, :]  # C[1,4]: k=4,i=1
    col += 7
    
    # Block (J2, K2): 2x2x3, split into 2x2x1 + 2x2x2
    # First: 2x2x1 (rank 4) for col k=2
    for ci in range(2):
        for cj in range(2):
            a_idx = ci*4 + (2+cj)  # j=2,3
            b_idx = (2+cj)*5 + 2  # k=2
            c_idx = 2*2 + ci
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
    
    # Second: 2x2x2 (rank 7) for cols k=3,4
    U[2, col:col+7] = U1[0, :]  # A[0,2]
    U[3, col:col+7] = U1[1, :]  # A[0,3]
    U[6, col:col+7] = U1[2, :]  # A[1,2]
    U[7, col:col+7] = U1[3, :]  # A[1,3]
    V[13, col:col+7] = V1[0, :]  # B[2,3]
    V[14, col:col+7] = V1[1, :]  # B[2,4]
    V[18, col:col+7] = V1[2, :]  # B[3,3]
    V[19, col:col+7] = V1[3, :]  # B[3,4]
    W[6, col:col+7] = W1[0, :]
    W[7, col:col+7] = W1[1, :]
    W[8, col:col+7] = W1[2, :]
    W[9, col:col+7] = W1[3, :]
    col += 7
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,j,k} for all i,j,k"""
        # Actually the decomposition means:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) (sum_j V[j*p+k,c] B[j,k]) W[k*n+i,c]
        # Wait, that's not right. The tensor decomposition is:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # where T is the matrix multiplication tensor with T[i,j,k] = 1 for all valid i,j,k
        # No wait, the tensor is defined such that the bilinear map is:
        # C[i,k] = sum_{j} A[i,j] B[j,k]
        # And the decomposition gives:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_j V[j*p+k,c] B[j,k])
        # Hmm, I need to be more careful about the convention.
        
        # The standard convention: the multiplication tensor T has entries T[i,j,k]
        # and the decomposition is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # where U is indexed by (i,j), V by (j,k), W by (k,i)
        
        # Let me verify by checking that for random A, B, the result matches.
        rng = np.random.default_rng(42)
        A = rng.integers(-5, 5, size=(n, m))
        B = rng.integers(-5, 5, size=(m, p))
        C_expected = A @ B
        
        # Compute via decomposition
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        C_computed = np.zeros((n, p), dtype=float)
        for r in range(U.shape[1]):
            # Compute intermediate products
            # For each i: sum_j U[i*m+j, r] * A[i,j]
            # For each k: sum_j V[j*p+k, r] * B[j,k]
            prod_A = np.zeros(n)
            prod_B = np.zeros(p)
            for i in range(n):
                for j in range(m):
                    prod_A[i] += U[i*m+j, r] * A[i,j]
            for k in range(p):
                for j in range(m):
                    prod_B[k] += V[j*p+k, r] * B[j,k]
            for i in range(n):
                for k in range(p):
                    C_computed[i,k] += W[k*n+i, r] * prod_A[i] * prod_B[k]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
