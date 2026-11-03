"""Matrix multiplication 3x3x8 rank-minimal decomposition."""
import time
import itertools
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    
    Strategy: Use block decomposition with Strassen for 2x2 sub-blocks.
    Split A (3x3) into [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    Split B (3x8) into [[B1], [B2]] where B1 is 2x8, B2 is 1x8.
    
    C = AB = [A11*B1 + A12*B2, A21*B1 + A22*B2]
    
    For A11*B1 (2x2 * 2x8): Split B1 into four 2x2 blocks, each using Strassen (rank 7). Total: 4*7 = 28.
    For A12*B2 (2x1 * 1x8): 16 scalar products. Rank 16.
    For A21*B1 (1x2 * 2x8): 16 scalar products. Rank 16.
    For A22*B2 (1x1 * 1x8): 8 scalar products. Rank 8.
    
    Total rank: 28 + 16 + 16 + 8 = 68.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
        return U, V, W
    
    # Check if we're running out of time
    if time.time() > deadline - 5:
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Construct the rank-68 decomposition using block Strassen
    
    # Indexing:
    # A[i,j] for i,j in 0,1,2 -> U index: i*m + j = i*3 + j
    # B[j,k] for j in 0,1,2, k in 0..7 -> V index: j*p + k = j*8 + k
    # C[i,k] for i in 0,1,2, k in 0..7 -> W index: k*n + i = k*3 + i
    
    # Block structure:
    # A11 = A[0:2, 0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # A12 = A[0:2, 2] = [A[0,2], A[1,2]]
    # A21 = A[2, 0:2] = [A[2,0], A[2,1]]
    # A22 = A[2, 2]
    
    # B1 = B[0:2, :] = rows 0,1 of B (2x8)
    # B2 = B[2, :] = row 2 of B (1x8)
    
    # C1 = C[0:2, :] = rows 0,1 of C (2x8) = A11*B1 + A12*B2
    # C2 = C[2, :] = row 2 of C (1x8) = A21*B1 + A22*B2
    
    # For A11*B1: Use Strassen on each 2x2 block of B1
    # B1 = [B1_0, B1_1, B1_2, B1_3] where B1_i is 2x2 for columns 2i, 2i+1
    
    U = np.zeros((n * m, 68))
    V = np.zeros((m * p, 68))
    W = np.zeros((p * n, 68))
    
    col = 0
    
    # Strassen for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-b)(e+f)
    # M7 = (d-a)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each 2x2 block of B1 (columns 2i, 2i+1 for i in 0,1,2,3):
    for block_idx in range(4):
        k0 = 2 * block_idx
        k1 = 2 * block_idx + 1
        
        # A11 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # B_block = [[B[0,k0], B[0,k1]], [B[1,k0], B[1,k1]]]
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k0])
        # This contributes to C[0,k0] and C[1,k1]
        U[0*3+0, col] = 1  # A[0,0]
        U[1*3+1, col] = 1  # A[1,1]
        V[0*8+k0, col] = 1  # B[0,k0]
        V[1*8+k0, col] = 1  # B[1,k0]
        W[k0*3+0, col] = 1  # C[0,k0]
        W[k1*3+1, col] = 1  # C[1,k1]
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        # This contributes to C[1,k0]
        U[1*3+0, col] = 1  # A[1,0]
        U[1*3+1, col] = 1  # A[1,1]
        V[0*8+k0, col] = 1  # B[0,k0]
        W[k0*3+1, col] = 1  # C[1,k0]
        col += 1
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        # This contributes to C[0,k1]
        U[0*3+0, col] = 1  # A[0,0]
        V[0*8+k1, col] = 1  # B[0,k1]
        V[1*8+k1, col] = -1  # B[1,k1]
        W[k1*3+0, col] = 1  # C[0,k1]
        col += 1
        
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        # This contributes to C[0,k0] and C[1,k0]
        U[1*3+1, col] = 1  # A[1,1]
        V[1*8+k0, col] = 1  # B[1,k0]
        V[0*8+k0, col] = -1  # B[0,k0]
        W[k0*3+0, col] = 1  # C[0,k0]
        W[k0*3+1, col] = 1  # C[1,k0]
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        # This contributes to C[0,k1] and C[1,k1]
        U[0*3+0, col] = 1  # A[0,0]
        U[0*3+1, col] = 1  # A[0,1]
        V[1*8+k1, col] = 1  # B[1,k1]
        W[k1*3+0, col] = 1  # C[0,k1]
        W[k1*3+1, col] = 1  # C[1,k1]
        col += 1
        
        # M6 = (A[1,0]-A[0,1]) * (B[0,k0]+B[0,k1])
        # This contributes to C[1,k1]
        U[1*3+0, col] = 1  # A[1,0]
        U[0*3+1, col] = -1  # A[0,1]
        V[0*8+k0, col] = 1  # B[0,k0]
        V[0*8+k1, col] = 1  # B[0,k1]
        W[k1*3+1, col] = 1  # C[1,k1]
        col += 1
        
        # M7 = (A[1,1]-A[0,0]) * (B[1,k0]+B[1,k1])
        # This contributes to C[0,k0] and C[1,k1]
        U[1*3+1, col] = 1  # A[1,1]
        U[0*3+0, col] = -1  # A[0,0]
        V[1*8+k0, col] = 1  # B[1,k0]
        V[1*8+k1, col] = 1  # B[1,k1]
        W[k0*3+0, col] = 1  # C[0,k0]
        W[k1*3+1, col] = 1  # C[1,k1]
        col += 1
    
    # Now handle A12*B2: A[0:2, 2] * B[2, :]
    # C[0,k] += A[0,2]*B[2,k]
    # C[1,k] += A[1,2]*B[2,k]
    for k in range(8):
        U[0*3+2, col] = 1  # A[0,2]
        V[2*8+k, col] = 1  # B[2,k]
        W[k*3+0, col] = 1  # C[0,k]
        col += 1
        
        U[1*3+2, col] = 1  # A[1,2]
        V[2*8+k, col] = 1  # B[2,k]
        W[k*3+1, col] = 1  # C[1,k]
        col += 1
    
    # Handle A21*B1: A[2, 0:2] * B[0:2, :]
    # C[2,k] += A[2,0]*B[0,k] + A[2,1]*B[1,k]
    for k in range(8):
        U[2*3+0, col] = 1  # A[2,0]
        V[0*8+k, col] = 1  # B[0,k]
        W[k*3+2, col] = 1  # C[2,k]
        col += 1
        
        U[2*3+1, col] = 1  # A[2,1]
        V[1*8+k, col] = 1  # B[1,k]
        W[k*3+2, col] = 1  # C[2,k]
        col += 1
    
    # Handle A22*B2: A[2,2] * B[2, :]
    # C[2,k] += A[2,2]*B[2,k]
    for k in range(8):
        U[2*3+2, col] = 1  # A[2,2]
        V[2*8+k, col] = 1  # B[2,k]
        W[k*3+2, col] = 1  # C[2,k]
        col += 1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition correctly computes matrix multiplication."""
        rank = U.shape[1]
        # Build the tensor T from the decomposition
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Check that T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # and T[a,b,c] = 0 otherwise
        
        # For efficiency, just check the nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1) > 1e-10:
                        return False
        
        # Check some zero entries (sample)
        import random
        random.seed(42)
        for _ in range(100):
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c = random.randint(0, p*n-1)
            # Check if this is a valid nonzero entry
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if i == i2 and j == j2 and j == k2:
                continue  # Skip valid entries
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val) > 1e-10:
                return False
        
        return True
    
    # Verify our decomposition
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U, V, W = schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
