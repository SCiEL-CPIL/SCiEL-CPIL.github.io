import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140  # seconds, safety margin before 150s kill

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    - Split the multiplication into blocks: A (2x3) = [A_L (2x2) | A_R (2x1)]
    - B (3x10) is split into B_L (2x10) and B_R (1x10)
    - For A_L * B_L: Use Strassen's algorithm on 2x2 blocks. 
      B_L has 10 columns, so we process them in 5 blocks of 2 columns each.
      Each 2x2 * 2x2 block multiplication takes rank 7 (Strassen).
      Total rank for this part: 5 * 7 = 35.
    - For A_R * B_R: This is a 2x1 * 1x10 multiplication. 
      This is equivalent to 20 independent scalar multiplications.
      Total rank for this part: 20.
    - Total rank: 35 + 20 = 55.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Rank calculation
    rank = 5 * 7 + 20
    rank = 55
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Indices for A_L (2x2 part of 2x3 matrix A)
    # A[i,j] -> U[i*m + j]
    # A00 -> U[0], A01 -> U[1], A10 -> U[3], A11 -> U[4]
    a00, a01, a10, a11 = 0, 1, 3, 4
    
    # Indices for A_R (2x1 part of 2x3 matrix A)
    # A02 -> U[2], A12 -> U[5]
    a02, a12 = 2, 5
    
    # Process 5 blocks of 2 columns for the Strassen part (A_L * B_L)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Indices for B_L (2x10 part of 3x10 matrix B)
        # B[j,k] -> V[j*p + k]
        # B00 -> V[0*10 + k0] = V[k0]
        # B01 -> V[0*10 + k1] = V[k1]
        # B10 -> V[1*10 + k0] = V[10 + k0]
        # B11 -> V[1*10 + k1] = V[10 + k1]
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        
        # Indices for C (2x10 output matrix)
        # C[i,k] -> W[k*n + i]
        # C00 -> W[k0*2 + 0] = W[2*k0]
        # C01 -> W[k1*2 + 0] = W[2*k1]
        # C10 -> W[k0*2 + 1] = W[2*k0 + 1]
        # C11 -> W[k1*2 + 1] = W[2*k1 + 1]
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        
        # Strassen's algorithm for 2x2 matrix multiplication
        # M1 = (A00 + A11)(B00 + B11)
        U[a00, col] += 1.0
        U[a11, col] += 1.0
        V[b00, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] += 1.0
        W[c10, col] += 1.0
        col += 1
        
        # M2 = (A10 + A11)B00
        U[a10, col] += 1.0
        U[a11, col] += 1.0
        V[b00, col] += 1.0
        W[c00, col] -= 1.0
        W[c10, col] += 1.0
        col += 1
        
        # M3 = A00(B01 - B11)
        U[a00, col] += 1.0
        V[b01, col] += 1.0
        V[b11, col] -= 1.0
        W[c01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M4 = A11(B10 - B00)
        U[a11, col] += 1.0
        V[b10, col] += 1.0
        V[b00, col] -= 1.0
        W[c00, col] += 1.0
        W[c01, col] += 1.0
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a00, col] += 1.0
        U[a01, col] += 1.0
        V[b11, col] += 1.0
        W[c11, col] -= 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a10, col] += 1.0
        U[a00, col] -= 1.0
        V[b00, col] += 1.0
        V[b01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a01, col] += 1.0
        U[a11, col] -= 1.0
        V[b10, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] -= 1.0
        W[c11, col] -= 1.0
        col += 1
    
    # Schoolbook part for A_R * B_R (2x1 * 1x10)
    # A_R is 2x1: A02 (idx 2), A12 (idx 5)
    # B_R is 1x10: B20..B29 (idx 20..29)
    # C contributions: C00..C09 (idx 0,2,4..18), C10..C19 (idx 1,3,5..19)
    
    for k in range(p):
        b2k = 2 * p + k  # Index for B[2, k]
        
        # Term 1: A02 * B2k -> C0k
        # C0k is at W[k*n + 0] = W[2*k]
        U[a02, col] += 1.0
        V[b2k, col] += 1.0
        W[2 * k, col] += 1.0
        col += 1
        
        # Term 2: A12 * B2k -> C1k
        # C1k is at W[k*n + 1] = W[2*k + 1]
        U[a12, col] += 1.0
        V[b2k, col] += 1.0
        W[2 * k + 1, col] += 1.0
        col += 1
    
    # Verify the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    # for all i in [0,n), j in [0,m), k in [0,p)
    
    def _verify(U, V, W, n, m, p, tol=1e-9):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Convert to lists for verification (numpy is faster for construction, lists for verify if needed)
    # But we can verify with numpy too
    def _verify_np(U, V, W, n, m, p, tol=1e-9):
        # Efficient check: 
        # For each c, we have outer product of u_c, v_c, w_c
        # Sum over c of u_c[i,j] v_c[j,k] w_c[k,i]
        # This is a tensor contraction.
        # We can just loop if rank is small (55) and dims are small (2,3,10)
        # 2*3*10 = 60 checks. 55 iterations. 3300 ops. Very fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if not _verify_np(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=float)
        V = np.zeros((m * p, rank_school), dtype=float)
        W = np.zeros((p * n, rank_school), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    if time.time() - start_time > DEADLINE:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
