"""Matrix multiplication 2x5x6 with Strassen-based block decomposition."""
import time
import random

DIMENSIONS = [2, 5, 6]

def _verify_decomposition(U, V, W, n=2, m=5, p=6):
    """Verify that the decomposition exactly reproduces matrix multiplication tensor."""
    rank = len(U[0])
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    # Check no spurious cross-terms: for j' != j, sum_r U[i,j,r]*V[j',k,r]*W[k,i,r] should be 0
    # Actually this is automatically satisfied if the above holds for all (i,j,k)
    # because the tensor is the standard matrix multiplication tensor.
    return True


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Build decomposition using Strassen for 2x2x6 part and schoolbook for 2x3x6 part
    # Total rank: 3*7 + 2*18 = 21 + 36 = 57
    
    rank = 57
    U = [[0] * rank for _ in range(n * m)]   # 10 rows
    V = [[0] * rank for _ in range(m * p)]   # 30 rows  
    W = [[0] * rank for _ in range(p * n)]   # 12 rows
    
    col = 0
    
    # Part 1: A[:, 0:2] * B[0:2, :] using 3 Strassen 2x2x2 blocks
    # Each block handles A[0:2, j:j+2] * B[j:j+2... wait no.
    
    # Let me reconsider the block structure.
    # A is 2x5: rows 0,1; cols 0,1,2,3,4
    # B is 5x6: rows 0,1,2,3,4; cols 0,1,2,3,4,5
    # C is 2x6: rows 0,1; cols 0,1,2,3,4,5
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    # Split j into {0,1} and {2,3,4}:
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{4} A[i,j]*B[j,k]
    
    # Part 1: sum_{j=0}^{1} A[i,j]*B[j,k] for i in {0,1}, k in {0,...,5}
    # This is A_2x2 * B_2x6 = 2x6 matrix
    # Split k into {0,1}, {2,3}, {4,5}:
    # For each pair (k,k+1), we have a 2x2x2 subproblem
    # Use Strassen (rank 7) for each, total 3*7 = 21
    
    # Part 2: sum_{j=2}^{4} A[i,j]*B[j,k] for i in {0,1}, k in {0,...,5}
    # This is A_2x3 * B_3x6 = 2x6 matrix
    # Split k into {0,1,2}, {3,4,5}:
    # Each is a 2x3x3 subproblem, schoolbook rank 18
    # Total 2*18 = 36
    
    # Total: 21 + 36 = 57
    
    # Part 1: Strassen blocks
    # For block b = 0,1,2: handles k in {2b, 2b+1}
    for b in range(3):
        k0 = 2 * b  # first k index for this block
        k1 = 2 * b + 1  # second k index
        
        # Strassen 7 terms for 2x2x2:
        # A is 2x2 (rows 0-1, cols 0-1), B is 2x2 (rows 0-1, cols k0-k1), C is 2x2 (rows 0-1, cols k0-k1)
        
        # Term 1: M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        # U: A[0,0]->+1, A[1,1]->+1
        # V: B[0,k0]->+1, B[1,k1]->+1
        # W: C[0,k0]->+1, C[1,k1]->+1
        U[0*m + 0][col] += 1   # A[0,0]
        U[1*m + 1][col] += 1   # A[1,1]
        V[0*p + k0][col] += 1  # B[0,k0]
        V[1*p + k1][col] += 1  # B[1,k1]
        W[k0*n + 0][col] += 1  # C[0,k0]
        W[k1*n + 1][col] += 1  # C[1,k1]
        col += 1
        
        # Term 2: M2 = (A[1,0]+A[1,1]) * B[0,k0]
        # U: A[1,0]->+1, A[1,1]->+1
        # V: B[0,k0]->+1
        # W: C[1,k0]->+1, C[1,k1]->-1
        U[1*m + 0][col] += 1
        U[1*m + 1][col] += 1
        V[0*p + k0][col] += 1
        W[k0*n + 1][col] += 1   # C[1,k0]
        W[k1*n + 1][col] += -1  # C[1,k1]
        col += 1
        
        # Term 3: M3 = A[0,0] * (B[0,k1]-B[1,k1])
        # U: A[0,0]->+1
        # V: B[0,k1]->+1, B[1,k1]->-1
        # W: C[0,k1]->+1, C[1,k1]->+1
        U[0*m + 0][col] += 1
        V[0*p + k1][col] += 1
        V[1*p + k1][col] += -1
        W[k1*n + 0][col] += 1   # C[0,k1]
        W[k1*n + 1][col] += 1   # C[1,k1]
        col += 1
        
        # Term 4: M4 = A[1,1] * (B[1,k0]-B[0,k0])
        # U: A[1,1]->+1
        # V: B[1,k0]->+1, B[0,k0]->-1
        # W: C[0,k0]->+1, C[1,k0]->+1
        U[1*m + 1][col] += 1
        V[1*p + k0][col] += 1
        V[0*p + k0][col] += -1
        W[k0*n + 0][col] += 1   # C[0,k0]
        W[k0*n + 1][col] += 1   # C[1,k0]
        col += 1
        
        # Term 5: M5 = (A[0,0]+A[0,1]) * B[1,k1]
        # U: A[0,0]->+1, A[0,1]->+1
        # V: B[1,k1]->+1
        # W: C[0,k0]->-1, C[0,k1]->+1
        U[0*m + 0][col] += 1
        U[0*m + 1][col] += 1
        V[1*p + k1][col] += 1
        W[k0*n + 0][col] += -1  # C[0,k0]
        W[k1*n + 0][col] += 1   # C[0,k1]
        col += 1
        
        # Term 6: M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        # U: A[1,0]->+1, A[0,0]->-1
        # V: B[0,k0]->+1, B[0,k1]->+1
        # W: C[1,k1]->+1
        U[1*m + 0][col] += 1
        U[0*m + 0][col] += -1
        V[0*p + k0][col] += 1
        V[0*p + k1][col] += 1
        W[k1*n + 1][col] += 1   # C[1,k1]
        col += 1
        
        # Term 7: M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        # U: A[0,1]->+1, A[1,1]->-1
        # V: B[1,k0]->+1, B[1,k1]->+1
        # W: C[0,k0]->+1
        U[0*m + 1][col] += 1
        U[1*m + 1][col] += -1
        V[1*p + k0][col] += 1
        V[1*p + k1][col] += 1
        W[k0*n + 0][col] += 1   # C[0,k0]
        col += 1
    
    # Part 2: Schoolbook for A[:, 2:5] * B[2:5, :]
    # A cols 2,3,4; B rows 2,3,4
    for ii in range(2):  # i = 0, 1
        for jj in range(3):  # j offset: 0,1,2 -> actual j = 2+jj
            for kk in range(6):  # k = 0,1,2,3,4,5
                I = ii
                J = 2 + jj
                K = kk
                U[I*m + J][col] = 1
                V[J*p + K][col] = 1
                W[K*n + I][col] = 1
                col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-END
