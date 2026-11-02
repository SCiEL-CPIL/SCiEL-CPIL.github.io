import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: n*m x rank = 8 x rank
    # V: m*p x rank = 32 x rank
    # W: p*n x rank = 16 x rank
    
    # Rank 56 construction using Strassen on 2x2 blocks
    # The 2x4 * 4x8 multiplication can be viewed as 8 independent 2x2x2 sub-problems
    # Each 2x2 multiplication has rank 7 (Strassen)
    # Total rank = 8 * 7 = 56
    
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # A is 2x4: A[i,j] with i in {0,1}, j in {0,1,2,3}
    # B is 4x8: B[j,k] with j in {0,1,2,3}, k in {0,...,7}
    # C is 2x8: C[i,k] with i in {0,1}, k in {0,...,7}
    # W index convention: W[k*n+i] = W[k*2+i]
    
    # Split into 8 blocks: u in {0,1} (middle dim pairs), v in {0,1,2,3} (last dim pairs)
    # For each (u,v): A block is rows 0-1, cols 2u,2u+1
    #                 B block is rows 2u,2u+1, cols 2v,2v+1
    #                 C block is rows 0-1, cols 2v,2v+1
    
    for u in range(2):
        for v in range(4):
            # Check time
            if time.time() > deadline:
                break
                
            base_col = 7 * (4 * u + v)
            
            # A block indices (row-major in U: i*m + j)
            a11_idx = 0 * m + 2 * u        # A[0, 2u]
            a12_idx = 0 * m + 2 * u + 1    # A[0, 2u+1]
            a21_idx = 1 * m + 2 * u        # A[1, 2u]
            a22_idx = 1 * m + 2 * u + 1    # A[1, 2u+1]
            
            # B block indices (row-major in V: j*p + k)
            b11_idx = 2 * u * p + 2 * v          # B[2u, 2v]
            b12_idx = 2 * u * p + 2 * v + 1      # B[2u, 2v+1]
            b21_idx = (2 * u + 1) * p + 2 * v    # B[2u+1, 2v]
            b22_idx = (2 * u + 1) * p + 2 * v + 1  # B[2u+1, 2v+1]
            
            # C block indices (W convention: k*n + i)
            c11_idx = 2 * v * n + 0       # C[0, 2v]
            c12_idx = (2 * v + 1) * n + 0  # C[0, 2v+1]
            c21_idx = 2 * v * n + 1       # C[1, 2v]
            c22_idx = (2 * v + 1) * n + 1  # C[1, 2v+1]
            
            # Strassen's 7 multiplications for 2x2 blocks
            
            # M1 = (A11+A22)(B11+B22)
            U[a11_idx][base_col] += 1
            U[a22_idx][base_col] += 1
            V[b11_idx][base_col] += 1
            V[b22_idx][base_col] += 1
            W[c11_idx][base_col] += 1
            W[c22_idx][base_col] += 1
            
            # M2 = (A21+A22)B11
            U[a21_idx][base_col + 1] += 1
            U[a22_idx][base_col + 1] += 1
            V[b11_idx][base_col + 1] += 1
            W[c21_idx][base_col + 1] += 1
            W[c22_idx][base_col + 1] -= 1
            
            # M3 = A11(B12-B22)
            U[a11_idx][base_col + 2] += 1
            V[b12_idx][base_col + 2] += 1
            V[b22_idx][base_col + 2] -= 1
            W[c12_idx][base_col + 2] += 1
            W[c22_idx][base_col + 2] += 1
            
            # M4 = A22(B21-B11)
            U[a22_idx][base_col + 3] += 1
            V[b21_idx][base_col + 3] += 1
            V[b11_idx][base_col + 3] -= 1
            W[c11_idx][base_col + 3] += 1
            W[c21_idx][base_col + 3] += 1
            
            # M5 = (A11+A12)B22
            U[a11_idx][base_col + 4] += 1
            U[a12_idx][base_col + 4] += 1
            V[b22_idx][base_col + 4] += 1
            W[c11_idx][base_col + 4] -= 1
            W[c12_idx][base_col + 4] += 1
            
            # M6 = (A21-A11)(B11+B12)
            U[a21_idx][base_col + 5] += 1
            U[a11_idx][base_col + 5] -= 1
            V[b11_idx][base_col + 5] += 1
            V[b12_idx][base_col + 5] += 1
            W[c22_idx][base_col + 5] += 1
            
            # M7 = (A12-A22)(B21+B22)
            U[a12_idx][base_col + 6] += 1
            U[a22_idx][base_col + 6] -= 1
            V[b21_idx][base_col + 6] += 1
            V[b22_idx][base_col + 6] += 1
            W[c11_idx][base_col + 6] += 1

    return {"U": U, "V": V, "W": W}
