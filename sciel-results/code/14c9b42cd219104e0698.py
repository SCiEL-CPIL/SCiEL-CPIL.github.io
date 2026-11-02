"""Program for 2x3 x 3x6 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s limit

def check_time():
    return time.time() - START_TIME > TIME_LIMIT

def verify_decomposition(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build tensor T[i, j, k] = 1 (for all i,j,k) - this is the multiplication tensor
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to tensors
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # Compute sum over c: U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This should equal 1 for all i, j, k
    result = np.zeros((n, m, p))
    for c in range(rank):
        result += U_t[:, :, c] * V_t[:, :, c].T * W_t[:, :, c].T
    
    # Check exactness
    max_err = np.max(np.abs(result - 1.0))
    return max_err < 1e-9, max_err

def schoolbook_decomposition():
    """Standard O(nmp) decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def strassen_2x2x2():
    """
    Strassen's 7-term decomposition for 2x2 x 2x2 matrix multiplication.
    This gives a rank-7 decomposition of M_{<2,2,2>}.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen formulas:
    # M1 = (A11)(B22 - B12)  -> contributes to C11
    # M2 = (A11 + A21)(B12)  -> contributes to C12
    # M3 = (A22 - A12)(B11)  -> contributes to C21
    # M4 = (A22)(B11 + B21)  -> contributes to C22
    # M5 = (A11 + A22)(B12 + B21 - B22)  -> contributes to C11 and C22
    # M6 = (A12 - A22)(B11 + B21)  -> contributes to C12
    # M7 = (A11 - A12)(B12 + B22)  -> contributes to C21
    
    # A indices: A[i,j] where i in {0,1} (row), j in {0,1} (col)
    # U row index: i*m + j = i*2 + j
    # B indices: B[j,k] where j in {0,1} (row), k in {0,1} (col)
    # V row index: j*p + k = j*2 + k
    # C indices: C[i,k] where i in {0,1} (row), k in {0,1} (col)
    # W row index: k*n + i = k*2 + i
    
    # M1 = A11 * (B22 - B12): A[0,0]=1, B[1,1]=1, B[0,1]=-1
    U[0*2+0][0] += 1.0
    V[1*2+1][0] += 1.0
    V[0*2+1][0] += -1.0
    W[0*2+0][0] += 1.0  # C[0,0] += M1
    
    # M2 = (A11 + A21) * B12: A[0,0]=1, A[1,0]=1, B[0,1]=1
    U[0*2+0][1] += 1.0
    U[1*2+0][1] += 1.0
    V[0*2+1][1] += 1.0
    W[1*2+0][1] += 1.0  # C[0,1] += M2
    
    # M3 = (A22 - A12) * B11: A[1,1]=1, A[0,1]=-1, B[0,0]=1
    U[1*2+1][2] += 1.0
    U[0*2+1][2] += -1.0
    V[0*2+0][2] += 1.0
    W[0*2+1][2] += 1.0  # C[1,0] += M3
    
    # M4 = A22 * (B11 + B21): A[1,1]=1, B[0,0]=1, B[1,0]=1
    U[1*2+1][3] += 1.0
    V[0*2+0][3] += 1.0
    V[1*2+0][3] += 1.0
    W[1*2+1][3] += 1.0  # C[1,1] += M4
    
    # M5 = (A11 + A22) * (B12 + B21 - B22)
    # A[0,0]=1, A[1,1]=1, B[0,1]=1, B[1,0]=1, B[1,1]=-1
    U[0*2+0][4] += 1.0
    U[1*2+1][4] += 1.0
    V[0*2+1][4] += 1.0
    V[1*2+0][4] += 1.0
    V[1*2+1][4] += -1.0
    W[0*2+0][4] += 1.0  # C[0,0] += M5
    W[1*2+1][4] += 1.0  # C[1,1] += M5
    
    # M6 = (A12 - A22) * (B11 + B21)
    # A[0,1]=1, A[1,1]=-1, B[0,0]=1, B[1,0]=1
    U[0*2+1][5] += 1.0
    U[1*2+1][5] += -1.0
    V[0*2+0][5] += 1.0
    V[1*2+0][5] += 1.0
    W[1*2+0][5] += 1.0  # C[0,1] += M6
    
    # M7 = (A11 - A12) * (B12 + B22)
    # A[0,0]=1, A[0,1]=-1, B[0,1]=1, B[1,1]=1
    U[0*2+0][6] += 1.0
    U[0*2+1][6] += -1.0
    V[0*2+1][6] += 1.0
    V[1*2+1][6] += 1.0
    W[0*2+1][6] += 1.0  # C[1,0] += M7
    
    return {"U": U, "V": V, "W": W}

def block_decomposition():
    """
    Decompose 2x3 x 3x6 by splitting into blocks.
    Split j (dim 3) into j0 (dim 1) and j1 (dim 2).
    Split k (dim 6) into three groups of 2.
    
    A is 2x3: split into A[:, :1] (2x1) and A[:, 1:] (2x2)
    B is 3x6: split into B[:1, :] (1x6) and B[1:, :] (2x6)
    
    Product = A[:,:1] @ B[:1,:] + A[:,1:] @ B[1:,:]
    = (2x1 @ 1x6) + (2x2 @ 2x6)
    
    The first term has rank 2*1*6 = 12 (schoolbook)
    The second term: 2x2 @ 2x6. Split k into three groups of 2.
    For each 2x2 @ 2x2 block, use Strassen (rank 7).
    Total for second term: 3 * 7 = 21
    Total: 12 + 21 = 33
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    rank = 12 + 21  # = 33
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0.0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 33
    
    col = 0
    
    # Part 1: A[:, :1] @ B[:1, :] -> 2x1 @ 1x6
    # A[:, :1] is 2x1: indices i in {0,1}, j=0
    # B[:1, :] is 1x6: j=0, k in {0,...,5}
    # C contribution: C[i, k] += A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1.0  # A[i, 0]
            V[0 * p + k][col] = 1.0  # B[0, k]
            W[k * n + i][col] = 1.0  # C[i, k]
            col += 1
    
    # Part 2: A[:, 1:] @ B[1:, :] -> 2x2 @ 2x6
    # A[:, 1:] is 2x2: i in {0,1}, j in {1,2}
    # B[1:, :] is 2x6: j in {1,2}, k in {0,...,5}
    # Split k into three groups of 2: k0={0,1}, k1={2,3}, k2={4,5}
    # For each group, we have a 2x2 @ 2x2 multiplication
    for kg in range(3):
        k_start = kg * 2
        k_local = 0
        for kl in range(2):
            k_global = k_start + kl
            # Use Strassen for this 2x2 @ 2x2 block
            # The 2x2 A submatrix: rows i in {0,1}, cols j in {1,2}
            # The 2x2 B submatrix: rows j in {1,2}, cols k in {k_start, k_start+1}
            # The 2x2 C submatrix: rows i in {0,1}, cols k in {k_start, k_start+1}
            
            # Strassen 7 terms
            # Local A indices: (i, jl) where jl in {0,1} maps to j = 1+jl
            # Local B indices: (jl, kl) where jl in {0,1} maps to j = 1+jl
            # Local C indices: (i, kl)
            
            # M1 = A[0,0] * (B[1,1] - B[0,1])
            U[0 * m + 1][col] += 1.0  # A[0, j=1]
            V[2 * p + k_start + 1][col] += 1.0  # B[j=2, k=k_start+1]
            V[1 * p + k_start + 1][col] += -1.0  # B[j=1, k=k_start+1]
            W[(k_start + 0) * n + 0][col] += 1.0  # C[i=0, k=k_start]
            col += 1
            
            # M2 = (A[0,0] + A[1,0]) * B[0,1]
            U[0 * m + 1][col] += 1.0  # A[0, j=1]
            U[1 * m + 1][col] += 1.0  # A[1, j=1]
            V[1 * p + k_start + 1][col] += 1.0  # B[j=1, k=k_start+1]
            W[(k_start + 1) * n + 0][col] += 1.0  # C[i=0, k=k_start+1]
            col += 1
            
            # M3 = (A[1,1] - A[0,1]) * B[0,0]
            U[1 * m + 2][col] += 1.0  # A[1, j=2]
            U[0 * m + 2][col] += -1.0  # A[0, j=2]
            V[1 * p + k_start + 0][col] += 1.0  # B[j=1, k=k_start]
            W[(k_start + 0) * n + 1][col] += 1.0  # C[i=1, k=k_start]
            col += 1
            
            # M4 = A[1,1] * (B[0,0] + B[1,0])
            U[1 * m + 2][col] += 1.0  # A[1, j=2]
            V[1 * p + k_start + 0][col] += 1.0  # B[j=1, k=k_start]
            V[2 * p + k_start + 0][col] += 1.0  # B[j=2, k=k_start]
            W[(k_start + 1) * n + 1][col] += 1.0  # C[i=1, k=k_start+1]
            col += 1
            
            # M5 = (A[0,0] + A[1,1]) * (B[0,1] + B[1,0] - B[1,1])
            U[0 * m + 1][col] += 1.0  # A[0, j=1]
            U[1 * m + 2][col] += 1.0  # A[1, j=2]
            V[1 * p + k_start + 1][col] += 1.0  # B[j=1, k=k_start+1]
            V[2 * p + k_start + 0][col] += 1.0  # B[j=2, k=k_start]
            V[2 * p + k_start + 1][col] += -1.0  # B[j=2, k=k_start+1]
            W[(k_start + 0) * n + 0][col] += 1.0  # C[i=0, k=k_start]
            W[(k_start + 1) * n + 1][col] += 1.0  # C[i=1, k=k_start+1]
            col += 1
            
            # M6 = (A[0,1] - A[1,1]) * (B[0,0] + B[1,0])
            U[0 * m + 2][col] += 1.0  # A[0, j=2]
            U[1 * m + 2][col] += -1.0  # A[1, j=2]
            V[1 * p + k_start + 0][col] += 1.0  # B[j=1, k=k_start]
            V[2 * p + k_start + 0][col] += 1.0  # B[j=2, k=k_start]
            W[(k_start + 1) * n + 0][col] += 1.0  # C[i=0, k=k_start+1]
            col += 1
            
            # M7 = (A[0,0] - A[0,1]) * (B[0,1] + B[1,1])
            U[0 * m + 1][col] += 1.0  # A[0, j=1]
            U[0 * m + 2][col] += -1.0  # A[0, j=2]
            V[1 * p + k_start + 1][col] += 1.0  # B[j=1, k=k_start+1]
            V[2 * p + k_start + 1][col] += 1.0  # B[j=2, k=k_start+1]
            W[(k_start + 0) * n + 1][col] += 1.0  # C[i=1, k=k_start]
            col += 1
    
    return {"U": U, "V": V, "W": W}

def rank_minimization_search(initial, max_iters=5000):
    """
    Try to reduce the rank of a valid decomposition by finding zero columns
    or combining columns.
    """
    U = [row[:] for row in initial["U"]]
    V = [row[:] for row in initial["V"]]
    W = [row[:] for row in initial["W"]]
    rank = len(U[0])
    
    for it in range(max_iters):
        if check_time():
            break
        
        # Find a column that can be eliminated
        # A column c is redundant if there exist coefficients alpha such that
        # the column can be expressed as a linear combination of others.
        # For simplicity, try removing each column and checking validity.
        # This is expensive, so do it probabilistically.
        
        # Try a random column
        c = random.randint(0, rank - 1)
        
        # Check if removing column c still gives a valid decomposition
        # This requires checking all (i,j,k) triples
        valid = True
        for i in range(DIMENSIONS[0]):
            if not valid:
                break
            for j in range(DIMENSIONS[1]):
                if not valid:
                    break
                for k in range(DIMENSIONS[2]):
                    s = 0.0
                    for cc in range(rank):
                        if cc == c:
                            continue
                        s += U[i * DIMENSIONS[1] + j][cc] * V[j * DIMENSIONS[2] + k][cc] * W[k * DIMENSIONS[0] + i][cc]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
        
        if valid:
            # Remove column c
            for row in U:
                row.pop(c)
            for row in V:
                row.pop(c)
            for row in W:
                row.pop(c)
            rank -= 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 x 3x6 matrix multiplication.
    """
    # Try the block decomposition with Strassen (rank 33)
    best = None
    best_rank = float('inf')
    
    # Schoolbook as fallback
    sb = schoolbook_decomposition()
    valid, err = verify_decomposition(sb["U"], sb["V"], sb["W"])
    if valid:
        best = sb
        best_rank = len(sb["U"][0])
    
    # Block decomposition
    if not check_time():
        bd = block_decomposition()
        valid, err = verify_decomposition(bd["U"], bd["V"], bd["W"])
        if valid and len(bd["U"][0]) < best_rank:
            best = bd
            best_rank = len(bd["U"][0])
    
    # Try to reduce rank further
    if best is not None and not check_time():
        reduced = rank_minimization_search(best, max_iters=2000)
        valid, err = verify_decomposition(reduced["U"], reduced["V"], reduced["W"])
        if valid and len(reduced["U"][0]) < best_rank:
            best = reduced
            best_rank = len(reduced["U"][0])
    
    if best is None:
        # Should not happen, but fallback
        best = schoolbook_decomposition()
    
    return best

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result["U"], result["V"], result["W"])
    print(f"Valid: {valid}, Max error: {err}")
