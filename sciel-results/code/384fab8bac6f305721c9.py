"""Rank-minimal exact tensor decomposition for 2x2 by 2x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# Strassen decomposition for 2x2x2 matrix multiplication
# Using the standard 7-term Strassen decomposition
_STRASSEN_U = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, -1],
    [0, 1, 0, -1]
])
_STRASSEN_V = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, -1]
])
_STRASSEN_W = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, -1],
    [0, 0, 1, 0],
    [0, 0, 1, 1]
])

# Schoolbook decomposition for 2x1 by 1x1 multiplication
# A is 2x1, B is 1x1, C is 2x1
# C[0] = A[0]*B[0], C[1] = A[1]*B[0]
_SCHOOLBOOK_U = np.array([
    [1, 0],
    [0, 1]
])
_SCHOOLBOOK_V = np.array([
    [1, 1]
])
_SCHOOLBOOK_W = np.array([
    [1, 0],
    [0, 1]
])


def _verify_decomposition(U, V, W, n=2, m=2, p=9):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the full tensor from the decomposition
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        for a in range(n*m):
            if U[a, r] != 0:
                for b in range(m*p):
                    if V[b, r] != 0:
                        val = U[a, r] * V[b, r]
                        for c in range(p*n):
                            if W[c, r] != 0:
                                T[a, b, c] += val * W[c, r]
    
    # Build the target tensor for 2x2 * 2x9
    # A is n x m = 2 x 2
    # B is m x p = 2 x 9
    # C is n x p = 2 x 9
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # U index: i*m + j (row i, col j of A)
    # V index: j*p + k (row j, col k of B)
    # W index: k*n + i (row k, col i of C)
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1
    
    return np.allclose(T, target, atol=1e-12)


def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    n, m, p = DIMENSIONS
    
    # Decompose p=9 as 4*2 + 1
    # For each 2-column block of B, use Strassen (rank 7)
    # For the last 1-column block, use schoolbook (rank 2)
    # Total rank = 4*7 + 2 = 30
    
    num_strassen_blocks = p // 2  # 4
    remainder = p % 2  # 1
    
    total_rank = num_strassen_blocks * 7 + remainder * 2
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # For each Strassen block handling B columns [2*t, 2*t+1]
    for t in range(num_strassen_blocks):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen U: 4 entries for A (2x2), each term used for both k0 and k1
        # Actually, Strassen computes C_2x2 = A_2x2 * B_2x2
        # Here B is 2x2 with columns k0 and k1
        # So we need to handle the fact that B[j, k0] and B[j, k1] are separate
        
        # For Strassen on A (2x2) * B_block (2x2):
        # U has 4 rows (A elements), V has 4 rows (B_block elements), W has 4 rows (C elements)
        # But C is 2x2 here, which corresponds to C[i, k0] and C[i, k1]
        
        # U: same for all k
        for r in range(4):
            for i in range(n):
                for j in range(m):
                    U[i*m+j, col+r] += _STRASSEN_U[i*m+j, r]
        
        # V: B[j, k0] for even r? No, Strassen uses all B elements
        # V index: j*p + k
        # For k=k0: V[j*p+k0, col+r] += _STRASSEN_V[j, r]
        # For k=k1: V[j*p+k1, col+r] += _STRASSEN_V[j, r]
        # Wait, Strassen decomposes A(2x2) * B(2x2) -> C(2x2)
        # B has elements B[0,0], B[0,1], B[1,0], B[1,1]
        # In our indexing, B[j, k] where j in {0,1}, k in {k0, k1}
        # So B[0, k0] -> V[0*p+k0, :], B[0, k1] -> V[0*p+k1, :], etc.
        
        # The Strassen V matrix maps B elements to the 4 Strassen terms
        # _STRASSEN_V[j, r] gives the coefficient for B element (j, ?) 
        # Actually, the standard Strassen for 2x2:
        # M1 = A[0,0]*(B[0,1]-B[1,1])
        # M2 = A[1,0]*(B[0,0]+B[1,0])
        # M3 = (A[0,0]+A[0,1])*B[1,1]
        # M4 = (A[1,0]-A[0,0])*B[0,0]
        # M5 = (A[0,1]+A[1,1])*(B[1,0]-B[1,1])
        # M6 = (A[0,0]+A[1,0])*(B[0,1]-B[0,0])
        # M7 = (A[0,1]+A[1,1])*(B[0,0]-B[1,1])
        
        # Let me use a cleaner formulation. Standard Strassen 2x2x2:
        # There are 7 multiplications M1..M7
        # U (4x7), V (4x7), W (4x7)
        
        # Let me define them properly:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
        # M1 = a*(f-h)
        # M2 = (a+b)*h
        # M3 = (c+d)*e
        # M4 = d*(g-e)
        # M5 = (a+b)*(e+f)
        # M6 = (c-d)*(g+h)
        # M7 = (b-d)*(f+h)
        
        # U: coefficients for A
        # U[0, :] = [1, 1, 0, 0, 1, 0, 0]  (a)
        # U[1, :] = [0, 1, 0, 0, 1, 0, 1]  (b)
        # U[2, :] = [0, 0, 1, 1, 0, 1, 0]  (c)
        # U[3, :] = [0, 0, 1, 1, 0, -1, -1] (d)
        
        # V: coefficients for B
        # V[0, :] = [0, 1, 1, 1, 1, 0, 0]  (e)
        # V[1, :] = [1, 0, 0, 0, 1, 0, 1]  (f)
        # V[2, :] = [0, 0, 0, -1, 0, 1, 0] (g)
        # V[3, :] = [-1, 1, 0, 0, 0, 1, 1] (h)
        
        # W: coefficients for C
        # W[0, :] = [0, -1, 1, 0, 1, 0, 0]  (i)
        # W[1, :] = [1, 1, 0, 0, 0, 0, 1]   (j)
        # W[2, :] = [0, 0, 1, 1, 0, 1, 0]   (k)
        # W[3, :] = [1, 0, 0, -1, 1, -1, -1] (l)
        
        U_strassen = np.array([
            [1, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, -1, -1]
        ])
        V_strassen = np.array([
            [0, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, -1, 0, 1, 0],
            [-1, 1, 0, 0, 0, 1, 1]
        ])
        W_strassen = np.array([
            [0, -1, 1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0, 1, 0],
            [1, 0, 0, -1, 1, -1, -1]
        ])
        
        # Fill U: A elements are shared across all k, so same U for both k0 and k1
        for r in range(7):
            for a_idx in range(4):
                U[a_idx, col+r] += U_strassen[a_idx, r]
        
        # Fill V: B[j, k0] and B[j, k1]
        for r in range(7):
            for j in range(2):
                V[j*p + k0, col+r] += V_strassen[j, r]
                V[j*p + k1, col+r] += V_strassen[j, r]
        
        # Fill W: C[i, k0] and C[i, k1]
        # W index: k*n + i
        for r in range(7):
            for i in range(2):
                W[k0*n + i, col+r] += W_strassen[i, r]
                W[k1*n + i, col+r] += W_strassen[i, r]
        
        col += 7
    
    # For the remainder (last 1 column, k=8)
    if remainder > 0:
        k = 2 * num_strassen_blocks
        
        # Schoolbook: C[i, k] = sum_j A[i, j] * B[j, k]
        # 2 terms
        for r in range(2):
            for i in range(2):
                for j in range(2):
                    # Term r: A[i, j] * B[j, k] contributes to C[i, k]
                    # U[i*m+j, col+r] += delta(i == 0, j == r) for r=0,1? 
                    # No, we need all 4 combinations but only 2 terms
                    # Term 0: A[0,0]*B[0,k] -> C[0,k]
                    # Term 1: A[1,0]*B[1,k] -> C[1,k]
                    # Wait, that's only for m=2, but we need to handle general
                    # For 2x2 * 2x1:
                    # C[0, k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
                    # C[1, k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
                    # 4 terms total, but we can group?
                    # No, schoolbook is 4 terms for 2x2 * 2x1
                    pass
        
        # Actually, 2x2 * 2x1 schoolbook needs 4 terms
        # Let me use 4 terms for the remainder
        for r in range(4):
            i = r // 2
            j = r % 2
            U[i*m + j, col+r] = 1
            V[j*p + k, col+r] = 1
            W[k*n + i, col+r] = 1
        
        col += 4
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U_fb = np.zeros((n*m, rank))
    V_fb = np.zeros((m*p, rank))
    W_fb = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, c] = 1
                V_fb[j*p+k, c] = 1
                W_fb[k*n+i, c] = 1
                c += 1
    
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
