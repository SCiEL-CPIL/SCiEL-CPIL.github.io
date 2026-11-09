"""Matrix multiplication 4,5,6 - Strassen-based rank reduction."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_decompose(n, m, p):
    """
    Decompose matrix multiplication n x m, m x p -> n x p using recursive Strassen.
    Returns (U, V, W) as lists of lists with integer entries.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    W[k*n + i, c] corresponds to output[i,k] contribution.
    """
    if n == 1 and m == 1 and p == 1:
        # Base case: rank 1
        return [[1]], [[1]], [[1]]
    
    # Choose split sizes
    # For Strassen, we want to split into two halves each
    n1, n2 = n // 2, n - n // 2
    m1, m2 = m // 2, m - m // 2
    p1, p2 = p // 2, p - p // 2
    
    # For proper Strassen, we need equal halves. If not equal, pad.
    # Actually, let's use a simpler approach: pad to next power of 2,
    # do full Strassen, then extract.
    
    # Find next power of 2 >= max(n, m, p)
    def next_pow2(x):
        x = max(1, x)
        p2 = 1
        while p2 < x:
            p2 *= 2
        return p2
    
    N = next_pow2(max(n, m, p))
    
    # Get full Strassen decomposition for N x N x N
    U_full, V_full, W_full = _strassen_full(N, N, N)
    
    # Now extract the sub-decomposition for n x m x p
    # U_full has shape (N*N, rank_full), indexed by [i*N + j]
    # V_full has shape (N*N, rank_full), indexed by [j*N + k]
    # W_full has shape (N*N, rank_full), indexed by [k*N + i]
    
    rank_full = len(U_full[0])
    
    # Build mapping: for each term in the full decomposition,
    # it contributes to output[i,k] via sum over j of A[i,j]*B[j,k]
    # We need to restrict i < n, j < m, k < p
    
    # The full tensor is: sum_c U_full[i*N+j, c] * V_full[j*N+k, c] * W_full[k*N+i, c]
    # = sum over (i,j,k) of (sum_c U[iN+j,c] V[jN+k,c] W[kN+i,c]) * A[i,j] B[j,k]
    
    # For the sub-problem, we only care about i in [0,n), j in [0,m), k in [0,p)
    # So we need to find a subset of columns c and possibly linear combinations
    # such that for all i<n, j<m, k<p:
    #   sum_{c in subset} U_sub[i*m+j, c'] * V_sub[j*p+k, c'] * W_sub[k*n+i, c'] = 1
    # and for all other (i,j,k) combinations that would be "outside", they don't matter
    # because we're just defining the decomposition for the sub-tensor.
    
    # Actually, the correct approach: the full Strassen decomposition satisfies
    # sum_c U_full[iN+j,c] V_full[jN+k,c] W_full[kN+i,c] = 1 for all i,j,k in [0,N)
    # We want a decomposition for i in [0,n), j in [0,m), k in [0,p)
    
    # We can use all rank_full columns, but remap indices:
    # U_sub[i*m+j, c] = U_full[i*N+j, c] for i<n, j<m
    # V_sub[j*p+k, c] = V_full[j*N+k, c] for j<m, k<p
    # W_sub[k*n+i, c] = W_full[k*N+i, c] for k<p, i<n
    
    # This gives a valid decomposition of rank rank_full.
    
    U_sub = [[U_full[i*N + j][c] for c in range(rank_full)] for i in range(n) for j in range(m)]
    V_sub = [[V_full[j*N + k][c] for c in range(rank_full)] for j in range(m) for k in range(p)]
    W_sub = [[W_full[k*N + i][c] for c in range(rank_full)] for k in range(p) for i in range(n)]
    
    # Now try to reduce rank by removing zero columns
    # A column c is zero if U_sub[:,c] is all zero, or V_sub[:,c] is all zero, or W_sub[:,c] is all zero
    # Actually, we need all three to be nonzero for the term to contribute.
    
    keep_cols = []
    for c in range(rank_full):
        u_nonzero = any(U_sub[i][c] != 0 for i in range(n * m))
        v_nonzero = any(V_sub[i][c] != 0 for i in range(m * p))
        w_nonzero = any(W_sub[i][c] != 0 for i in range(p * n))
        if u_nonzero and v_nonzero and w_nonzero:
            keep_cols.append(c)
    
    U_final = [[U_sub[i][c] for c in keep_cols] for i in range(n * m)]
    V_final = [[V_sub[i][c] for c in keep_cols] for i in range(m * p)]
    W_final = [[W_sub[i][c] for c in keep_cols] for i in range(p * n)]
    
    return U_final, V_final, W_final


def _strassen_full(n, m, p):
    """
    Full Strassen decomposition for n x m x p where n=m=p is a power of 2.
    Returns (U, V, W) as lists of lists.
    """
    if n == 1:
        return [[1]], [[1]], [[1]]
    
    half = n // 2
    # Recursively get decomposition for half x half x half
    U_h, V_h, W_h = _strassen_full(half, half, half)
    
    rank_h = len(U_h[0])
    
    # Strassen combines 7 recursive calls, each of rank rank_h
    # Total rank = 7 * rank_h
    rank = 7 * rank_h
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Output:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each recursive call t (0..6), we have rank_h terms.
    # For term t, sub-term s:
    #   U contribution: from the appropriate block of A
    #   V contribution: from the appropriate block of B
    #   W contribution: to the appropriate block of C
    
    # Block indexing:
    # A is n x m, split into A11 (top-left), A12 (top-right), A21 (bottom-left), A22 (bottom-right)
    # Each block is half x half
    # A[i,j] where i in [0,half) is top, [half, n) is bottom
    # j in [0,half) is left, [half, m) is right
    
    # For U: row index is i*m + j
    # For V: row index is j*p + k  
    # For W: row index is k*n + i
    
    # Strassen terms and their A-block, B-block, C-blocks:
    # t=0: M1: A=(A11+A22), B=(B11+B22), C: C11+=1, C22+=1
    # t=1: M2: A=(A21+A22), B=B11, C: C21+=1, C22-=1
    # t=2: M3: A=A11, B=(B12-B22), C: C12+=1, C22+=1
    # t=3: M4: A=A22, B=(B21-B11), C: C11+=1, C21+=1
    # t=4: M5: A=(A11+A12), B=B22, C: C11-=1, C12+=1
    # t=5: M6: A=(A21-A11), B=(B11+B12), C: C22+=1
    # t=6: M7: A=(A11+A22), B=(B12+B22), C: C11+=1, C22+=1
    
    # Let me define the Strassen coefficients more carefully.
    # For each term t, we need:
    #   A_coeffs: list of (block, coeff) for A
    #   B_coeffs: list of (block, coeff) for B
    #   C_coeffs: list of (block, coeff) for C
    
    # Block indices: 0=A11, 1=A12, 2=A21, 3=A22 for A
    #                 0=B11, 1=B12, 2=B21, 3=B22 for B
    #                 0=C11, 1=C12, 2=C21, 3=C22 for C
    
    strassen_terms = [
        # (A_blocks, B_blocks, C_blocks)
        # Each is a list of (block_idx, coefficient)
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),    # M1
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),              # M2
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),              # M3
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ]),              # M4
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]),              # M5
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]),              # M6
        ([ (0, 1), (3, 1) ], [ (1, 1), (3, 1) ], [ (0, 1), (3, 1) ]),       # M7
    ]
    
    for t in range(7):
        A_blocks, B_blocks, C_blocks = strassen_terms[t]
        base = t * rank_h
        
        for s in range(rank_h):
            c = base + s
            
            # U: sum over A blocks
            for (ablock, acoeff) in A_blocks:
                # ablock: 0=A11, 1=A12, 2=A21, 3=A22
                ai = ablock // 2  # 0 for top, 1 for bottom
                aj = ablock % 2   # 0 for left, 1 for right
                for i in range(half):
                    for j in range(half):
                        ui = (ai * half + i) * m + (aj * half + j)
                        U[ui][c] += acoeff * U_h[i * half + j][s]
            
            # V: sum over B blocks
            for (bblock, bcoeff) in B_blocks:
                bj = bblock // 2   # 0 for top, 1 for bottom
                bk = bblock % 2    # 0 for left, 1 for right
                for j in range(half):
                    for k in range(half):
                        vj = (bj * half + j) * p + (bk * half + k)
                        V[vj][c] += bcoeff * V_h[j * half + k][s]
            
            # W: sum over C blocks
            for (cblock, ccoeff) in C_blocks:
                ci = cblock // 2   # 0 for top, 1 for bottom
                ck = cblock % 2    # 0 for left, 1 for right
                for k in range(half):
                    for i in range(half):
                        wk = (ck * half + k) * n + (ci * half + i)
                        W[wk][c] += ccoeff * W_h[k * half + i][s]
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    # Check that for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Try Strassen-based decomposition
    try:
        U, V, W = _strassen_decompose(n, m, p)
        rank = len(U[0])
        
        # Verify correctness
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
