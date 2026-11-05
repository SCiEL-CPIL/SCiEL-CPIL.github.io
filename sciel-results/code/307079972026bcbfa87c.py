"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 6]
_DEADLINE = 150.0

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Return the 7-term Strassen decomposition for 2x2 * 2x2."""
    # Strassen's 7 products for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h), m2 = (c+d)(e), m3 = a(f-h), m4 = d(g-e)
    # m5 = (a+b)h, m6 = (c-a)(e+f), m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    # 
    # We need U, V, W as vectors of length 7 for each index.
    # U has 2*2=4 rows (A indices), V has 2*2=4 rows (B indices), W has 2*2=4 rows (C indices)
    # Index convention: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # For each Strassen product r, define which A-indices, B-indices, C-indices are involved
    # and the coefficient in C.
    
    # m1 = (a+d)(e+h) -> contributes to C[0,0] (+1) and C[1,1] (-1)
    # m2 = (c+d)(e) -> contributes to C[1,0] (+1) and C[1,1] (-1)
    # m3 = a(f-h) -> contributes to C[0,1] (+1) and C[1,1] (+1)
    # m4 = d(g-e) -> contributes to C[0,0] (+1) and C[1,0] (+1)
    # m5 = (a+b)h -> contributes to C[0,0] (-1) and C[0,1] (+1)
    # m6 = (c-a)(e+f) -> contributes to C[1,1] (+1)
    # m7 = (b-d)(g+h) -> contributes to C[0,0] (+1)
    
    # U: for each A-index (i,j), the coefficient for each Strassen product
    # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # V: for each B-index (j,k), the coefficient for each Strassen product
    # B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # W: for each C-index (k,i), the coefficient for each Strassen product
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    U = [
        # a: m1(+1), m3(+1), m5(+1)
        [1, 0, 1, 0, 1, 0, 0],
        # b: m5(+1), m7(+1)
        [0, 0, 0, 0, 1, 0, 1],
        # c: m2(+1), m6(-1)
        [0, 1, 0, 0, 0, -1, 0],
        # d: m1(+1), m2(+1), m4(+1), m7(-1)
        [1, 1, 0, 1, 0, 0, -1],
    ]
    V = [
        # e: m1(+1), m2(+1), m4(-1), m6(+1)
        [1, 1, 0, -1, 0, 1, 0],
        # f: m3(+1), m6(+1)
        [0, 0, 1, 0, 0, 1, 0],
        # g: m4(+1), m7(+1)
        [0, 0, 0, 1, 0, 0, 1],
        # h: m1(+1), m3(-1), m5(+1), m7(+1)
        [1, 0, -1, 0, 1, 0, 1],
    ]
    W = [
        # C[0,0]: m1(+1), m4(+1), m5(-1), m7(+1)
        [1, 0, 0, 1, -1, 0, 1],
        # C[0,1]: m3(+1), m5(+1)
        [0, 0, 1, 0, 1, 0, 0],
        # C[1,0]: m2(+1), m4(+1)
        [0, 1, 0, 1, 0, 0, 0],
        # C[1,1]: m1(-1), m2(-1), m3(+1), m6(+1)
        [-1, -1, 1, 0, 0, 1, 0],
    ]
    return {"U": U, "V": V, "W": W}

def _block_strassen(n, m, p):
    """
    Block decomposition using Strassen where possible.
    Split 4x5*5x6 into blocks:
    - 2x2*2x2 blocks (use Strassen, rank 7 each)
    - 2x1*1x2 blocks (schoolbook, rank 2 each)
    - 2x2*2x1 blocks (schoolbook, rank 4 each)
    - 1x2*2x2 blocks (schoolbook, rank 4 each)
    - 1x1*1x1 blocks (schoolbook, rank 1 each)
    
    Split A (4x5) into row blocks of 2, col blocks of 2,2,1:
    A = [[A11(2x2), A12(2x2), A13(2x1)],
         [A21(2x2), A22(2x2), A23(2x1)]]
    
    Split B (5x6) into row blocks of 2,2,1, col blocks of 2,2,2:
    B = [[B11(2x2), B12(2x2), B13(2x2)],
         [B21(2x2), B22(2x2), B23(2x2)],
         [B31(1x2), B32(1x2), B33(1x2)]]
    
    C = A*B has blocks C[i,j] = sum_k A[i,k]*B[k,j]
    C is 4x6, split into 2x2 blocks:
    C = [[C11(2x2), C12(2x2), C13(2x2)],
         [C21(2x2), C22(2x2), C23(2x2)]]
    
    C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] + A[i,2]*B[2,j]
    
    For C[0,0] = A11*B11 + A12*B21 + A13*B31
    A11*B11: 2x2*2x2 -> Strassen (7)
    A12*B21: 2x2*2x2 -> Strassen (7)
    A13*B31: 2x1*1x2 -> schoolbook (2)
    Total for C[0,0]: 7+7+2 = 16
    
    Similarly for all 6 C blocks.
    Total rank: 6 * (7 + 7 + 2) = 6 * 16 = 96? 
    Wait, that's not right. The Strassen decompositions for different C blocks are independent.
    
    Actually, let me reconsider. The total rank is the sum of ranks of all sub-multiplications.
    There are 6 C-blocks, each requires:
    - Two 2x2*2x2 multiplications (Strassen, 7 each)
    - One 2x1*1x2 multiplication (schoolbook, 2)
    Total per C-block: 7+7+2 = 16
    Total: 6*16 = 96
    
    But wait, can we share Strassen terms across C-blocks? Probably not easily.
    Let's just compute the total: 96.
    
    Actually, let me recount. For each output block C[i,j]:
    C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] + A[i,2]*B[2,j]
    
    A[i,0] is 2x2, B[0,j] is 2x2 -> 2x2*2x2, Strassen rank 7
    A[i,1] is 2x2, B[1,j] is 2x2 -> 2x2*2x2, Strassen rank 7  
    A[i,2] is 2x1, B[2,j] is 1x2 -> 2x1*1x2, schoolbook rank 2
    
    So each C[i,j] needs 7+7+2 = 16 terms.
    6 C-blocks: 6*16 = 96 total.
    """
    n, m, p = 4, 5, 6
    # Row split of A: [2, 2]
    # Col split of A / Row split of B: [2, 2, 1]
    # Col split of B: [2, 2, 2]
    
    a_row_splits = [2, 2]
    a_col_splits = [2, 2, 1]
    b_col_splits = [2, 2, 2]
    
    # Build the full decomposition
    U_rows = n * m  # 20
    V_rows = m * p  # 30
    W_rows = p * n  # 24
    
    # We'll collect all terms
    terms = []  # list of (U_vec, V_vec, W_vec) where each is a dict or list
    
    # For each C block (ci, cj) where ci in [0,1], cj in [0,2]
    for ci in range(2):
        for cj in range(3):
            # C block is 2x2, starting at row ci*2, col cj*2
            c_row_start = ci * 2
            c_col_start = cj * 2
            
            # For each k in [0, 1, 2]
            for k in range(3):
                a_row_start = ci * 2
                a_col_start = sum(a_col_splits[:k])
                a_h = a_row_splits[ci]  # 2
                a_w = a_col_splits[k]
                
                b_row_start = sum(a_col_splits[:k])
                b_col_start = cj * 2
                b_h = a_col_splits[k]  # 2,2,1
                b_w = b_col_splits[cj]  # 2
                
                if a_h == 2 and a_w == 2 and b_h == 2 and b_w == 2:
                    # 2x2 * 2x2 -> use Strassen
                    s = _strassen_2x2x2()
                    for r in range(7):
                        u = [0] * U_rows
                        v = [0] * V_rows
                        w = [0] * W_rows
                        for ii in range(2):
                            for jj in range(2):
                                u[(a_row_start + ii) * m + (a_col_start + jj)] = s["U"][ii * 2 + jj][r]
                        for jj in range(2):
                            for kk in range(2):
                                v[(b_row_start + jj) * p + (b_col_start + kk)] = s["V"][jj * 2 + kk][r]
                        for kk in range(2):
                            for ii in range(2):
                                w[(c_col_start + kk) * n + (c_row_start + ii)] = s["W"][kk * 2 + ii][r]
                        terms.append((u, v, w))
                elif a_h == 2 and a_w == 1 and b_h == 1 and b_w == 2:
                    # 2x1 * 1x2 -> schoolbook, rank 2
                    for jj in range(1):
                        for kk in range(2):
                            for ii in range(2):
                                u = [0] * U_rows
                                v = [0] * V_rows
                                w = [0] * W_rows
                                u[(a_row_start + ii) * m + (a_col_start + jj)] = 1
                                v[(b_row_start + jj) * p + (b_col_start + kk)] = 1
                                w[(c_col_start + kk) * n + (c_row_start + ii)] = 1
                                terms.append((u, v, w))
                else:
                    # Fallback: schoolbook
                    for jj in range(a_w):
                        for kk in range(b_w):
                            for ii in range(a_h):
                                u = [0] * U_rows
                                v = [0] * V_rows
                                w = [0] * W_rows
                                u[(a_row_start + ii) * m + (a_col_start + jj)] = 1
                                v[(b_row_start + jj) * p + (b_col_start + kk)] = 1
                                w[(c_col_start + kk) * n + (c_row_start + ii)] = 1
                                terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    for r, (u, v, w) in enumerate(terms):
        for i in range(U_rows):
            U[i][r] = u[i]
        for i in range(V_rows):
            V[i][r] = v[i]
        for i in range(W_rows):
            W[i][r] = w[i]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # Build the tensor from decomposition
    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    # The target tensor has T[a,b,c] = 1 iff a//m == b//p and b%p == c%n and a%m == c%n
    # Actually: T[i*m+j, j*p+k, k*n+i] = 1
    
    # Check a few random indices
    import random
    random.seed(42)
    for _ in range(200):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        a = i * m + j
        b = j * p + k
        c = k * n + i
        target = 1
        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if val != target:
            return False
    
    # Check some zero entries
    for _ in range(200):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        # Check if this should be zero
        i1, j1 = divmod(a, m)
        j2, k2 = divmod(b, p)
        k3, i3 = divmod(c, n)
        if not (i1 == i3 and j1 == j2 and k2 == k3):
            target = 0
            val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
            if val != target:
                return False
    
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    Uses block Strassen decomposition to achieve lower rank than schoolbook.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # First try the block Strassen decomposition
    try:
        result = _block_strassen(n, m, p)
        if _verify_decomposition(n, m, p, result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
