"""Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 6]
DEADLINE = 120.0

def _verify(U, V, W, n, m, p, tol=1e-12):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And check it equals delta_{i,k} (1 if i==k else 0)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                expected = 1.0 if i == k else 0.0
                if abs(s - expected) > tol:
                    return False
    return True

def _build_strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2.
    U[2*2, 7], V[2*2, 7], W[2*2, 7] with W[k*2+i, c]"""
    n = m = p = 2
    # Strassen's 7 products
    # U rows: (i,j) pairs for A
    # V rows: (j,k) pairs for B  
    # W rows: (k,i) pairs for C
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # U[i*m+j, c]: coefficient for A_{i,j} in product c
    # V[j*p+k, c]: coefficient for B_{j,k} in product c
    # W[k*n+i, c]: coefficient for C_{k,i} in product c (note: W[k*n+i] per convention)
    
    # Product 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # Product 1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10
    W[3][1] = -1 # C11
    
    # Product 2: M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # Product 3: M4 = A11(B01-B10)
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    W[0][3] = -1 # C00
    W[1][3] = 1  # C01
    
    # Product 4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1  # C00
    W[2][4] = 1  # C10
    
    # Product 5: M6 = (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1  # C11
    
    # Product 6: M7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = -1 # C00
    
    return U, V, W

def _combine_decompositions(decs, dims):
    """Combine multiple block decompositions.
    decs: list of (U, V, W) for each block
    dims: list of (n_i, m_i, p_i) for each block
    """
    n = sum(d[0] for d in dims)
    m = sum(d[1] for d in dims)
    p = sum(d[2] for d in dims)
    rank = sum(len(U[0]) for U, V, W in decs)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    offset_c = 0
    offset_n = 0
    offset_m = 0
    offset_p = 0
    for (U_b, V_b, W_b), (n_i, m_i, p_i) in zip(decs, dims):
        rank_b = len(U_b[0])
        for i in range(n_i):
            for j in range(m_i):
                for c in range(rank_b):
                    U[(offset_n+i)*m + (offset_m+j)][offset_c + c] = U_b[i*m_i+j][c]
        for j in range(m_i):
            for k in range(p_i):
                for c in range(rank_b):
                    V[(offset_m+j)*p + (offset_p+k)][offset_c + c] = V_b[j*p_i+k][c]
        for k in range(p_i):
            for i in range(n_i):
                for c in range(rank_b):
                    W[(offset_p+k)*n + (offset_n+i)][offset_c + c] = W_b[k*n_i+i][c]
        offset_c += rank_b
        offset_n += n_i
        offset_m += m_i
        offset_p += p_i
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Strategy: Try different block decompositions
    # 4x4x6 can be split as:
    # Option 1: Two 4x4x3 blocks
    # Option 2: One 4x4x4 + one 4x4x2
    # Option 3: Four 2x2x3 blocks
    # Option 4: Two 2x4x3 blocks
    # Option 5: Use Strassen-like for 4x4x4 and extend
    
    # First, let's try to find a good decomposition for 4x4x4
    # Known: rank 23 for 4x4x4 (Bayer, Strassen, etc.)
    # For simplicity, let's start with known constructions
    
    # Try: 4x4x4 using two 2x2x2 Strassen + cross terms
    # This is complex. Let's try a simpler approach first.
    
    # Approach: Use the fact that we can do 4x4x4 with rank <= 23
    # and 4x4x2 with some rank, then combine.
    
    # For now, let's try a few structured approaches:
    
    # Approach A: 4x4x4 (rank 23) + 4x4x2 (rank ?)
    # Approach B: 4x4x3 + 4x4x3
    # Approach C: Direct search for low-rank
    
    # Let's first try to construct a 4x4x3 decomposition
    # 4x4x3: n=4, m=4, p=3
    # Schoolbook: 48
    # Can we do better?
    
    # For 4x4x3, we can try:
    # Split into 2x2x3 blocks (4 blocks)
    # Each 2x2x3 can be done with rank... let's think
    # 2x2x3: A is 2x2, B is 2x3, C is 2x3
    # T[i,j,k] = delta_{i,k} for i in {0,1}, j in {0,1}, k in {0,1,2}
    # But wait, k goes up to 2, and i only up to 1, so delta_{i,k} is 1 only if i==k and k<=1
    # So for k=2, all entries are 0
    # This means the 2x2x3 tensor has zeros in the k=2 slice
    
    # Actually for 2x2x3: n=2, m=2, p=3
    # T[i,j,k] = 1 if i==k else 0, for i in {0,1}, j in {0,1}, k in {0,1,2}
    # Since i max is 1 and k goes to 2, T[i,j,2] = 0 for all i,j
    # So effectively it's like 2x2x2 with an extra zero dimension
    # Rank of 2x2x3 = rank of 2x2x2 = 7 (Strassen)
    
    # So 4x4x3 as four 2x2x3 blocks: rank = 4*7 = 28
    # Then 4x4x6 as two 4x4x3: rank = 2*28 = 56
    
    # Can we do better? Let's try 4x4x4 + 4x4x2
    # 4x4x4: best known is 23
    # 4x4x2: n=4, m=4, p=2
    # T[i,j,k] = delta_{i,k}, i in {0..3}, j in {0..3}, k in {0,1}
    # For k=0: T[i,j,0] = 1 if i==0 else 0
    # For k=1: T[i,j,1] = 1 if i==1 else 0
    # This is a structured tensor
    
    # Let's just try the 4x4x3 + 4x4x3 approach first and verify
    
    # Build 2x2x3 Strassen-like
    def _strassen_2x2x3():
        """Rank-7 decomposition for 2x2 * 2x3.
        U[4,7], V[6,7], W[6,7] where W[k*2+i, c]"""
        n, m, p = 2, 2, 3
        U, V, W2 = _build_strassen_2x2()
        # U is [4,7], V is [4,7], W2 is [4,7] for 2x2x2
        # We need to extend to 2x2x3
        # The 2x2x3 tensor: T[i,j,k] = delta_{i,k} for k in {0,1,2}
        # For k=2: all zero
        # So we take the 2x2x2 decomposition and pad with zeros for k=2
        
        U_new = [[0]*7 for _ in range(4)]  # n*m = 4
        V_new = [[0]*7 for _ in range(6)]  # m*p = 6
        W_new = [[0]*7 for _ in range(6)]  # p*n = 6
        
        for i in range(2):
            for j in range(2):
                for c in range(7):
                    U_new[i*2+j][c] = U[i*2+j][c]
        
        for j in range(2):
            for k in range(3):
                if k < 2:
                    for c in range(7):
                        V_new[j*3+k][c] = V[j*2+k][c]
                else:
                    # k=2, all zero
                    pass
        
        for k in range(3):
            for i in range(2):
                if k < 2:
                    for c in range(7):
                        W_new[k*2+i][c] = W2[k*2+i][c]
                else:
                    # k=2, all zero
                    pass
        
        return U_new, V_new, W_new
    
    # Build 4x4x3 as four 2x2x3 blocks
    def _build_4x4x3():
        n, m, p = 4, 4, 3
        # Four 2x2x3 blocks
        decs = []
        dims = []
        for ni in range(2):
            for mi in range(2):
                U_b, V_b, W_b = _strassen_2x2x3()
                decs.append((U_b, V_b, W_b))
                dims.append((2, 2, 3))
        
        U, V, W = _combine_decompositions(decs, dims)
        return U, V, W
    
    # Try 4x4x3 + 4x4x3 for 4x4x6
    def _build_4x4x6_from_two_4x4x3():
        n, m, p = 4, 4, 6
        # Two 4x4x3 blocks
        U1, V1, W1 = _build_4x4x3()
        U2, V2, W2 = _build_4x4x3()
        
        decs = [(U1, V1, W1), (U2, V2, W2)]
        dims = [(4, 4, 3), (4, 4, 3)]
        
        U, V, W = _combine_decompositions(decs, dims)
        return U, V, W
    
    # Try this construction
    U, V, W = _build_4x4x6_from_two_4x4x3()
    rank = len(U[0])
    if _verify(U, V, W, n, m, p):
        best_rank = rank
        best_result = {"U": U, "V": V, "W": W}
    
    # Try other approaches if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30 and best_rank > 48:
        # Try to find better decompositions
        # For now, return what we have
        pass

    if best_result is None:
        # Fallback to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    return best_result
