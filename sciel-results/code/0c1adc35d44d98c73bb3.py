import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def validate_exact(U, V, W, n, m, p):
    """Validate with exact fraction arithmetic."""
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = Fraction(0, 1)
                for c in range(r):
                    val += Fraction(U[idx_u][c]) * Fraction(V[idx_v][c]) * Fraction(W[idx_w][c])
                if val != Fraction(1, 1):
                    return False
    return True

def schoolbook_decomposition(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def build_strassen_decomposition(n, m, p):
    """
    Build a low-rank decomposition for 2x5x5 using Strassen blocks.
    
    Strategy:
    - Use Strassen (7 terms) for 2x2x2 blocks where all three dimensions have size 2
    - Use schoolbook (4 terms) for 2x2x1 or 1x2x2 or 2x1x2 blocks
    - Use schoolbook (2 terms) for 2x1x1 or 1x2x1 or 1x1x2 blocks
    - Use schoolbook (1 term) for 1x1x1 blocks
    
    Partition:
    i in {0,1} (n=2)
    j in {0,1,2,3,4} (m=5)
    k in {0,1,2,3,4} (p=5)
    
    Blocks:
    (j={0,1}, k={0,1}): Strassen 2x2x2 -> 7 terms
    (j={0,1}, k={2,3}): Strassen 2x2x2 -> 7 terms
    (j={0,1}, k={4}):    schoolbook 2x2x1 -> 4 terms
    (j={2,3}, k={0,1}): Strassen 2x2x2 -> 7 terms
    (j={2,3}, k={2,3}): Strassen 2x2x2 -> 7 terms
    (j={2,3}, k={4}):    schoolbook 2x2x1 -> 4 terms
    (j={4}, k={0,1}):    schoolbook 2x1x2 -> 4 terms
    (j={4}, k={2,3}):    schoolbook 2x1x2 -> 4 terms
    (j={4}, k={4}):      schoolbook 2x1x1 -> 2 terms
    
    Total: 7+7+4+7+7+4+4+4+2 = 46 terms
    """
    terms = []
    
    def add_strassen_2x2x2(i0, j0, k0):
        """Add 7 Strassen terms for 2x2x2 block starting at (i0, j0, k0)."""
        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1]) * (C[k0,i0]+C[k0+1,i0+1])
        u = {(i0, j0): 1, (i0+1, j0+1): 1}
        v = {(j0, k0): 1, (j0+1, k0+1): 1}
        w = {(k0, i0): 1, (k0+1, i0+1): 1}
        terms.append((u, v, w))
        
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0] * (C[k0,i0+1]-C[k0+1,i0+1])
        u = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        v = {(j0, k0): 1}
        w = {(k0, i0+1): 1, (k0+1, i0+1): -1}
        terms.append((u, v, w))
        
        # M3 = A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1]) * (C[k0+1,i0]+C[k0+1,i0+1])
        u = {(i0, j0): 1}
        v = {(j0, k0+1): 1, (j0+1, k0+1): -1}
        w = {(k0+1, i0): 1, (k0+1, i0+1): 1}
        terms.append((u, v, w))
        
        # M4 = A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0]) * (C[k0,i0]+C[k0,i0+1])
        u = {(i0+1, j0+1): 1}
        v = {(j0+1, k0): 1, (j0, k0): -1}
        w = {(k0, i0): 1, (k0, i0+1): 1}
        terms.append((u, v, w))
        
        # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1] * (-C[k0,i0]+C[k0+1,i0])
        u = {(i0, j0): 1, (i0, j0+1): 1}
        v = {(j0+1, k0+1): 1}
        w = {(k0, i0): -1, (k0+1, i0): 1}
        terms.append((u, v, w))
        
        # M6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1]) * C[k0+1,i0+1]
        u = {(i0+1, j0): 1, (i0, j0): -1}
        v = {(j0, k0): 1, (j0, k0+1): 1}
        w = {(k0+1, i0+1): 1}
        terms.append((u, v, w))
        
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1]) * C[k0,i0]
        u = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        v = {(j0+1, k0): 1, (j0+1, k0+1): 1}
        w = {(k0, i0): 1}
        terms.append((u, v, w))
    
    def add_schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end):
        """Add schoolbook terms for a block."""
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))
    
    # Apply Strassen where possible, schoolbook otherwise
    # i is always 0..1 (n=2)
    
    # j={0,1}, k={0,1}: Strassen
    add_strassen_2x2x2(0, 0, 0)
    
    # j={0,1}, k={2,3}: Strassen
    add_strassen_2x2x2(0, 0, 2)
    
    # j={0,1}, k={4}: schoolbook
    add_schoolbook_block(0, 2, 0, 2, 4, 5)
    
    # j={2,3}, k={0,1}: Strassen
    add_strassen_2x2x2(0, 2, 0)
    
    # j={2,3}, k={2,3}: Strassen
    add_strassen_2x2x2(0, 2, 2)
    
    # j={2,3}, k={4}: schoolbook
    add_schoolbook_block(0, 2, 2, 4, 4, 5)
    
    # j={4}, k={0,1}: schoolbook
    add_schoolbook_block(0, 2, 4, 5, 0, 2)
    
    # j={4}, k={2,3}: schoolbook
    add_schoolbook_block(0, 2, 4, 5, 2, 4)
    
    # j={4}, k={4}: schoolbook
    add_schoolbook_block(0, 2, 4, 5, 4, 5)
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try Strassen-based decomposition first
    U, V, W = build_strassen_decomposition(n, m, p)
    rank = len(W[0])
    
    # Validate
    if validate_exact(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
