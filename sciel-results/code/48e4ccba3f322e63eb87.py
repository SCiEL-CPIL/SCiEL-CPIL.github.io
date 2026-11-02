import numpy as np
import time

def construct():
    """
    Returns a rank-14 exact tensor decomposition for 2x2 by 2x4 matrix multiplication.
    U: (n*m, rank) = (4, 14)
    V: (m*p, rank) = (8, 14)
    W: (p*n, rank) = (8, 14)
    
    Uses Strassen's 7-term algorithm for each 2x2 block.
    The rank of M_{2,2,4} is known to be 14.
    """
    n, m, p = 2, 2, 4
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 matrix multiplication
    # Indices for A (U): 0=A00, 1=A01, 2=A10, 3=A11
    # Indices for B (V): 0=B00, 1=B01, 2=B02, 3=B03, 4=B10, 5=B11, 6=B12, 7=B13
    # Indices for C (W): 0=C00, 1=C10, 2=C01, 3=C11, 4=C02, 5=C12, 6=C03, 7=C13
    # Note: W index is k*n + i.
    # k=0, i=0 -> 0 (C00)
    # k=0, i=1 -> 1 (C10)
    # k=1, i=0 -> 2 (C01)
    # k=1, i=1 -> 3 (C11)
    
    # Define the 7 Strassen terms for a local 2x2x2 block.
    # Each term is (u_indices, v_indices, w_indices) where indices are local 0-3.
    # Local V indices: 0=B00, 1=B01, 2=B10, 3=B11
    # Local W indices: 0=C00, 1=C10, 2=C01, 3=C11
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10+A11)B00
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # M3 = A00(B01-B11)
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        # M4 = A11(B10-B00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # M5 = (A00+A01)B11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        # M6 = (A10-A00)(B00+B01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (A01-A11)(B10+B11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # Block 1: B cols 0,1 -> C cols 0,1
    # Global V indices: j*p + k. j in {0,1}, k in {0,1}
    # Global W indices: k*n + i. k in {0,1}, i in {0,1}
    
    for r_local in range(7):
        r_global = r_local
        u_terms, v_terms, w_terms = strassen_terms[r_local]
        
        for idx, val in u_terms:
            U[idx][r_global] += val
            
        for idx, val in v_terms:
            j = idx // 2
            k = idx % 2
            global_v_idx = j * p + k
            V[global_v_idx][r_global] += val
            
        for idx, val in w_terms:
            k = idx // 2
            i = idx % 2
            global_w_idx = k * n + i
            W[global_w_idx][r_global] += val

    # Block 2: B cols 2,3 -> C cols 2,3
    # Global V indices: j*p + k. j in {0,1}, k in {2,3}
    # Global W indices: k*n + i. k in {2,3}, i in {0,1}
    
    for r_local in range(7):
        r_global = 7 + r_local
        u_terms, v_terms, w_terms = strassen_terms[r_local]
        
        for idx, val in u_terms:
            U[idx][r_global] += val
            
        for idx, val in v_terms:
            j = idx // 2
            k_local = idx % 2
            k_global = k_local + 2
            global_v_idx = j * p + k_global
            V[global_v_idx][r_global] += val
            
        for idx, val in w_terms:
            k_local = idx // 2
            i = idx % 2
            k_global = k_local + 2
            global_w_idx = k_global * n + i
            W[global_w_idx][r_global] += val

    return {
        "U": U,
        "V": V,
        "W": W
    }
