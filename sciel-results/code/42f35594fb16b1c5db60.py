"""
Optimized rank-17 decomposition for 2x3 x 3x3 matrix multiplication.
Uses Strassen for the 2x2x2 core and schoolbook for the remaining terms.
"""

def construct():
    """
    Returns a rank-17 exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x3 matrix.
    
    Interface:
    U: shape (n*m, rank) = (6, 17)
    V: shape (m*p, rank) = (9, 17)
    W: shape (p*n, rank) = (6, 17)
    
    n=2, m=3, p=3.
    rank = 17.
    """
    import numpy as np
    
    n, m, p = 2, 3, 3
    rank = 17
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define the terms for the decomposition.
    # Each term is a tuple: (u_indices, v_indices, w_indices, coefficient)
    # u_indices: list of (index_in_U, value)
    # v_indices: list of (index_in_V, value)
    # w_indices: list of (index_in_W, value)
    
    terms = []
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0(A00), 1(A01), 3(A10), 4(A11)
    # B indices: 0(B00), 1(B01), 3(B10), 4(B11)
    # C indices: 0(C00), 1(C10), 2(C01), 3(C11)
    
    # M1: (A00+A11)(B00+B11) -> C00 + C11
    terms.append(([(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (3, 1)], 1))
    
    # M2: (A10+A11)B00 -> C10 - C11
    terms.append(([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1))
    
    # M3: A00(B01-B11) -> C01 + C11
    terms.append(([(0, 1)], [(1, 1), (4, -1)], [(2, 1), (3, 1)], 1))
    
    # M4: A11(B10-B00) -> C00 + C10
    terms.append(([(4, 1)], [(3, 1), (0, -1)], [(0, 1), (1, 1)], 1))
    
    # M5: (A00+A01)B11 -> -C00 + C01
    terms.append(([(0, 1), (1, 1)], [(4, 1)], [(0, -1), (2, 1)], 1))
    
    # M6: (A10-A00)(B00+B01) -> C11
    terms.append(([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 1))
    
    # M7: (A01-A11)(B10+B11) -> C00
    terms.append(([(1, 1), (4, -1)], [(3, 1), (4, 1)], [(0, 1)], 1))
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A indices: 2(A02), 5(A12)
    # B indices: 6(B20), 7(B21)
    # C indices: 0(C00), 1(C10), 2(C01), 3(C11)
    
    # r=7: A02 * B20 -> C00
    terms.append(([(2, 1)], [(6, 1)], [(0, 1)], 1))
    
    # r=8: A02 * B21 -> C01
    terms.append(([(2, 1)], [(7, 1)], [(2, 1)], 1))
    
    # r=9: A12 * B20 -> C10
    terms.append(([(5, 1)], [(6, 1)], [(1, 1)], 1))
    
    # r=10: A12 * B21 -> C11
    terms.append(([(5, 1)], [(7, 1)], [(3, 1)], 1))
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    # A indices: 0,1,2,3,4,5
    # B indices: 2(B02), 5(B12), 8(B22)
    # C indices: 4(C02), 5(C12)
    
    # r=11: A00 * B02 -> C02
    terms.append(([(0, 1)], [(2, 1)], [(4, 1)], 1))
    
    # r=12: A01 * B12 -> C02
    terms.append(([(1, 1)], [(5, 1)], [(4, 1)], 1))
    
    # r=13: A02 * B22 -> C02
    terms.append(([(2, 1)], [(8, 1)], [(4, 1)], 1))
    
    # r=14: A10 * B02 -> C12
    terms.append(([(3, 1)], [(2, 1)], [(5, 1)], 1))
    
    # r=15: A11 * B12 -> C12
    terms.append(([(4, 1)], [(5, 1)], [(5, 1)], 1))
    
    # r=16: A12 * B22 -> C12
    terms.append(([(5, 1)], [(8, 1)], [(5, 1)], 1))
    
    # Populate matrices from terms
    for r, (u_terms, v_terms, w_terms, coeff) in enumerate(terms):
        for idx, val in u_terms:
            U[idx, r] += val * coeff
        for idx, val in v_terms:
            V[idx, r] += val * coeff
        for idx, val in w_terms:
            W[idx, r] += val * coeff
            
    # Convert to lists for return
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
