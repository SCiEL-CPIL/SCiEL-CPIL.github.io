"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_block(a_indices, b_indices, c_indices):
    """
    Generate U, V, W contributions for a Strassen-like product.
    a_indices: list of (row, col) pairs in A (n x m) that are added/subtracted
    b_indices: list of (row, col) pairs in B (m x p) that are added/subtracted
    c_indices: list of (row, col) pairs in C (n x p) with signs
    Returns: (u_dict, v_dict, w_dict) mapping flat indices to coefficients
    """
    n, m, p = DIMENSIONS
    u = {}
    v = {}
    w = {}
    for sign, (ar, ac) in a_indices:
        u[(ar * m + ac)] = sign
    for sign, (br, bc) in b_indices:
        v[(br * p + bc)] = sign
    for sign, (cr, cc) in c_indices:
        # W index: k*n + i where k=col of C, i=row of C
        w[(cc * n + cr)] = sign
    return u, v, w

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # n=2, m=2, p=4
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 14)
    V = [[0] * rank for _ in range(m * p)]  # (8, 14)
    W = [[0] * rank for _ in range(p * n)]  # (8, 14)
    
    # Define Strassen factors for left half (columns 0,1) and right half (columns 2,3)
    # Each factor is (a_indices, b_indices, c_indices)
    # a_indices: [(sign, (row, col)), ...] in A (2x2)
    # b_indices: [(sign, (row, col)), ...] in B (2x4)
    # c_indices: [(sign, (row, col)), ...] in C (2x4)
    
    factors = [
        # Left half (k=0,1)
        # M1 = (A00+A11)(B00+B11) -> C00+=, C11+=
        ([(1,(0,0)), (1,(1,1))], [(1,(0,0)), (1,(1,1))], [(1,(0,0)), (1,(1,1))]),
        # M2 = (A10+A11)B00 -> C10+=, C11-=
        ([(1,(1,0)), (1,(1,1))], [(1,(0,0))], [(1,(1,0)), (-1,(1,1))]),
        # M3 = A00(B01-B11) -> C01+=, C11+=
        ([(1,(0,0))], [(1,(0,1)), (-1,(1,1))], [(1,(0,1)), (1,(1,1))]),
        # M4 = A11(B10-B00) -> C00+=, C10+=
        ([(1,(1,1))], [(1,(1,0)), (-1,(0,0))], [(1,(0,0)), (1,(1,0))]),
        # M5 = (A00+A01)B11 -> C00-=, C01+=
        ([(1,(0,0)), (1,(0,1))], [(1,(1,1))], [(-1,(0,0)), (1,(0,1))]),
        # M6 = (A10-A00)(B00+B01) -> C11+=
        ([(1,(1,0)), (-1,(0,0))], [(1,(0,0)), (1,(0,1))], [(1,(1,1))]),
        # M7 = (A01-A11)(B10+B11) -> C00+=
        ([(1,(0,1)), (-1,(1,1))], [(1,(1,0)), (1,(1,1))], [(1,(0,0))]),
        # Right half (k=2,3)
        # M8 = (A00+A11)(B02+B13) -> C02+=, C13+=
        ([(1,(0,0)), (1,(1,1))], [(1,(0,2)), (1,(1,3))], [(1,(0,2)), (1,(1,3))]),
        # M9 = (A10+A11)B02 -> C12+=, C13-=
        ([(1,(1,0)), (1,(1,1))], [(1,(0,2))], [(1,(1,2)), (-1,(1,3))]),
        # M10 = A00(B03-B13) -> C03+=, C13+=
        ([(1,(0,0))], [(1,(0,3)), (-1,(1,3))], [(1,(0,3)), (1,(1,3))]),
        # M11 = A11(B12-B02) -> C02+=, C12+=
        ([(1,(1,1))], [(1,(1,2)), (-1,(0,2))], [(1,(0,2)), (1,(1,2))]),
        # M12 = (A00+A01)B13 -> C02-=, C03+=
        ([(1,(0,0)), (1,(0,1))], [(1,(1,3))], [(-1,(0,2)), (1,(0,3))]),
        # M13 = (A10-A00)(B02+B03) -> C13+=
        ([(1,(1,0)), (-1,(0,0))], [(1,(0,2)), (1,(0,3))], [(1,(1,3))]),
        # M14 = (A01-A11)(B12+B13) -> C02+=
        ([(1,(0,1)), (-1,(1,1))], [(1,(1,2)), (1,(1,3))], [(1,(0,2))]),
    ]
    
    for c, (a_idx, b_idx, c_idx) in enumerate(factors):
        if time.time() > deadline:
            break
        for (ar, ac), sign in a_idx:
            U[ar * m + ac][c] += sign
        for (br, bc), sign in b_idx:
            V[br * p + bc][c] += sign
        for (cr, cc), sign in c_idx:
            W[cc * n + cr][c] += sign
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
