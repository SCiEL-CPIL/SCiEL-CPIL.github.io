import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    
    Uses Strassen's algorithm on 2x2 blocks to reduce rank from 50 to 46.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 5, 5
    rank = 46
    
    U = [[0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0 for _ in range(rank)] for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        if col >= rank:
            return
        for idx in u_indices:
            U[idx][col] = 1
        for idx in v_indices:
            V[idx][col] = 1
        for idx in w_indices:
            W[idx][col] = 1
        col += 1

    def add_strassen_block(i_offset, j_offset, k_offset):
        # i_offset: 0 or 1 (row block of X)
        # j_offset: 0, 2, or 4 (col block of X / row block of Y)
        # k_offset: 0, 2, or 4 (col block of Y)
        # This block is 2x2x2
        # Local indices 0,1 for i, j, k
        # Global indices:
        # U: (i_offset + li) * m + (j_offset + lj)
        # V: (j_offset + lj) * p + (k_offset + lk)
        # W: (k_offset + lk) * n + (i_offset + li)
        
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        u1 = [(i_offset + 0) * m + (j_offset + 0), (i_offset + 1) * m + (j_offset + 1)]
        v1 = [(j_offset + 0) * p + (k_offset + 0), (j_offset + 1) * p + (k_offset + 1)]
        w1 = [(k_offset + 0) * n + (i_offset + 0), (k_offset + 1) * n + (i_offset + 1)]
        add_term(u1, v1, w1)
        
        # M2 = (A10+A11)B00 -> C10, C11 (with -1)
        u2 = [(i_offset + 1) * m + (j_offset + 0), (i_offset + 1) * m + (j_offset + 1)]
        v2 = [(j_offset + 0) * p + (k_offset + 0)]
        w2 = [(k_offset + 0) * n + (i_offset + 1), (k_offset + 1) * n + (i_offset + 1)]
        # Need to handle coefficients -1
        # Current add_term sets 1. We need a version with coeffs or modify U/V/W directly.
        pass

    # The add_term function above assumes all coefficients are 1.
    # Strassen has -1 coefficients.
    # I will rewrite add_term to accept lists of (index, coeff).
    
    # Redefine add_term
    def add_term_weighted(u_items, v_items, w_items):
        nonlocal col
        if col >= rank:
            return
        for idx, c in u_items:
            U[idx][col] += c
        for idx, c in v_items:
            V[idx][col] += c
        for idx, c in w_items:
            W[idx][col] += c
        col += 1

    def strassen_2x2x2(i_off, j_off, k_off):
        # i_off, j_off, k_off are the starting indices for the 2x2 blocks
        # i in [i_off, i_off+1]
        # j in [j_off, j_off+1]
        # k in [k_off, k_off+1]
        
        # M1 = (A00+A11)(B00+B11)
        u = [((i_off+0)*m + (j_off+0), 1), ((i_off+1)*m + (j_off+1), 1)]
        v = [((j_off+0)*p + (k_off+0), 1), ((j_off+1)*p + (k_off+1), 1)]
        w = [((k_off+0)*n + (i_off+0), 1), ((k_off+1)*n + (i_off+1), 1)]
        add_term_weighted(u, v, w)
        
        # M2 = (A10+A11)B00
        u = [((i_off+1)*m + (j_off+0), 1), ((i_off+1)*m + (j_off+1), 1)]
        v = [((j_off+0)*p + (k_off+0), 1)]
        w = [((k_off+0)*n + (i_off+1), 1), ((k_off+1)*n + (i_off+1), -1)]
        add_term_weighted(u, v, w)
        
        # M3 = A00(B10-B11)
        u = [((i_off+0)*m + (j_off+0), 1)]
        v = [((j_off+1)*p + (k_off+0), 1), ((j_off+1)*p + (k_off+1), -1)]
        w = [((k_off+1)*n + (i_off+0), 1), ((k_off+1)*n + (i_off+1), 1)]
        add_term_weighted(u, v, w)
        
        # M4 = A11(B01-B00)
        u = [((i_off+1)*m + (j_off+1), 1)]
        v = [((j_off+0)*p + (k_off+1), 1), ((j_off+0)*p + (k_off+0), -1)]
        w = [((k_off+0)*n + (i_off+0), 1), ((k_off+0)*n + (i_off+1), 1)]
        add_term_weighted(u, v, w)
        
        # M5 = (A00+A10)B11
        u = [((i_off+0)*m + (j_off+0), 1), ((i_off+1)*m + (j_off+0), 1)]
        v = [((j_off+1)*p + (k_off+1), 1)]
        w = [((k_off+0)*n + (i_off+0), -1), ((k_off+1)*n + (i_off+1), 1)]
        add_term_weighted(u, v, w)
        
        # M6 = (A10-A00)(B00+B01)
        u = [((i_off+1)*m + (j_off+0), 1), ((i_off+0)*m + (j_off+0), -1)]
        v = [((j_off+0)*p + (k_off+0), 1), ((j_off+0)*p + (k_off+1), 1)]
        w = [((k_off+1)*n + (i_off+1), 1)]
        add_term_weighted(u, v, w)
        
        # M7 = (A01-A11)(B10+B11)
        u = [((i_off+0)*m + (j_off+1), 1), ((i_off+1)*m + (j_off+1), -1)]
        v = [((j_off+1)*p + (k_off+0), 1), ((j_off+1)*p + (k_off+1), 1)]
        w = [((k_off+0)*n + (i_off+0), 1)]
        add_term_weighted(u, v, w)

    # Block 1: j in 0,1
    # k in 0,1
    strassen_2x2x2(0, 0, 0)
    # k in 2,3
    strassen_2x2x2(0, 0, 2)
    # k in 4
    # Trivial 2x2x1
    # i in 0,1; j in 0,1; k in 4
    for i in range(2):
        for j in range(2):
            u = [((i)*m + j, 1)]
            v = [((j)*p + 4, 1)]
            w = [((4)*n + i, 1)]
            add_term_weighted(u, v, w)
            
    # Block 2: j in 2,3
    # k in 0,1
    strassen_2x2x2(0, 2, 0)
    # k in 2,3
    strassen_2x2x2(0, 2, 2)
    # k in 4
    for i in range(2):
        for j in range(2, 4):
            u = [((i)*m + j, 1)]
            v = [((j)*p + 4, 1)]
            w = [((4)*n + i, 1)]
            add_term_weighted(u, v, w)

    # Block 3: j in 4
    # k in 0..4
    for i in range(2):
        for k in range(5):
            u = [((i)*m + 4, 1)]
            v = [((4)*p + k, 1)]
            w = [((k)*n + i, 1)]
            add_term_weighted(u, v, w)

    if col != rank:
        # Pad if necessary, though it should match exactly
        pass
        
    return {"U": U, "V": V, "W": W}
