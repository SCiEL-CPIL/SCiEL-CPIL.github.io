import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    terms = []
    
    def add_term(u_sparsity, v_sparsity, w_sparsity):
        """
        u_sparsity: list of (index, value)
        v_sparsity: list of (index, value)
        w_sparsity: list of (index, value)
        """
        terms.append((u_sparsity, v_sparsity, w_sparsity))
        
    def add_simple_term(i, k, j, coeff=1.0):
        """
        Adds a term for A[i,k] * B[k,j] -> C[i,j]
        u_idx = i*m + k
        v_idx = k*p + j
        w_idx = j*n + i
        """
        u_sp = [(i * m + k, coeff)]
        v_sp = [(k * p + j, coeff)]
        w_sp = [(j * n + i, coeff)]
        add_term(u_sp, v_sp, w_sp)
        
    # Part 1: 4x4x4 multiplication (k in 0..3, j in 0..3) using Strassen
    # Strassen products for 2x2 blocks
    # A blocks: 00, 01, 10, 11 (indices 0, 1, 2, 3)
    # B blocks: 00, 01, 10, 11 (indices 0, 1, 2, 3)
    
    # Global index helpers
    def a_idx(block, li, lk):
        # block 0: A00 (i=0,1; k=0,1)
        # block 1: A01 (i=0,1; k=2,3)
        # block 2: A10 (i=2,3; k=0,1)
        # block 3: A11 (i=2,3; k=2,3)
        if block == 0: return li * m + lk
        if block == 1: return li * m + (lk + 2)
        if block == 2: return (li + 2) * m + lk
        if block == 3: return (li + 2) * m + (lk + 2)
        
    def b_idx(block, lk, lj):
        # block 0: B00 (k=0,1; j=0,1)
        # block 1: B01 (k=0,1; j=2,3)
        # block 2: B10 (k=2,3; j=0,1)
        # block 3: B11 (k=2,3; j=2,3)
        if block == 0: return lk * p + lj
        if block == 1: return lk * p + (lj + 2)
        if block == 2: return (lk + 2) * p + lj
        if block == 3: return (lk + 2) * p + (lj + 2)
        
    def c_idx(block, li, lj):
        # block 0: C00 (i=0,1; j=0,1)
        # block 1: C01 (i=0,1; j=2,3)
        # block 2: C10 (i=2,3; j=0,1)
        # block 3: C11 (i=2,3; j=2,3)
        if block == 0: return lj * n + li
        if block == 1: return (lj + 2) * n + li
        if block == 2: return lj * n + (li + 2)
        if block == 3: return (lj + 2) * n + (li + 2)

    # Strassen definitions
    # Each product is a list of (a_block, a_sign), (b_block, b_sign), (c_block, c_sign)
    # We iterate over the 7 products.
    
    # M1 = (A00+A11)(B00+B11) -> C00(+1), C11(+1)
    # M2 = (A10+A11)B00 -> C10(+1), C11(-1)
    # M3 = A00(B01-B11) -> C01(+1), C11(+1)
    # M4 = A11(B10-B00) -> C00(+1), C10(+1)
    # M5 = (A00+A01)B11 -> C00(-1), C01(+1)
    # M6 = (A10-A00)(B00+B01) -> C11(+1)
    # M7 = (A01-A11)(B10+B11) -> C00(+1)
    
    strassen_defs = [
        # (A_parts, B_parts, C_parts)
        # A_parts: list of (block, sign)
        # B_parts: list of (block, sign)
        # C_parts: list of (block, sign)
        
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    for a_parts, b_parts, c_parts in strassen_defs:
        # For each local index in the 2x2 block
        for li in range(2):
            for lk in range(2):
                for lj in range(2):
                    u_sp = []
                    v_sp = []
                    w_sp = []
                    
                    for block, sign in a_parts:
                        idx = a_idx(block, li, lk)
                        u_sp.append((idx, sign))
                        
                    for block, sign in b_parts:
                        idx = b_idx(block, lk, lj)
                        v_sp.append((idx, sign))
                        
                    for block, sign in c_parts:
                        idx = c_idx(block, li, lj)
                        w_sp.append((idx, sign))
                        
                    add_term(u_sp, v_sp, w_sp)
                    
                    if time.time() - _START > _DEADLINE - 10:
                        break
                if time.time() - _START > _DEADLINE - 10:
                    break
            if time.time() - _START > _DEADLINE - 10:
                break
        if time.time() - _START > _DEADLINE - 10:
            break
            
    # Part 2: 4x4x1 (k in 0..3, j=4) - schoolbook
    # A[i,k] * B[k,4] -> C[i,4]
    for i in range(4):
        for k in range(4):
            if time.time() - _START > _DEADLINE - 5:
                break
            add_simple_term(i, k, 4, 1.0)
        if time.time() - _START > _DEADLINE - 5:
            break
            
    # Part 3: 4x1x5 (k=4, j in 0..4) - schoolbook
    # A[i,4] * B[4,j] -> C[i,j]
    for i in range(4):
        for j in range(5):
            if time.time() - _START > _DEADLINE - 5:
                break
            add_simple_term(i, 4, j, 1.0)
        if time.time() - _START > _DEADLINE - 5:
            break
            
    rank = len(terms)
    
    if rank == 0:
        add_simple_term(0, 0, 0, 1.0)
        rank = 1
        
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u_sp, v_sp, w_sp) in enumerate(terms):
        for idx, val in u_sp:
            U[idx, r] = val
        for idx, val in v_sp:
            V[idx, r] = val
        for idx, val in w_sp:
            W[idx, r] = val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
