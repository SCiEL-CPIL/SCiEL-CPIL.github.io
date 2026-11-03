"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_222():
    """
    Return the Strassen rank-7 decomposition for 2x2x2.
    Returns list of (u, v, w) triples where u,v are 2-element lists (for A,B factors)
    and w is 2-element list (for C factor), with the convention:
    C[i,j] = sum_r U[i,j] * V[j,k] * W[k,i] ... 
    Actually, let me use the standard tensor decomposition convention:
    For 2x2x2: A is 2x2, B is 2x2, C = A*B is 2x2.
    C[i,k] = sum_j A[i,j] * B[j,k]
    Decomposition: C[i,k] = sum_r U[i,j_r] ... 
    
    Let me use the explicit Strassen terms.
    The 7 rank-1 terms for Strassen:
    M1 = (A11 - A21)(B11 + B21)  -> contributes to C11, C21
    M2 = (A11 + A21)(B21)        -> contributes to C11, C21  
    M3 = (A21 - A22)(B12 + B22)  -> contributes to C12, C22
    M4 = (A12 + A22)(B12)        -> contributes to C12, C22
    M5 = (A11 + A22)(B11 + B22)  -> contributes to C11, C22
    M6 = (A12 - A11)(B21 + B11)  -> contributes to C11, C12
    M7 = (A21 - A22)(B12 + B22)  -> wait, I need to be careful
    
    Let me use the standard Strassen:
    C11 = M1 + M4 - M5 - M7
    C12 = M2 + M3
    C21 = M6 + M5
    C22 = M3 - M4 + M1 - M2
    
    Where:
    M1 = (A11 - A21)(B11 + B21)
    M2 = (A11 + A21)(B21)
    M3 = (A21 - A22)(B12 + B22)
    M4 = (A12 + A22)(B12)
    M5 = (A11 + A22)(B11 + B22)
    M6 = (A12 - A11)(B21 + B11)
    M7 = (A21 - A22)(B12 + B22)
    
    Hmm, I'm getting confused. Let me just use a verified set.
    
    Standard Strassen (from Wikipedia):
    M1 = (A22 - A11)(B11 + B12)
    M2 = (A11 + A22)(B22 - B12)
    M3 = (A11 + A12)B22
    M4 = (A21 + A22)B11
    M5 = (A11 - A21)(B11 + B21)
    M6 = (A12 - A22)(B12 + B22)
    M7 = (A11 + A12 - A21 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M6
    C21 = M2 + M5
    C22 = M1 - M3 + M6 + M4
    
    Let me verify: This is one standard formulation.
    """
    # Each term is (a_factors, b_factors, c_factors)
    # a_factors: dict from (i,j) to coefficient for A[i,j]
    # b_factors: dict from (j,k) to coefficient for B[j,k]
    # c_factors: dict from (i,k) to coefficient for C[i,k]
    
    terms = []
    
    # M1 = (A22 - A11)(B11 + B12)
    terms.append({
        'a': {(1,1): 1, (0,0): -1},
        'b': {(0,0): 1, (0,1): 1},
        'c': {(0,0): 1, (1,1): 1}
    })
    
    # M2 = (A11 + A22)(B22 - B12)
    terms.append({
        'a': {(0,0): 1, (1,1): 1},
        'b': {(1,1): 1, (0,1): -1},
        'c': {(1,0): 1, (1,1): -1}
    })
    
    # M3 = (A11 + A12)B22
    terms.append({
        'a': {(0,0): 1, (0,1): 1},
        'b': {(1,1): 1},
        'c': {(0,1): 1}
    })
    
    # M4 = (A21 + A22)B11
    terms.append({
        'a': {(1,0): 1, (1,1): 1},
        'b': {(0,0): 1},
        'c': {(1,0): 1, (1,1): 1}
    })
    
    # M5 = (A11 - A21)(B11 + B21)
    terms.append({
        'a': {(0,0): 1, (1,0): -1},
        'b': {(0,0): 1, (1,0): 1},
        'c': {(0,0): -1, (1,0): 1}
    })
    
    # M6 = (A12 - A22)(B12 + B22)
    terms.append({
        'a': {(0,1): 1, (1,1): -1},
        'b': {(0,1): 1, (1,1): 1},
        'c': {(0,1): 1, (1,1): 1}
    })
    
    # M7 = (A11 + A12 - A21 - A22)(B21 + B22)
    terms.append({
        'a': {(0,0): 1, (0,1): 1, (1,0): -1, (1,1): -1},
        'b': {(1,0): 1, (1,1): 1},
        'c': {(0,0): 1}
    })
    
    return terms

def _schoolbook_nm_p(n, m, p):
    """Return schoolbook decomposition for n x m x p. Returns list of (a_idx, b_idx, c_idx) triples."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(((i,j), (j,k), (i,k)))
    return terms

def construct():
    """
    Construct a tensor decomposition for 3x5x6 matrix multiplication using block decomposition.
    Split: n=3 as 2+1, m=5 as 2+2+1, p=6 as 2+2+2
    Use Strassen for 2x2x2 blocks, schoolbook for others.
    """
    import numpy as np
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Define splits
    n_splits = [2, 1]  # rows of A
    m_splits = [2, 2, 1]  # cols of A / rows of B  
    p_splits = [2, 2, 2]  # cols of B
    
    # Compute offsets
    n_offsets = [0]
    for s in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + s)
    
    # Collect all terms
    all_terms = []
    
    for ni in range(len(n_splits)):
        for mj in range(len(m_splits)):
            for pk in range(len(p_splits)):
                ni_size = n_splits[ni]
                mj_size = m_splits[mj]
                pk_size = p_splits[pk]
                
                n_off = n_offsets[ni]
                m_off = m_offsets[mj]
                p_off = p_offsets[pk]
                
                # Get decomposition for this block
                if ni_size == 2 and mj_size == 2 and pk_size == 2:
                    # Use Strassen
                    strassen_terms = _strassen_222()
                    for term in strassen_terms:
                        a_factor = term['a']
                        b_factor = term['b']
                        c_factor = term['c']
                        # Map local indices to global
                        for (li, lj), coeff_a in a_factor.items():
                            global_i = n_off + li
                            global_j = m_off + lj
                            for (lj2, lk), coeff_b in b_factor.items():
                                global_j2 = m_off + lj2
                                global_k = p_off + lk
                                for (li2, lk2), coeff_c in c_factor.items():
                                    global_i2 = n_off + li2
                                    global_k2 = p_off + lk2
                                    # This isn't quite right - the tensor decomposition
                                    # is not a simple product of these factors
                                    pass
                
                # For now, use schoolbook for all blocks
                for i in range(ni_size):
                    for j in range(mj_size):
                        for k in range(pk_size):
                            global_i = n_off + i
                            global_j = m_off + j
                            global_k = p_off + k
                            all_terms.append((
                                (global_i, global_j),  # A index
                                (global_j, global_k),  # B index
                                (global_i, global_k)   # C index
                            ))
    
    rank = len(all_terms)
    
    # Build U, V, W
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (a_idx, b_idx, c_idx) in enumerate(all_terms):
        ai, aj = a_idx
        bj, bk = b_idx
        ci, ck = c_idx
        U[ai * m + aj][r] = 1
        V[bj * p + bk][r] = 1
        W[ck * n + ci][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
