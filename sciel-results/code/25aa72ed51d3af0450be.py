"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        """
        u_entries: list of (i, j, val)
        v_entries: list of (j, k, val)
        w_entries: list of (k, i, val)
        """
        terms.append((u_entries, v_entries, w_entries))
        
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = [(i, j, 1)]
                    v = [(j, k, 1)]
                    w = [(k, i, 1)]
                    add_term(u, v, w)
                    
    def add_strassen(i_start, j_start, k_start):
        """
        Adds 7 terms for a 2x2x2 block starting at i_start, j_start, k_start.
        Assumes n=2, so i_start must be 0 and block covers i=0,1.
        j_start covers j_start, j_start+1.
        k_start covers k_start, k_start+1.
        """
        # Indices
        i0, i1 = i_start, i_start + 1
        j0, j1 = j_start, j_start + 1
        k0, k1 = k_start, k_start + 1
        
        # M1 = (A00 + A11) * (B00 + B11)
        u1 = [(i0, j0, 1), (i1, j1, 1)]
        v1 = [(j0, k0, 1), (j1, k1, 1)]
        w1 = [(k0, i0, 1), (k1, i1, 1)]
        add_term(u1, v1, w1)
        
        # M2 = (A10 + A11) * B00
        u2 = [(i1, j0, 1), (i1, j1, 1)]
        v2 = [(j0, k0, 1)]
        w2 = [(k0, i1, 1), (k1, i1, -1)]
        add_term(u2, v2, w2)
        
        # M3 = A00 * (B01 - B11)
        u3 = [(i0, j0, 1)]
        v3 = [(j0, k1, 1), (j1, k1, -1)]
        w3 = [(k1, i0, 1), (k1, i1, 1)]
        add_term(u3, v3, w3)
        
        # M4 = A11 * (B10 - B00)
        u4 = [(i1, j1, 1)]
        v4 = [(j1, k0, 1), (j0, k0, -1)]
        w4 = [(k0, i0, 1), (k0, i1, 1)]
        add_term(u4, v4, w4)
        
        # M5 = (A00 + A01) * B11
        u5 = [(i0, j0, 1), (i0, j1, 1)]
        v5 = [(j1, k1, 1)]
        w5 = [(k0, i0, -1), (k1, i0, 1)]
        add_term(u5, v5, w5)
        
        # M6 = (A10 - A00) * (B00 + B01)
        u6 = [(i1, j0, 1), (i0, j0, -1)]
        v6 = [(j0, k0, 1), (j0, k1, 1)]
        w6 = [(k1, i1, 1)]
        add_term(u6, v6, w6)
        
        # M7 = (A01 - A11) * (B10 + B11)
        u7 = [(i0, j1, 1), (i1, j1, -1)]
        v7 = [(j1, k0, 1), (j1, k1, 1)]
        w7 = [(k0, i0, 1)]
        add_term(u7, v7, w7)
        
    # Partitioning
    # n = 2 (indices 0, 1)
    # m = 5 (indices 0..4) -> M1: 0,1 ; M2: 2,3 ; M3: 4
    # p = 5 (indices 0..4) -> P1: 0,1 ; P2: 2,3 ; P3: 4
    
    # Block (M1, P1): 2x2x2 -> Strassen
    add_strassen(0, 0, 0)
    
    # Block (M1, P2): 2x2x2 -> Strassen
    add_strassen(0, 0, 2)
    
    # Block (M1, P3): 2x2x1 -> Schoolbook
    add_schoolbook(0, 2, 0, 2, 4, 5)
    
    # Block (M2, P1): 2x2x2 -> Strassen
    add_strassen(0, 2, 0)
    
    # Block (M2, P2): 2x2x2 -> Strassen
    add_strassen(0, 2, 2)
    
    # Block (M2, P3): 2x2x1 -> Schoolbook
    add_schoolbook(0, 2, 2, 4, 4, 5)
    
    # Block (M3, P1): 2x1x2 -> Schoolbook
    add_schoolbook(0, 2, 4, 5, 0, 2)
    
    # Block (M3, P2): 2x1x2 -> Schoolbook
    add_schoolbook(0, 2, 4, 5, 2, 4)
    
    # Block (M3, P3): 2x1x1 -> Schoolbook
    add_schoolbook(0, 2, 4, 5, 4, 5)
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_entries, v_entries, w_entries = terms[c]
        for (i, j, val) in u_entries:
            U[i * m + j][c] = val
        for (j, k, val) in v_entries:
            V[j * p + k][c] = val
        for (k, i, val) in w_entries:
            W[k * n + i][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
