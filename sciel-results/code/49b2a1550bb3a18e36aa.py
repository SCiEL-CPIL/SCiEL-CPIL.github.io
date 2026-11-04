import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 140.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Construct rank-77 tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy:
    1. 4x4 * 4x4 part (cols 0-3 of B): Rank 49 using Kronecker product of two Strassen(2x2x2) decompositions.
    2. 4x4 * 4x2 part (cols 4-5 of B): Rank 28 using four Strassen(2x2x2) decompositions on 2x2 blocks.
    Total Rank: 49 + 28 = 77.
    """
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add one rank-1 term."""
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            if coeff != 0:
                u_col[i * m + j] = coeff
        for (j, k), coeff in v_dict.items():
            if coeff != 0:
                v_col[j * p + k] = coeff
        for (i, k), coeff in w_dict.items():
            if coeff != 0:
                w_col[k * n + i] = coeff
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    # Define Strassen terms for 2x2x2
    # Returns list of 7 tuples (u_dict, v_dict, w_dict)
    def get_strassen_2x2_terms(oi, oj, ok):
        terms = []
        # m1: (a00+a11)*(b00+b11)
        terms.append((
            {(oi, oj): 1, (oi+1, oj+1): 1},
            {(oj, ok): 1, (oj+1, ok+1): 1},
            {(oi, ok): 1, (oi+1, ok+1): 1}
        ))
        # m2: (a10+a11)*b00
        terms.append((
            {(oi+1, oj): 1, (oi+1, oj+1): 1},
            {(oj, ok): 1},
            {(oi+1, ok): 1, (oi+1, ok+1): -1}
        ))
        # m3: a00*(b01-b11)
        terms.append((
            {(oi, oj): 1},
            {(oj, ok+1): 1, (oj+1, ok+1): -1},
            {(oi, ok+1): 1, (oi+1, ok+1): 1}
        ))
        # m4: a11*(b10-b00)
        terms.append((
            {(oi+1, oj+1): 1},
            {(oj+1, ok): 1, (oj, ok): -1},
            {(oi, ok): 1, (oi+1, ok): 1}
        ))
        # m5: (a00+a01)*b11
        terms.append((
            {(oi, oj): 1, (oi, oj+1): 1},
            {(oj+1, ok+1): 1},
            {(oi, ok): -1, (oi, ok+1): 1}
        ))
        # m6: (a10-a00)*(b00+b01)
        terms.append((
            {(oi+1, oj): 1, (oi, oj): -1},
            {(oj, ok): 1, (oj, ok+1): 1},
            {(oi+1, ok+1): 1}
        ))
        # m7: (a01-a11)*(b10+b11)
        terms.append((
            {(oi, oj+1): 1, (oi+1, oj+1): -1},
            {(oj+1, ok): 1, (oj+1, ok+1): 1},
            {(oi, ok): 1}
        ))
        return terms

    # PART 1: 4x4 * 4x4 (columns 0-3 of B) -> rank 49
    if _time_remaining() > 5.0:
        base_terms = get_strassen_2x2_terms(0, 0, 0)
        
        for r in range(7):
            u_r, v_r, w_r = base_terms[r]
            for s in range(7):
                u_s, v_s, w_s = base_terms[s]
                
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                for (i, j), c1 in u_r.items():
                    for (k, l), c2 in u_s.items():
                        ri = 2*i + k
                        rj = 2*j + l
                        u_dict[(ri, rj)] = u_dict.get((ri, rj), 0) + c1 * c2
                        
                for (i, j), c1 in v_r.items():
                    for (k, l), c2 in v_s.items():
                        ri = 2*i + k
                        rj = 2*j + l
                        v_dict[(ri, rj)] = v_dict.get((ri, rj), 0) + c1 * c2
                        
                for (i, j), c1 in w_r.items():
                    for (k, l), c2 in w_s.items():
                        ri = 2*i + k
                        rj = 2*j + l
                        w_dict[(ri, rj)] = w_dict.get((ri, rj), 0) + c1 * c2
                        
                add_term(u_dict, v_dict, w_dict)

    # PART 2: 4x4 * 4x2 (columns 4-5 of B) -> rank 28
    if _time_remaining() > 5.0:
        for rb in range(2):
            for mb in range(2):
                a_oi = 2*rb
                a_oj = 2*mb
                b_ok = 4
                
                terms = get_strassen_2x2_terms(a_oi, a_oj, b_ok)
                for u_d, v_d, w_d in terms:
                    add_term(u_d, v_d, w_d)

    rank = len(U_cols)
    
    if rank == 0:
        # Fallback to schoolbook
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_col = [0] * (n * m)
                    v_col = [0] * (m * p)
                    w_col = [0] * (p * n)
                    u_col[i * m + j] = 1
                    v_col[j * p + k] = 1
                    w_col[k * n + i] = 1
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
        rank = len(U_cols)
    
    # Build matrices efficiently
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
