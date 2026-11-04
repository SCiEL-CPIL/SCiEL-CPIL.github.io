import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # We will construct a decomposition with rank < rank_schoolbook
    # Using 12 blocks of 2x2x2 Strassen and 24 schoolbook terms for j=4
    
    # Total rank estimate: 12 * 7 + 24 = 108
    # We will build U, V, W dynamically
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        """
        u_vec: list of (i, j, coeff)
        v_vec: list of (j, k, coeff)
        w_vec: list of (k, i, coeff)
        """
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    # Strassen coefficients for 2x2x2
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: {(0,0):1, (1,1):1}, v: {(0,0):1, (1,1):1}, w: {(0,0):1, (1,1):1}
        ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]),
        # M2: u: {(1,0):1, (1,1):1}, v: {(0,0):1}, w: {(1,0):1, (0,0):-1} -> C10 += M2, C00 -= M2
        ([((1,0,1), (1,1,1))], [((0,0,1))], [((1,0,1), (0,0,-1))]),
        # M3: u: {(0,0):1}, v: {(1,0):1, (1,1):-1}, w: {(0,1):1, (1,1):-1} -> C01 += M3, C11 -= M3
        ([((0,0,1))], [((1,0,1), (1,1,-1))], [((0,1,1), (1,1,-1))]),
        # M4: u: {(1,1):1}, v: {(0,1):1, (0,0):-1}, w: {(0,0):1, (1,0):-1} -> C00 += M4, C10 -= M4
        ([((1,1,1))], [((0,1,1), (0,0,-1))], [((0,0,1), (1,0,-1))]),
        # M5: u: {(0,0):1, (0,1):1}, v: {(1,1):1}, w: {(1,1):1} -> C11 += M5
        ([((0,0,1), (0,1,1))], [((1,1,1))], [((1,1,1))]),
        # M6: u: {(1,0):1, (0,0):-1}, v: {(0,0):1, (0,1):1}, w: {(1,0):1, (1,1):-1} -> C10 += M6, C11 -= M6
        ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))], [((1,0,1), (1,1,-1))]),
        # M7: u: {(0,1):1, (1,1):-1}, v: {(1,0):1, (1,1):1}, w: {(0,1):1, (0,0):-1} -> C01 += M7, C00 -= M7
        ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))], [((0,1,1), (0,0,-1))]),
    ]
    
    # Define blocks
    # I pairs: (0,1), (2,3)
    # J pairs: (0,1), (2,3)
    # K pairs: (0,1), (2,3), (4,5)
    
    i_pairs = [(0, 1), (2, 3)]
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for i0, i1 in i_pairs:
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                for u_t, v_t, w_t in strassen_terms:
                    u_vec = []
                    v_vec = []
                    w_vec = []
                    for (li, lj, coeff) in u_t:
                        gi = i0 + li
                        gj = j0 + lj
                        u_vec.append((gi, gj, coeff))
                    for (lj, lk, coeff) in v_t:
                        gj = j0 + lj
                        gk = k0 + lk
                        v_vec.append((gj, gk, coeff))
                    for (lk, li, coeff) in w_t:
                        gk = k0 + lk
                        gi = i0 + li
                        w_vec.append((gk, gi, coeff))
                    add_term(u_vec, v_vec, w_vec)
                    
    # Remaining terms: j=4
    # i in 0..3, k in 0..5
    for i in range(n):
        for k in range(p):
            add_term([((i, 4, 1))], [((4, k, 1))], [((k, i, 1))])
            
    # Convert to matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    rank = len(U)
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for (i, j, coeff) in U[c]:
            U_mat[i * m + j][c] = coeff
        for (j, k, coeff) in V[c]:
            V_mat[j * p + k][c] = coeff
        for (k, i, coeff) in W[c]:
            W_mat[k * n + i][c] = coeff
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}

if __name__ == "__main__":
    print(construct())
