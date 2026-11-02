import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    """
    start_time = time.time()
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Level 1: Block Strassen (2x2 blocks)
    # L_t, R_t, C_t are 2x2 matrices of coefficients
    L = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[0, 0], [1, -1]]),
        np.array([[0, 1], [0, -1]])
    ]
    R = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[0, -1], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[0, 0], [1, 1]])
    ]
    C = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, -1]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[1, 0], [1, 0]]),
        np.array([[-1, 1], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]])
    ]
    
    # Level 2: Scalar Strassen (2x2 scalars)
    # l_s, r_s, W_s are 2x2 matrices of coefficients
    # m_s = x_s * y_s
    # X = sum l_s[u,v] X[u,v]
    # Y = sum r_s[u,v] Y[u,v]
    l = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[0, 0], [1, -1]]),
        np.array([[0, 1], [0, -1]])
    ]
    r = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[0, -1], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[0, 0], [1, 1]])
    ]
    W_s = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, -1]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[1, 0], [1, 0]]),
        np.array([[-1, 1], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]])
    ]
    
    for t in range(7):
        for s in range(7):
            r_idx = 7 * t + s
            
            Lt = L[t]
            Rt = R[t]
            Ct = C[t]
            
            ls = l[s]
            rs = r[s]
            Wst = W_s[s]
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    coeff_l = Lt[i, j]
                    if coeff_l == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            coeff_s = ls[u, v]
                            if coeff_s == 0:
                                continue
                            val = coeff_l * coeff_s
                            if val != 0:
                                # U index: row = 2*i + u, col = 2*j + v
                                U[2*i + u, 2*j + v, r_idx] = val # Wait, U is 2D. U[row, col]
                                # Correction: U is (16, 49). U[row, rank]
                                U[2*i + u, 2*j + v] # No, U is 2D.
                                # U[row, rank_idx]
                                # row = 2*i + u, col = 2*j + v is for A?
                                # A is 4x4. U is (16, 49).
                                # U index is (i*4+k, r).
                                # i is row, k is col.
                                # So U[2*i+u, 2*j+v] is wrong.
                                # U is (16, 49).
                                # The first index is the flattened index of A.
                                # A[row, col] -> index row*4 + col.
                                # So U[row*4 + col, rank_idx]
                                
                                # Let's re-verify U shape.
                                # U shape: (n*m, rank) = (16, 49).
                                # n=4, m=4.
                                # A is 4x4.
                                # Index in U corresponds to (i, k) of A.
                                # i is row, k is col.
                                # So U[i*4 + k, r]
                                
                                # My loop:
                                # i, j are block indices for A.
                                # u, v are local indices.
                                # Global row = 2*i + u.
                                # Global col = 2*j + v.
                                # So U[ (2*i+u)*4 + (2*j+v), r_idx]
                                
                                U[(2*i + u) * 4 + (2*j + v), r_idx] = val
            
            # Fill V
            for k in range(2):
                for l_idx in range(2):
                    coeff_r = Rt[k, l_idx]
                    if coeff_r == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            coeff_s = rs[u, v]
                            if coeff_s == 0:
                                continue
                            val = coeff_r * coeff_s
                            if val != 0:
                                # V index: (k*4 + j, r) ?
                                # V shape: (m*p, rank) = (16, 49).
                                # B is 4x4.
                                # Index in V corresponds to (k, j) of B.
                                # k is row, j is col.
                                # Global row = 2*k + u.
                                # Global col = 2*l_idx + v.
                                # V[ (2*k+u)*4 + (2*l_idx+v), r_idx]
                                
                                V[(2*k + u) * 4 + (2*l_idx + v), r_idx] = val
            
            # Fill W
            for i in range(2):
                for j in range(2):
                    coeff_c = Ct[i, j]
                    if coeff_c == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            coeff_s = Wst[u, v]
                            if coeff_s == 0:
                                continue
                            val = coeff_c * coeff_s
                            if val != 0:
                                # W index: (j*n + i, r) ?
                                # W shape: (p*n, rank) = (16, 49).
                                # C is 4x4.
                                # Index in W corresponds to (j, i) of C?
                                # Convention: W[k*n + i, r]
                                # k is col, i is row.
                                # Global row = 2*i + u.
                                # Global col = 2*j + v.
                                # W[ (2*j+v)*4 + (2*i+u), r_idx]
                                
                                W[(2*j + v) * 4 + (2*i + u), r_idx] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
