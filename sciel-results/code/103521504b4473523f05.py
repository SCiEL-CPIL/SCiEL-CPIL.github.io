"""Rewritten program for 6x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    def verify_decomp(U, V, W, n, m, p):
        rank = len(U[0])
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        U_reshaped = U_np.reshape(n, m, rank)
        V_reshaped = V_np.reshape(m, p, rank)
        W_reshaped = W_np.reshape(p, n, rank)
        
        T = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(n):
            for c in range(m):
                for d in range(p):
                    i = r * m + c
                    j = c * p + d
                    k = d * n + r
                    target[i, j, k] = 1.0
        
        return np.allclose(T, target, atol=1e-9)

    # Correct Laderman 3x3 algorithm (23 terms)
    # Each term: (u, v, w) where u, v, w are 3x3 matrices (flat, row-major)
    # Convention: sum_t u_t[i,j] * v_t[j,k] * w_t[k,i] = 1 for all i,j,k in 0..2
    laderman_terms = [
        # Term 1: a11*b11
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 2: a12*b21
        ([0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # Term 3: a13*b31
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        # Term 4: a21*b12
        ([0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 5: a22*b22
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # Term 6: a23*b32
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0]),
        # Term 7: a31*b13
        ([0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 8: a32*b23
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,1, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # Term 9: a33*b33
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0]),
        # Term 10: (a11+a22-a33)*(b12+b23-b31)
        ([1,0,0, 0,1,0, 0,0,-1], [0,1,0, 0,0,1, 0,0,0], [0,0,1, 0,0,0, 1,0,0]),
        # Term 11: (a12+a23-a31)*(b21+b32-b13)
        ([0,1,0, 0,0,1, -1,0,0], [0,0,1, 0,0,0, 1,0,0], [0,1,0, 0,0,0, 0,0,1]),
        # Term 12: (a13+a31-a22)*(b31+b12-b23)
        ([0,0,1, 0,0,0, 1,0,0], [0,0,0, 0,1,0, 0,0,1], [0,0,1, 1,0,0, 0,0,0]),
        # Term 13: (a21+a33-a11)*(b13+b32-b21)
        ([0,0,0, 1,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 1,0,0]),
        # Term 14: (a32+a11-a23)*(b23+b13-b31)
        ([0,0,0, 0,0,-1, 0,1,0], [0,0,1, 0,0,0, 0,0,1], [0,0,1, 1,0,0, 0,0,0]),
        # Term 15: (a11+a23-a32)*(b32+b12-b21)
        ([1,0,0, 0,0,1, 0,0,0], [0,0,0, 0,1,0, 1,0,0], [0,0,1, 1,0,0, 0,0,0]),
        # Term 16: (a22+a31-a13)*(b13+b31-b12)
        ([0,0,0, 0,1,0, 0,0,1], [0,0,1, 0,0,0, 0,0,1], [0,0,1, 1,0,0, 0,0,0]),
        # Term 17: (a33+a12-a21)*(b21+b12-b13)
        ([0,0,0, 1,0,0, 0,1,0], [0,0,1, 0,1,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0]),
        # Term 18: (a11+a33)*(b13+b32)
        ([1,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,1,0, 0,0,0]),
        # Term 19: (a22+a11)*(b21+b13)
        ([1,0,0, 0,1,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0]),
        # Term 20: (a33+a22)*(b31+b23)
        ([0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0], [0,1,0, 0,0,1, 0,0,0]),
        # Term 21: (a12+a21)*(b12+b21)
        ([0,1,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0]),
        # Term 22: (a13+a31)*(b13+b31)
        ([0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,1, 1,0,0, 0,0,0]),
        # Term 23: (a23+a32)*(b23+b32)
        ([0,0,0, 0,0,1, 0,1,0], [0,0,0, 0,0,1, 0,0,0], [0,0,1, 0,1,0, 0,0,0]),
    ]

    def verify_laderman(terms):
        u_np = np.array([t[0] for t in terms], dtype=np.float64).reshape(len(terms), 3, 3)
        v_np = np.array([t[1] for t in terms], dtype=np.float64).reshape(len(terms), 3, 3)
        w_np = np.array([t[2] for t in terms], dtype=np.float64).reshape(len(terms), 3, 3)
        
        T = np.einsum('tij,tjk,tki->ijk', u_np, v_np, w_np)
        
        return np.allclose(T, 1.0, atol=1e-9)

    laderman_ok = verify_laderman(laderman_terms)

    if laderman_ok and time.time() < deadline:
        # Construct 6x6 decomposition using Strassen(2x2) of Laderman(3x3)
        strassen_2x2 = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        rank = 7 * 23
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for u_coeffs, v_coeffs, w_coeffs in strassen_2x2:
            for t in range(23):
                u_mat = np.array(laderman_terms[t][0]).reshape(3,3)
                v_mat = np.array(laderman_terms[t][1]).reshape(3,3)
                w_mat = np.array(laderman_terms[t][2]).reshape(3,3)
                
                for (bi, bj), u_val in u_coeffs.items():
                    for si in range(3):
                        for sj in range(3):
                            gr = bi * 3 + si
                            gc = bj * 3 + sj
                            U[gr * m + gc][col_idx] += u_val * u_mat[si, sj]
                            
                for (bk, bl), v_val in v_coeffs.items():
                    for sk in range(3):
                        for sl in range(3):
                            gr = bk * 3 + sk
                            gc = bl * 3 + sl
                            V[gr * p + gc][col_idx] += v_val * v_mat[sk, sl]
                            
                for (bi, bk), w_val in w_coeffs.items():
                    for si in range(3):
                        for sk in range(3):
                            i = bi * 3 + si
                            k = bk * 3 + sk
                            W_flat_idx = k * n + i
                            W[W_flat_idx][col_idx] += w_val * w_mat[sk, si]
                            
                col_idx += 1
        
        if verify_decomp(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    # Fallback: Rank 189 (3x3 blocks of Strassen 2x2)
    block_size = 2
    blocks_per_dim = 3
    
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_2x2:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
