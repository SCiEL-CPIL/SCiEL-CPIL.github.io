import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    deadline = start_time + 115

    def verify(rank, U, V, W):
        """Check T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1 for all i,j,k."""
        # Vectorized check
        # U: (16, rank), V: (20, rank), W: (20, rank)
        # We need to check for all i in 0..3, j in 0..3, k in 0..4
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m + j, :]
                    v_val = V[j*p + k, :]
                    w_val = W[k*n + i, :]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, c] = 1.0
                    V[j*p + k, c] = 1.0
                    W[k*n + i, c] = 1.0
                    c += 1
        return {'U': U, 'V': V, 'W': W}

    def make_strassen_4x4x5():
        rank = 65
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        # Scalar Strassen factors for 2x2
        # (a_idx, a_coeff), (b_idx, b_coeff), (c_idx, c_coeff)
        # Indices are (row, col)
        scalar_factors = [
            # m1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # m2
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            # m3
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            # m4
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            # m5
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            # m6
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            # m7
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        ]
        
        # Block Strassen factors for 4x4
        # (a_blocks, b_blocks, c_blocks)
        # Block indices are (block_row, block_col)
        block_factors = [
            # S1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # S2
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            # S3
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            # S4
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            # S5
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            # S6
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            # S7
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        ]
        
        for t in range(7):
            for s in range(7):
                r = t * 7 + s
                ba, bb, bc = block_factors[t]
                sa, sb, sc = scalar_factors[s]
                
                for (abr, abc, ca) in ba:
                    for (ar, ac, cs) in sa:
                        i = 2 * abr + ar
                        j = 2 * abc + ac
                        U[i*m + j, r] += ca * cs
                        
                for (bbr, bbc, cb) in bb:
                    for (br, bc, cs) in sb:
                        j = 2 * bbr + br
                        k = 2 * bbc + bc
                        V[j*p + k, r] += cb * cs
                        
                for (cbr, cbc, cc) in bc:
                    for (cr, cc2, cs2) in sc:
                        k = 2 * cbr + cr
                        i = 2 * cbc + cc2
                        W[k*n + i, r] += cc * cs2
        
        # 5th column schoolbook
        for i in range(n):
            for j in range(m):
                r = 49 + i * m + j
                U[i*m + j, r] = 1.0
                V[j*p + 4, r] = 1.0
                W[4*n + i, r] = 1.0
                
        return {'U': U, 'V': V, 'W': W}

    best = make_strassen_4x4x5()
    
    if verify(65, best['U'], best['V'], best['W']):
        return best
    else:
        # Fallback
        return make_schoolbook()

# EVOLVE-BLOCK-END
