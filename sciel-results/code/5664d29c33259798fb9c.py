import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_school[u_idx][r] = 1
                V_school[v_idx][r] = 1
                W_school[w_idx][r] = 1
                r += 1
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r_idx in range(rank):
                        s += U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
                    if s != 1:
                        return False
        return True

    def try_block_decomp():
        # Decompose 4x5x5 into:
        # - 4x4x4 block (j<4, k<4): use Strassen 49 terms
        # - 4x1x4 block (j<4, k==4): schoolbook 16 terms
        # - 1x4x4 block (j==4, k<4): schoolbook 16 terms  
        # - 1x1x4 block (j==4, k==4): schoolbook 4 terms
        # Total: 49 + 16 + 16 + 4 = 85 terms
        
        # Strassen 2x2x2 terms
        strassen_2x2x2 = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
            ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
        ]
        
        def tensor_product_vec(v1, v2):
            result = [0] * 16
            for a1 in range(2):
                for b1 in range(2):
                    for a2 in range(2):
                        for b2 in range(2):
                            idx = (a1*2+a2)*4 + (b1*2+b2)
                            result[idx] = v1[a1*2+b1] * v2[a2*2+b2]
            return result
        
        strassen_4x4x4 = []
        for t1 in strassen_2x2x2:
            for t2 in strassen_2x2x2:
                u = tensor_product_vec(t1[0], t2[0])
                v = tensor_product_vec(t1[1], t2[1])
                w = tensor_product_vec(t1[2], t2[2])
                strassen_4x4x4.append((u, v, w))
        
        rank = 49 + 16 + 16 + 4
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Add Strassen 4x4x4 terms (handles j<4, k<4)
        for r_idx, (u, v, w) in enumerate(strassen_4x4x4):
            for i in range(4):
                for j in range(4):
                    U[i*5+j][r_idx] = u[i*4+j]
            for j in range(4):
                for k in range(4):
                    V[j*5+k][r_idx] = v[j*4+k]
            for k in range(4):
                for i in range(4):
                    W[k*4+i][r_idx] = w[k*4+i]
        
        # Add 4x1x4 block (j<4, k==4): schoolbook
        r_offset = 49
        for i in range(4):
            for j in range(4):
                U[i*5+j][r_offset] = 1
                V[j*5+4][r_offset] = 1
                W[4*4+i][r_offset] = 1
                r_offset += 1
        
        # Add 1x4x4 block (j==4, k<4): schoolbook
        for i in range(4):
            for k in range(4):
                U[i*5+4][r_offset] = 1
                V[4*5+k][r_offset] = 1
                W[k*4+i][r_offset] = 1
                r_offset += 1
        
        # Add 1x1x4 block (j==4, k==4): schoolbook
        for i in range(4):
            U[i*5+4][r_offset] = 1
            V[4*5+4][r_offset] = 1
            W[4*4+i][r_offset] = 1
            r_offset += 1
        
        if verify(U, V, W):
            return {'U': U, 'V': V, 'W': W}
        
        return None

    result = try_block_decomp()
    if result is not None:
        return result
    
    return {'U': U_school, 'V': V_school, 'W': W_school}
