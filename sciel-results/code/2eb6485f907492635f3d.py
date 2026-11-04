import numpy as np
import time

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
    
    # Strassen 2x2x2 rank-1 terms (u, v, w)
    # u indexed by [i*2+j], v by [j*2+k], w by [k*2+i]
    strassen_2x2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),      # M1=(A00+A11)(B00+B11)
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),       # M2=(A10+A11)B00
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),      # M3=A00(B01-B11)
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),      # M4=A11(B10-B00)
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),      # M5=(A00+A01)B11
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),      # M6=(A10-A00)(B00+B01)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),      # M7=(A01-A11)(B10+B11)
    ]
    
    def tensor_product_vec(v1, v2):
        """Tensor product: v1[a1*2+b1], v2[a2*2+b2] -> result[(a1*2+a2)*4+(b1*2+b2)]"""
        result = [0] * 16
        for a1 in range(2):
            for b1 in range(2):
                for a2 in range(2):
                    for b2 in range(2):
                        idx = (a1*2+a2)*4 + (b1*2+b2)
                        result[idx] = v1[a1*2+b1] * v2[a2*2+b2]
        return result
    
    # Generate 4x4x4 Strassen decomposition (49 terms)
    strassen_4x4x4 = []
    for t1 in strassen_2x2x2:
        for t2 in strassen_2x2x2:
            u = tensor_product_vec(t1[0], t2[0])
            v = tensor_product_vec(t1[1], t2[1])
            w = tensor_product_vec(t1[2], t2[2])
            strassen_4x4x4.append((u, v, w))
    
    # Build full decomposition: 49 (Strassen 4x4x4) + 36 (schoolbook remainder) = 85
    rank = 49 + 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Add Strassen 4x4x4 terms (i,j,k in {0,1,2,3})
    for r_idx, (u, v, w) in enumerate(strassen_4x4x4):
        for idx in range(16):
            U[idx][r_idx] = u[idx]
        for idx in range(16):
            V[idx][r_idx] = v[idx]
        for idx in range(16):
            W[idx][r_idx] = w[idx]
    
    # Add schoolbook terms for remaining (i,j,k) not all in {0,1,2,3}
    r = 49
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 4 and j < 4 and k < 4:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
    
    # Verify
    if verify(U, V, W):
        return {'U': U, 'V': V, 'W': W}
    
    # Fallback to schoolbook
    return {'U': U_school, 'V': V_school, 'W': W_school}
