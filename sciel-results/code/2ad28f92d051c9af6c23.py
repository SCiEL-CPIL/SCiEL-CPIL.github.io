import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s margin
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Rank calculation
    # 2 blocks of m (size 2 each)
    # 4 blocks of p (size 2 each)
    # 7 terms per block multiplication
    rank = 2 * 4 * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen vectors for 2x2 multiplication
    # Indices: i, u (0..1) for U; u, v (0..1) for V; v, i (0..1) for W
    # Note: W index in global is k*n+i. In local, k is v, i is i.
    # Local W index corresponds to (v, i).
    
    # U vectors (length 4, index 2*i + u)
    # V vectors (length 4, index 2*u + v)
    # W vectors (length 4, index 2*v + i)
    
    U_vecs = [
        [1, 0, 0, 1], # M1: A00, A11
        [0, 0, 1, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00, A01
        [1, 0, -1, 0],# M6: A00, A10 (1, -1) -> idx 0, 2
        [0, 1, 0, -1] # M7: A01, A11 (1, -1) -> idx 1, 3
    ]
    
    V_vecs = [
        [1, 0, 0, 1], # M1: B00, B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01, B11
        [-1, 0, 1, 0],# M4: B00, B10 (-1, 1) -> idx 0, 2
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00, B01
        [0, 0, 1, 1]  # M7: B10, B11
    ]
    
    W_vecs = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 1, 0, -1],# M2: C10, C11
        [0, 1, 1, 0], # M3: C01, C11 (Wait, M3: C01+=1, C11+=1) -> v=1,i=0 (idx 2), v=1,i=1 (idx 3)
        [-1, 1, 0, 0],# M4: C00, C10 (C00-=1, C10+=1) -> v=0,i=0 (idx 0), v=0,i=1 (idx 1)
        [1, 1, 0, 0], # M5: C00, C01 (C00+=1, C01+=1) -> v=0,i=0 (idx 0), v=1,i=0 (idx 2)? No.
        # Let's re-verify W vectors carefully.
        
        # W index is 2*v + i.
        # v=0, i=0 -> 0
        # v=0, i=1 -> 1
        # v=1, i=0 -> 2
        # v=1, i=1 -> 3
        
        # M1: C00(0,0)+=1, C11(1,1)+=1. W=[1, 0, 0, 1]
        # M2: C10(0,1)+=1, C11(1,1)-=1. W=[0, 1, 0, -1]
        # M3: C01(1,0)+=1, C11(1,1)+=1. W=[0, 0, 1, 1]
        # M4: C00(0,0)-=1, C10(0,1)+=1. W=[-1, 1, 0, 0]
        # M5: C00(0,0)+=1, C01(1,0)+=1. W=[1, 0, 1, 0]
        # M6: C11(1,1)+=1. W=[0, 0, 0, 1]
        # M7: C00(0,0)-=1, C11(1,1)-=1. W=[-1, 0, 0, -1]
    ]
    
    W_vecs = [
        [1, 0, 0, 1], 
        [0, 1, 0, -1],
        [0, 0, 1, 1], 
        [-1, 1, 0, 0],
        [1, 0, 1, 0], 
        [0, 0, 0, 1], 
        [-1, 0, 0, -1]
    ]

    for s in range(2):
        for ell in range(4):
            for r in range(7):
                if time.time() > deadline:
                    break
                if col >= rank:
                    break
                    
                u_vec = U_vecs[r]
                v_vec = V_vecs[r]
                w_vec = W_vecs[r]
                
                for i in range(2):
                    for u in range(2):
                        if u_vec[2 * i + u] != 0:
                            # U index: i * m + j = i * 4 + (2 * s + u)
                            u_idx = i * 4 + 2 * s + u
                            U[u_idx][col] = u_vec[2 * i + u]
                            
                for u in range(2):
                    for v in range(2):
                        if v_vec[2 * u + v] != 0:
                            # V index: j * p + k = (2 * s + u) * 8 + (2 * ell + v)
                            v_idx = (2 * s + u) * 8 + 2 * ell + v
                            V[v_idx][col] = v_vec[2 * u + v]
                            
                for v in range(2):
                    for i in range(2):
                        if w_vec[2 * v + i] != 0:
                            # W index: k * n + i = (2 * ell + v) * 2 + i
                            w_idx = (2 * ell + v) * 2 + i
                            W[w_idx][col] = w_vec[2 * v + i]
                            
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
