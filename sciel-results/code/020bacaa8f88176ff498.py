import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    rank = 38
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(offset_u, offset_v, offset_w):
        nonlocal col
        if time.time() > deadline:
            return
            
        # M1
        U[offset_u + 0][col] = 1
        U[offset_u + 3][col] = 1
        V[offset_v + 0][col] = 1
        V[offset_v + 2][col] = 1
        W[offset_w + 0][col] = 1
        W[offset_w + 1][col] = -1
        W[offset_w + 2][col] = 1
        W[offset_w + 3][col] = -1
        col += 1
        
        # M2
        U[offset_u + 2][col] = 1
        U[offset_u + 3][col] = 1
        V[offset_v + 0][col] = 1
        W[offset_w + 2][col] = -1
        W[offset_w + 3][col] = 1
        col += 1
        
        # M3
        U[offset_u + 0][col] = 1
        V[offset_v + 2][col] = 1
        V[offset_v + 3][col] = -1
        W[offset_w + 1][col] = 1
        W[offset_w + 3][col] = -1
        col += 1
        
        # M4
        U[offset_u + 3][col] = 1
        V[offset_v + 1][col] = 1
        V[offset_v + 0][col] = -1
        W[offset_w + 0][col] = 1
        W[offset_w + 3][col] = 1
        col += 1
        
        # M5
        U[offset_u + 0][col] = 1
        U[offset_u + 1][col] = 1
        V[offset_v + 3][col] = 1
        W[offset_w + 0][col] = -1
        W[offset_w + 1][col] = 1
        col += 1
        
        # M6
        U[offset_u + 2][col] = 1
        U[offset_u + 0][col] = -1
        V[offset_v + 0][col] = 1
        V[offset_v + 1][col] = 1
        W[offset_w + 1][col] = 1
        W[offset_w + 2][col] = 1
        col += 1
        
        # M7
        U[offset_u + 1][col] = 1
        U[offset_u + 3][col] = -1
        V[offset_v + 2][col] = 1
        V[offset_v + 3][col] = 1
        W[offset_w + 0][col] = 1
        W[offset_w + 3][col] = -1
        col += 1
        
        # Schoolbook for cols 2,3,4
        for c in range(2, 5):
            v_idx1 = offset_v + c * 2
            v_idx2 = offset_v + c * 2 + 1
            w_idx1 = offset_w + c
            w_idx2 = offset_w + c + 1
            
            U[offset_u + 0][col] = 1
            V[v_idx1][col] = 1
            W[w_idx1][col] = 1
            col += 1
            
            U[offset_u + 1][col] = 1
            V[v_idx2][col] = 1
            W[w_idx1][col] = 1
            col += 1
            
            U[offset_u + 2][col] = 1
            V[v_idx1][col] = 1
            W[w_idx2][col] = 1
            col += 1
            
            U[offset_u + 3][col] = 1
            V[v_idx2][col] = 1
            W[w_idx2][col] = 1
            col += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(4, 10, 0)
    
    return {"U": U, "V": V, "W": W}
