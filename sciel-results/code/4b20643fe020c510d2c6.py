import time

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 5 blocks * 7 Strassen multiplications = 35
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (2x2): a=0, b=1, c=2, d=3
    # Indices for B block (2x2) relative to block b:
    # e = B[0, 2b] -> V index 2b
    # f = B[0, 2b+1] -> V index 2b+1
    # g = B[1, 2b] -> V index 20+2b
    # h = B[1, 2b+1] -> V index 20+2b+1
    # Indices for C block (2x2) relative to block b:
    # C00 = C[0, 2b] -> W index 4b
    # C01 = C[0, 2b+1] -> W index 4b+2
    # C10 = C[1, 2b] -> W index 4b+1
    # C11 = C[1, 2b+1] -> W index 4b+3
    
    for b in range(5):
        base_col = 7 * b
        
        # Term 1: M1 = (a+d)(e+h)
        U[0][base_col] += 1
        U[3][base_col] += 1
        V[2*b][base_col] += 1
        V[20+2*b+1][base_col] += 1
        W[4*b][base_col] += 1
        W[4*b+3][base_col] -= 1
        
        # Term 2: M2 = (c+d)e
        U[2][base_col+1] += 1
        U[3][base_col+1] += 1
        V[2*b][base_col+1] += 1
        W[4*b+1][base_col+1] += 1
        
        # Term 3: M3 = a(f-h)
        U[0][base_col+2] += 1
        V[2*b+1][base_col+2] += 1
        V[20+2*b+1][base_col+2] -= 1
        W[4*b+2][base_col+2] += 1
        W[4*b+3][base_col+2] += 1
        
        # Term 4: M4 = d(g-e)
        U[3][base_col+3] += 1
        V[20+2*b][base_col+3] += 1
        V[2*b][base_col+3] -= 1
        W[4*b][base_col+3] -= 1
        W[4*b+1][base_col+3] += 1
        
        # Term 5: M5 = (a+b)h
        U[0][base_col+4] += 1
        U[1][base_col+4] += 1
        V[20+2*b+1][base_col+4] += 1
        W[4*b+2][base_col+4] += 1
        W[4*b+3][base_col+4] += 1
        
        # Term 6: M6 = (c-a)(e+f)
        U[2][base_col+5] += 1
        U[0][base_col+5] -= 1
        V[2*b][base_col+5] += 1
        V[2*b+1][base_col+5] += 1
        W[4*b][base_col+5] += 1
        
        # Term 7: M7 = (b-d)(g+h)
        U[1][base_col+6] += 1
        U[3][base_col+6] -= 1
        V[20+2*b][base_col+6] += 1
        V[20+2*b+1][base_col+6] += 1
        W[4*b][base_col+6] -= 1
        W[4*b+3][base_col+6] += 1
        
    return {"U": U, "V": V, "W": W}
