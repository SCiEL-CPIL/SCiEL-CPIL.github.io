import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices:
    # 0: a00, 1: a01, 2: a02
    # 3: a10, 4: a11, 5: a12
    
    # B indices for block b (cols 2b, 2b+1):
    # b00: 0*8 + 2b = 16b
    # b01: 0*8 + 2b+1 = 16b+1
    # b10: 1*8 + 2b = 8+16b
    # b11: 1*8 + 2b+1 = 9+16b
    # b20: 2*8 + 2b = 16+16b
    # b21: 2*8 + 2b+1 = 17+16b
    
    # C indices for block b (cols 2b, 2b+1):
    # C00 (i=0, k=2b): 2b*2 + 0 = 4b
    # C10 (i=1, k=2b): 2b*2 + 1 = 4b+1
    # C01 (i=0, k=2b+1): (2b+1)*2 + 0 = 4b+2
    # C11 (i=1, k=2b+1): (2b+1)*2 + 1 = 4b+3
    
    for b in range(4):
        base_rank = b * 11
        
        # M1
        r = base_rank
        U[0][r] += 1
        U[4][r] += 1
        V[16*b][r] += 1
        V[9+16*b][r] += 1
        W[4*b][r] += 1
        W[4*b+3][r] += 1
        
        # M2
        r = base_rank + 1
        U[3][r] += 1
        U[4][r] += 1
        V[16*b][r] += 1
        W[4*b+1][r] += 1
        W[4*b+3][r] -= 1
        
        # M3
        r = base_rank + 2
        U[0][r] += 1
        V[16*b+1][r] += 1
        V[9+16*b][r] -= 1
        W[4*b+2][r] += 1
        W[4*b+3][r] += 1
        
        # M4
        r = base_rank + 3
        U[4][r] += 1
        V[8+16*b][r] += 1
        V[16*b][r] -= 1
        W[4*b][r] += 1
        W[4*b+1][r] += 1
        
        # M5
        r = base_rank + 4
        U[0][r] += 1
        U[1][r] += 1
        V[9+16*b][r] += 1
        W[4*b][r] -= 1
        W[4*b+2][r] += 1
        
        # M6
        r = base_rank + 5
        U[3][r] += 1
        U[0][r] -= 1
        V[16*b][r] += 1
        V[16*b+1][r] += 1
        W[4*b+3][r] += 1
        
        # M7
        r = base_rank + 6
        U[1][r] += 1
        U[4][r] -= 1
        V[8+16*b][r] += 1
        V[9+16*b][r] += 1
        W[4*b][r] += 1
        W[4*b+3][r] += 1
        
        # N1
        r = base_rank + 7
        U[2][r] += 1
        V[16+16*b][r] += 1
        W[4*b][r] += 1
        
        # N2
        r = base_rank + 8
        U[2][r] += 1
        V[17+16*b][r] += 1
        W[4*b+2][r] += 1
        
        # N3
        r = base_rank + 9
        U[5][r] += 1
        V[16+16*b][r] += 1
        W[4*b+1][r] += 1
        
        # N4
        r = base_rank + 10
        U[5][r] += 1
        V[17+16*b][r] += 1
        W[4*b+3][r] += 1

    return {"U": U, "V": V, "W": W}
