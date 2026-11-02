import time

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # We use 5 pairs of columns (0-1, 2-3, 4-5, 6-7, 8-9) and 1 single column (10)
    # Rank = 5 * 7 (Strassen) + 4 (Schoolbook) = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Process 5 pairs
    for pair_idx in range(5):
        l = 2 * pair_idx
        l_next = l + 1
        
        # M1: (A00 + A11) * (B0l + B1_l_next)
        U[0][col] += 1
        U[3][col] += 1
        V[l][col] += 1
        V[11 + l_next][col] += 1
        W[2*l][col] += 1      # C0l
        W[2*l+3][col] += 1    # C1_l_next
        
        # M2: (A10 + A11) * B0l
        U[2][col] += 1
        U[3][col] += 1
        V[l][col] += 1
        W[2*l+1][col] += 1    # C1l
        W[2*l+3][col] -= 1    # C1_l_next
        
        # M3: A00 * (B0_l_next - B1_l_next)
        U[0][col] += 1
        V[l_next][col] += 1
        V[11 + l_next][col] -= 1
        W[2*l+2][col] += 1    # C0_l_next
        W[2*l+3][col] += 1    # C1_l_next
        
        # M4: A11 * (B0l - B0_l_next)
        U[3][col] += 1
        V[l][col] += 1
        V[l_next][col] -= 1
        W[2*l][col] -= 1      # C0l
        W[2*l+1][col] += 1    # C1l
        
        # M5: (A00 + A01) * B1_l_next
        U[0][col] += 1
        U[1][col] += 1
        V[11 + l_next][col] += 1
        W[2*l][col] += 1      # C0l
        
        # M6: (A10 - A00) * (B0l + B0_l_next)
        U[2][col] += 1
        U[0][col] -= 1
        V[l][col] += 1
        V[l_next][col] += 1
        W[2*l+3][col] += 1    # C1_l_next
        
        # M7: (A01 - A11) * (B1l + B1_l_next)
        U[1][col] += 1
        U[3][col] -= 1
        V[11 + l][col] += 1
        V[11 + l_next][col] += 1
        W[2*l][col] -= 1      # C0l
        W[2*l+2][col] += 1    # C0_l_next
        
        col += 7
        
    # Process last single column (l=10)
    l = 10
    
    # A00 * B0l -> C0l
    U[0][col] += 1
    V[l][col] += 1
    W[2*l][col] += 1
    col += 1
    
    # A01 * B1l -> C0l
    U[1][col] += 1
    V[11 + l][col] += 1
    W[2*l][col] += 1
    col += 1
    
    # A10 * B0l -> C1l
    U[2][col] += 1
    V[l][col] += 1
    W[2*l+1][col] += 1
    col += 1
    
    # A11 * B1l -> C1l
    U[3][col] += 1
    V[11 + l][col] += 1
    W[2*l+1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
