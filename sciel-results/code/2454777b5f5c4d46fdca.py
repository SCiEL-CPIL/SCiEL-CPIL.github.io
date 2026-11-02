import time

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2 matrices, applied to 4 blocks
    # Block l corresponds to columns 2l and 2l+1 of B and C
    
    for l in range(4):
        # Base indices for B columns
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # Base indices for V (rows j*p + k)
        # j=0: k0, k1
        # j=1: p+k0, p+k1
        v_00 = 0 * p + k0
        v_01 = 0 * p + k1
        v_10 = 1 * p + k0
        v_11 = 1 * p + k1
        
        # Base indices for W (rows k*n + i)
        # k=k0, i=0: 2*k0
        # k=k0, i=1: 2*k0 + 1
        # k=k1, i=0: 2*k1
        # k=k1, i=1: 2*k1 + 1
        w_00 = k0 * n + 0
        w_10 = k0 * n + 1
        w_01 = k1 * n + 0
        w_11 = k1 * n + 1
        
        # U indices (rows i*m + j)
        # a00: 0, a01: 1, a10: 2, a11: 3
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        # 7 Strassen products for this block
        # r = l * 7 + s
        
        # M1 = (a00 + a11)(b00 + b11)
        r = l * 7 + 0
        U[u_00][r] += 1
        U[u_11][r] += 1
        V[v_00][r] += 1
        V[v_11][r] += 1
        W[w_00][r] += 1
        W[w_11][r] += 1
        
        # M2 = (a10 + a11)b00
        r = l * 7 + 1
        U[u_10][r] += 1
        U[u_11][r] += 1
        V[v_00][r] += 1
        W[w_10][r] += 1
        W[w_11][r] += -1
        
        # M3 = a00(b01 - b11)
        r = l * 7 + 2
        U[u_00][r] += 1
        V[v_01][r] += 1
        V[v_11][r] += -1
        W[w_01][r] += 1
        W[w_11][r] += 1
        
        # M4 = a11(b10 - b00)
        r = l * 7 + 3
        U[u_11][r] += 1
        V[v_10][r] += 1
        V[v_00][r] += -1
        W[w_00][r] += 1
        W[w_10][r] += 1
        
        # M5 = (a00 + a10)b11
        r = l * 7 + 4
        U[u_00][r] += 1
        U[u_10][r] += 1
        V[v_11][r] += 1
        W[w_00][r] += -1
        W[w_01][r] += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        r = l * 7 + 5
        U[u_10][r] += 1
        U[u_00][r] += -1
        V[v_00][r] += 1
        V[v_01][r] += 1
        W[w_11][r] += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        r = l * 7 + 6
        U[u_01][r] += 1
        U[u_11][r] += -1
        V[v_10][r] += 1
        V[v_11][r] += 1
        W[w_00][r] += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
