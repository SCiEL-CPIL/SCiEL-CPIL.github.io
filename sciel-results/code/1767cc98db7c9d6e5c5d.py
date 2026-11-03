"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100.0
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # 4 blocks of 11 rank terms each
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U indices:
    # 0: a00, 1: a01, 2: a02
    # 3: a10, 4: a11, 5: a12
    
    col = 0
    for q in range(4):
        # V indices for this block
        v_b00 = 0 * 8 + 2 * q
        v_b01 = 0 * 8 + 2 * q + 1
        v_b10 = 1 * 8 + 2 * q
        v_b11 = 1 * 8 + 2 * q + 1
        v_b20 = 2 * 8 + 2 * q
        v_b21 = 2 * 8 + 2 * q + 1
        
        # W indices: W[k*n+i, r] where k=col of C (0..7), i=row of C (0..1)
        # C[0][2q]:   i=0, k=2q   -> 2*(2q)+0 = 4q
        # C[0][2q+1]: i=0, k=2q+1 -> 2*(2q+1)+0 = 4q+2
        # C[1][2q]:   i=1, k=2q   -> 2*(2q)+1 = 4q+1
        # C[1][2q+1]: i=1, k=2q+1 -> 2*(2q+1)+1 = 4q+3
        
        w_c00 = 4 * q
        w_c01 = 4 * q + 2
        w_c10 = 4 * q + 1
        w_c11 = 4 * q + 3

        # Term 1: P1 = (a00 + a11)(b00 + b11)
        U[0][col] = 1
        U[4][col] = 1
        V[v_b00][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # Term 2: P2 = (a10 + a11)b00
        U[3][col] = 1
        U[4][col] = 1
        V[v_b00][col] = 1
        W[w_c10][col] = 1
        W[w_c11][col] = -1
        col += 1
        
        # Term 3: P3 = a00(b01 - b11)
        U[0][col] = 1
        V[v_b01][col] = 1
        V[v_b11][col] = -1
        W[w_c01][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # Term 4: P4 = a11(b10 - b00)
        U[4][col] = 1
        V[v_b10][col] = 1
        V[v_b00][col] = -1
        W[w_c00][col] = 1
        W[w_c10][col] = 1
        col += 1
        
        # Term 5: P5 = (a00 + a01)b11
        U[0][col] = 1
        U[1][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = -1
        W[w_c01][col] = 1
        col += 1
        
        # Term 6: P6 = (a10 - a00)(b00 + b01)
        U[3][col] = 1
        U[0][col] = -1
        V[v_b00][col] = 1
        V[v_b01][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # Term 7: P7 = (a01 - a11)(b10 + b11)
        U[1][col] = 1
        U[4][col] = -1
        V[v_b10][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = 1
        col += 1
        
        # Term 8: D00 = a02 b20
        U[2][col] = 1
        V[v_b20][col] = 1
        W[w_c00][col] = 1
        col += 1
        
        # Term 9: D01 = a02 b21
        U[2][col] = 1
        V[v_b21][col] = 1
        W[w_c01][col] = 1
        col += 1
        
        # Term 10: D10 = a12 b20
        U[5][col] = 1
        V[v_b20][col] = 1
        W[w_c10][col] = 1
        col += 1
        
        # Term 11: D11 = a12 b21
        U[5][col] = 1
        V[v_b21][col] = 1
        W[w_c11][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
