"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # We'll use block decomposition: 4=2+2, 5=2+3, 6=3+3
    # Blocks:
    #   (0,0): 2x2x3, rank 12
    #   (0,1): 2x3x3, rank 18
    #   (1,0): 2x2x3, rank 12
    #   (1,1): 2x3x3, rank 18
    # Total rank: 12+18+12+18 = 60
    
    def build_block(nb, mb, pb, rank):
        """Build schoolbook decomposition for nb x mb x pb."""
        U = [[0]*rank for _ in range(nb*mb)]
        V = [[0]*rank for _ in range(mb*pb)]
        W = [[0]*rank for _ in range(pb*nb)]
        col = 0
        for i in range(nb):
            for j in range(mb):
                for k in range(pb):
                    U[i*mb+j][col] = 1
                    V[j*pb+k][col] = 1
                    W[k*nb+i][col] = 1
                    col += 1
        return U, V, W
    
    # Block dimensions
    n0, n1 = 2, 2  # split n=4
    m0, m1 = 2, 3  # split m=5
    p0, p1 = 3, 3  # split p=6
    
    # Build each block
    U00, V00, W00 = build_block(n0, m0, p0, n0*m0*p0)  # 2x2x3, rank 12
    U01, V01, W01 = build_block(n0, m1, p1, n0*m1*p1)  # 2x3x3, rank 18
    U10, V10, W10 = build_block(n1, m0, p0, n1*m0*p0)  # 2x2x3, rank 12
    U11, V11, W11 = build_block(n1, m1, p1, n1*m1*p1)  # 2x3x3, rank 18
    
    # Assemble full matrices
    # U: shape (n*m, rank) = (20, 60)
    # V: shape (m*p, rank) = (30, 60)
    # W: shape (p*n, rank) = (24, 60)
    
    total_rank = n0*m0*p0 + n0*m1*p1 + n1*m0*p0 + n1*m1*p1  # 12+18+12+18=60
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Place block (0,0): rows 0..1 in n, 0..1 in m, 0..2 in p
    offset = 0
    for i in range(n0):
        for j in range(m0):
            for k in range(p0):
                U[i*m + j][offset] = 1
                V[j*p + k][offset] = 1
                W[k*n + i][offset] = 1
                offset += 1
    
    # Place block (0,1): rows 0..1 in n, 2..4 in m, 0..2 in p
    for i in range(n0):
        for j in range(m1):
            for k in range(p1):
                U[i*m + (m0+j)][offset] = 1
                V[(m0+j)*p + k][offset] = 1
                W[k*n + i][offset] = 1
                offset += 1
    
    # Place block (1,0): rows 2..3 in n, 0..1 in m, 0..2 in p
    for i in range(n1):
        for j in range(m0):
            for k in range(p0):
                U[(n0+i)*m + j][offset] = 1
                V[j*p + k][offset] = 1
                W[k*n + (n0+i)][offset] = 1
                offset += 1
    
    # Place block (1,1): rows 2..3 in n, 2..4 in m, 0..2 in p
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U[(n0+i)*m + (m0+j)][offset] = 1
                V[(m0+j)*p + k][offset] = 1
                W[k*n + (n0+i)][offset] = 1
                offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

