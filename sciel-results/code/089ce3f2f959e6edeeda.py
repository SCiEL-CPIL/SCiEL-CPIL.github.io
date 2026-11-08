"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Split m=3 into m1=1 and m2=2
    m1 = 1
    m2 = 2
    
    # Part 1: (n, m1, p) = (2, 1, 6) - schoolbook, rank = n*m1*p = 12
    rank1 = n * m1 * p
    # Part 2: (n, m2, p) = (2, 2, 6) - use Strassen for each of p/2 = 3 blocks, rank = 3*7 = 21
    rank2 = (p // 2) * 7  # 3 * 7 = 21
    
    total_rank = rank1 + rank2
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]  # (6, 33)
    V = [[0] * total_rank for _ in range(m * p)]  # (18, 33)
    W = [[0] * total_rank for _ in range(p * n)]  # (12, 33)
    
    col = 0
    
    # Part 1: (2, 1, 6) schoolbook
    # A1 is 2x1, B1 is 1x6, C contribution is 2x6
    # U index: i*m + j where i in [0,2), j in [0, m1) = [0,1)
    # V index: j*p + k where j in [0, m1) = [0,1), k in [0,6)
    # W index: k*n + i where k in [0,6), i in [0,2)
    for i in range(n):
        for j in range(m1):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
    
    # Part 2: (2, 2, 6) using Strassen for each 2x2 block
    # A2 is 2x2 (rows i=0,1; cols j'=0,1 corresponding to global j=1,2)
    # B2 is 2x6 (rows j'=0,1; cols k=0..5)
    # C2 is 2x6 (rows i=0,1; cols k=0..5)
    
    # Strassen factors for 2x2 * 2x2 -> 2x2
    # U_S: 4 rows (a00, a01, a10, a11) x 7 cols (M1..M7)
    strassen_U = [
        [1, 0, 1, 0, 1, 1, 0],  # a00
        [0, 0, 0, 0, 1, 0, 1],  # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, 1],  # a11
    ]
    
    # V_S: 4 rows (b00, b01, b10, b11) x 7 cols (M1..M7)
    strassen_V = [
        [1, 1, 0, 1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0],  # b01
        [0, 0, 0, 1, 0, 0, 1],  # b10
        [1, 0, 1, 0, 1, 0, 1],  # b11
    ]
    
    # W_S: 4 rows (c00, c01, c10, c11) x 7 cols (M1..M7)
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 1, 0, 0, 1, 0],  # c11
    ]
    
    # Apply Strassen for each of the 3 blocks (b=0,1,2)
    for b in range(p // 2):
        for s in range(7):
            # U: A2 element at (i, j') -> global index i*m + (m1 + j')
            for i in range(n):
                for j_prime in range(m2):
                    # strassen_U row index: i * 2 + j_prime
                    U[i * m + (m1 + j_prime)][col] = strassen_U[i * 2 + j_prime][s]
            
            # V: B2 element at (j', k') -> global index (m1 + j')*p + (b*2 + k')
            for j_prime in range(m2):
                for k_prime in range(2):
                    # strassen_V row index: j_prime * 2 + k_prime
                    V[(m1 + j_prime) * p + (b * 2 + k_prime)][col] = strassen_V[j_prime * 2 + k_prime][s]
            
            # W: C2 element at (i, k') -> global index (b*2 + k')*n + i
            for i in range(n):
                for k_prime in range(2):
                    # strassen_W row index: k_prime * 2 + i
                    W[(b * 2 + k_prime) * n + i][col] = strassen_W[k_prime * 2 + i][s]
            
            col += 1
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
    
    assert col == total_rank, f"col={col}, total_rank={total_rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
