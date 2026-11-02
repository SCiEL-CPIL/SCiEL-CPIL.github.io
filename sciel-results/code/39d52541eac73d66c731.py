"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * total_rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 44
    
    col = 0
    
    # For each (j_block, k_block) pair
    for jb in range(2):  # j block: 0 or 1
        for kb in range(2):  # k block: 0 or 1
            # j indices: 2*jb, 2*jb+1
            # k indices: 2*kb, 2*kb+1
            j0, j1 = 2 * jb, 2 * jb + 1
            k0, k1 = 2 * kb, 2 * kb + 1
            
            # --- Strassen part for i in {0,1}, j in {j0,j1}, k in {k0,k1} ---
            # 7 Strassen terms for 2x2x2
            # Local indices: i_loc in {0,1}, j_loc in {0,1}, k_loc in {0,1}
            # U index: i*m + j, V index: j*p + k, W index: k*n + i
            
            # M1 = (a00+a11)(b00+b11)
            U[0 * m + j0][col] = 1
            U[1 * m + j1][col] = 1
            V[j0 * p + k0][col] = 1
            V[j1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1
            
            # M2 = (a10+a11)b00
            U[1 * m + j0][col] = 1
            U[1 * m + j1][col] = 1
            V[j0 * p + k0][col] = 1
            W[k0 * n + 1][col] = 1
            col += 1
            
            # M3 = a00(b10-b11)
            U[0 * m + j0][col] = 1
            V[j1 * p + k0][col] = 1
            V[j1 * p + k1][col] = -1
            W[k0 * n + 0][col] = 1
            W[k1 * n + 0][col] = -1
            col += 1
            
            # M4 = a11(b01-b00)
            U[1 * m + j1][col] = 1
            V[j0 * p + k1][col] = 1
            V[j0 * p + k0][col] = -1
            W[k1 * n + 1][col] = 1
            W[k0 * n + 1][col] = -1
            col += 1
            
            # M5 = (a00+a01)b11
            U[0 * m + j0][col] = 1
            U[0 * m + j1][col] = 1
            V[j1 * p + k1][col] = 1
            W[k1 * n + 0][col] = 1
            col += 1
            
            # M6 = (a10-a00)(b00+b10)
            U[1 * m + j0][col] = 1
            U[0 * m + j0][col] = -1
            V[j0 * p + k0][col] = 1
            V[j1 * p + k0][col] = 1
            W[k0 * n + 1][col] = 1
            col += 1
            
            # M7 = (a01-a11)(b01+b11)
            U[0 * m + j1][col] = 1
            U[1 * m + j1][col] = -1
            V[j0 * p + k1][col] = 1
            V[j1 * p + k1][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1
            
            # --- Trivial part for i=2, j in {j0,j1}, k in {k0,k1} ---
            # 4 terms
            for j_loc in range(2):
                for k_loc in range(2):
                    j = j0 + j_loc
                    k = k0 + k_loc
                    U[2 * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + 2][col] = 1
                    col += 1
    
    # Verify col == total_rank
    assert col == total_rank, f"col={col}, expected {total_rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
