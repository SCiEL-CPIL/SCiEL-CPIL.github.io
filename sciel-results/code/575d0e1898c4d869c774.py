import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    _start_time = time.time()
    n, m, p = DIMENSIONS
    
    # 6 Strassen 2x2x2 blocks (rank 7 each = 42) + 53 schoolbook = rank 95
    # Strassen blocks cover: rows {0,1}, cols {0,1}x{0,1},{0,1}x{2,3},{0,1}x{4,5},
    #                       rows {0,1}, cols {2,3}x{0,1},{2,3}x{2,3},{2,3}x{4,5}
    # Schoolbook covers:
    #   rows {0,1}, col 0, k=6: 1
    #   rows {0,1}, col 1, k=6: 1
    #   rows {0,1}, col 2, k=6: 1
    #   rows {0,1}, col 3, k=6: 1
    #   rows {0,1}, col 4, k=0..6: 7
    #   row 2, col 0, k=0..6: 7
    #   row 2, col 1, k=0..6: 7
    #   row 2, col 2, k=0..6: 7
    #   row 2, col 3, k=0..6: 7
    #   row 2, col 4, k=0..6: 7
    # Total schoolbook: 4 + 7 + 35 = 46... let me recount
    # Actually: rows{0,1} col{0,1} k{6}: 2*1*1=2, rows{0,1} col{2,3} k{6}: 2*1*1=2
    # rows{0,1} col{4} k{0..6}: 2*1*7=14, row{2} col{0..4} k{0..6}: 1*5*7=35
    # Total: 2+2+14+35 = 53
    # Grand total: 42 + 53 = 95
    
    total_rank = 95
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col = 0
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][col] = 1
                    V[idx_v][col] = 1
                    W[idx_w][col] = 1
                    col += 1

    def add_strassen(i0, j0, k0):
        nonlocal col
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        u00 = i0 * m + j0
        u01 = i0 * m + j1
        u10 = i1 * m + j0
        u11 = i1 * m + j1
        
        v00 = j0 * p + k0
        v01 = j0 * p + k1
        v10 = j1 * p + k0
        v11 = j1 * p + k1
        
        w00 = k0 * n + i0
        w01 = k1 * n + i0
        w10 = k0 * n + i1
        w11 = k1 * n + i1
        
        # M1 = (A00 + A11) * (B00 + B11) -> C00 += M1, C11 += M1
        U[u00][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B00 -> C10 += M2, C11 -= M2
        U[u10][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        col += 1
        
        # M3 = A00 * (B01 - B11) -> C01 += M3, C11 += M3
        U[u00][col] = 1
        V[v01][col] = 1
        V[v11][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M4 = A11 * (B10 - B00) -> C00 += M4, C10 += M4
        U[u11][col] = 1
        V[v10][col] = 1
        V[v00][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B11 -> C00 -= M5, C01 += M5
        U[u00][col] = 1
        U[u01][col] = 1
        V[v11][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01) -> C11 += M6
        U[u10][col] = 1
        U[u00][col] = -1
        V[v00][col] = 1
        V[v01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11) -> C00 += M7
        U[u01][col] = 1
        U[u11][col] = -1
        V[v10][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        col += 1

    # 6 Strassen blocks:
    # Block 1: rows(0,1), cols(0,1), k(0,1)
    add_strassen(0, 0, 0)
    # Block 2: rows(0,1), cols(0,1), k(2,3)
    add_strassen(0, 0, 2)
    # Block 3: rows(0,1), cols(0,1), k(4,5)
    add_strassen(0, 0, 4)
    # Block 4: rows(0,1), cols(2,3), k(0,1)
    add_strassen(0, 2, 0)
    # Block 5: rows(0,1), cols(2,3), k(2,3)
    add_strassen(0, 2, 2)
    # Block 6: rows(0,1), cols(2,3), k(4,5)
    add_strassen(0, 2, 4)
    
    # Schoolbook for remaining terms:
    # Rows 0-1, cols 0-1, k=6: 2*1*1 = 2
    add_schoolbook(0, 2, 0, 2, 6, 7)
    # Rows 0-1, cols 2-3, k=6: 2*1*1 = 2
    add_schoolbook(0, 2, 2, 4, 6, 7)
    # Rows 0-1, col 4, k=0..6: 2*1*7 = 14
    add_schoolbook(0, 2, 4, 5, 0, 7)
    # Row 2, cols 0-4, k=0..6: 1*5*7 = 35
    add_schoolbook(2, 3, 0, 5, 0, 7)
    
    # Deadline guard
    if time.time() - _start_time > 110:
        pass
    
    # Trim trailing zero columns if any
    final_rank = col
    U = [row[:final_rank] for row in U]
    V = [row[:final_rank] for row in V]
    W = [row[:final_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result['U'][0])
    
    assert rank == 95, f"Expected rank 95, got {rank}"
    assert len(result['U']) == n * m
    assert len(result['V']) == m * p
    assert len(result['W']) == p * n
    
    import random
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    C_decomp = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(p):
            val = 0
            for r in range(rank):
                ua = 0
                for jj in range(m):
                    ua += result['U'][i * m + jj][r] * A[i][jj]
                vb = 0
                for jj in range(m):
                    vb += result['V'][jj * p + k][r] * B[jj][k]
                val += result['W'][k * n + i][r] * ua * vb
            C_decomp[i][k] = val
            
    if C_decomp != C:
        print("Verification failed!", file=sys.stderr)
    else:
        print("Verification passed.", file=sys.stderr)
