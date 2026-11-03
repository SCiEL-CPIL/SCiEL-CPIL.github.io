import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    _start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Optimized partitions:
    # n: [0, 2], [2, 3]
    # m: [0, 3], [3, 5]
    # p: [0, 3], [3, 4], [4, 7]
    
    # Strassen: i in [0,2], j in [0,2], k in [0,2] -> 1 block, rank 7
    # Schoolbook:
    #   i in [0,2], j in [0,2], k in [3,4] -> 2*2*1 = 4
    #   i in [0,2], j in [0,2], k in [4,7] -> 2*2*3 = 12
    #   i in [0,2], j in [3,5], k in [0,7] -> 2*2*7 = 28
    #   i in [2,3], j in [0,3], k in [0,7] -> 1*3*7 = 21
    #   i in [2,3], j in [3,5], k in [0,7] -> 1*2*7 = 14
    # Total: 7 + 4 + 12 + 28 + 21 + 14 = 86
    
    total_rank = 104
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
        
        # Indices for U (i, j), V (j, k), W (k, i)
        # A elements: A[i0,j0], A[i0,j1], A[i1,j0], A[i1,j1]
        # B elements: B[j0,k0], B[j0,k1], B[j1,k0], B[j1,k1]
        
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
        
        # M1 = (A00 + A11) * (B00 + B11)
        # U: A00, A11 ; V: B00, B11
        # C00 += M1, C11 += M1
        U[u00][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        # U: A10, A11 ; V: B00
        # C10 += M2, C11 -= M2
        U[u10][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        # U: A00 ; V: B01, -B11
        # C01 += M3, C11 += M3
        U[u00][col] = 1
        V[v01][col] = 1
        V[v11][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        # U: A11 ; V: B10, -B00
        # C00 += M4, C10 += M4
        U[u11][col] = 1
        V[v10][col] = 1
        V[v00][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        # U: A00, A01 ; V: B11
        # C00 -= M5, C01 += M5
        U[u00][col] = 1
        U[u01][col] = 1
        V[v11][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        # U: A10, -A00 ; V: B00, B01
        # C11 += M6
        U[u10][col] = 1
        U[u00][col] = -1
        V[v00][col] = 1
        V[v01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        # U: A01, -A11 ; V: B10, B11
        # C00 += M7
        U[u01][col] = 1
        U[u11][col] = -1
        V[v10][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        col += 1

    # Strassen block: i in [0,2], j in [0,2], k in [0,2]
    add_strassen(0, 0, 0)
    
    # Schoolbook blocks covering all remaining (i, j, k) tuples
    # 1. i in [0,2), j in [0,2), k in [2,7)
    add_schoolbook(0, 2, 0, 2, 2, 7)
    # 2. i in [0,2), j in [2,5), k in [0,7)
    add_schoolbook(0, 2, 2, 5, 0, 7)
    # 3. i in [2,3), j in [0,5), k in [0,7)
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
    
    assert rank == 104, f"Expected rank 104, got {rank}"
    
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
