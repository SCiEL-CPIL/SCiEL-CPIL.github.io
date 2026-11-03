import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Theoretical minimum rank for this block decomposition family is 99.
    # 105 (schoolbook) - 6 (savings from 6 Strassen blocks) = 99.
    
    U = [[0] * 99 for _ in range(n * m)]
    V = [[0] * 99 for _ in range(m * p)]
    W = [[0] * 99 for _ in range(p * n)]
    
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
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[u00][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M2: (A10+A11)(B00) -> C10, C11
        U[u10][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        col += 1
        
        # M3: (A00)(B01-B11) -> C01, C11
        U[u00][col] = 1
        V[v01][col] = 1
        V[v11][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M4: (A11)(B10-B00) -> C00, C10
        U[u11][col] = 1
        V[v10][col] = 1
        V[v00][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # M5: (A00+A01)(B11) -> C00, C01
        U[u00][col] = 1
        U[u01][col] = 1
        V[v11][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U[u10][col] = 1
        U[u00][col] = -1
        V[v00][col] = 1
        V[v01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00
        U[u01][col] = 1
        U[u11][col] = -1
        V[v10][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        col += 1

    # Partition: n=[0:2],[2:3]; m=[0:2],[2:4],[4:5]; p=[0:2],[2:4],[4:6],[6:7]
    
    # 2x2x2 Strassen blocks (6 total)
    add_strassen(0, 0, 0)  # n[0:2], m[0:2], p[0:2]
    add_strassen(0, 0, 2)  # n[0:2], m[0:2], p[2:4]
    add_strassen(0, 0, 4)  # n[0:2], m[0:2], p[4:6]
    add_strassen(0, 2, 0)  # n[0:2], m[2:4], p[0:2]
    add_strassen(0, 2, 2)  # n[0:2], m[2:4], p[2:4]
    add_strassen(0, 2, 4)  # n[0:2], m[2:4], p[4:6]
    
    # Remaining schoolbook blocks
    add_schoolbook(0, 2, 0, 2, 6, 7)  # n[0:2], m[0:2], p[6:7]
    add_schoolbook(0, 2, 2, 4, 6, 7)  # n[0:2], m[2:4], p[6:7]
    add_schoolbook(0, 2, 4, 5, 0, 2)  # n[0:2], m[4:5], p[0:2]
    add_schoolbook(0, 2, 4, 5, 2, 4)  # n[0:2], m[4:5], p[2:4]
    add_schoolbook(0, 2, 4, 5, 4, 6)  # n[0:2], m[4:5], p[4:6]
    add_schoolbook(0, 2, 4, 5, 6, 7)  # n[0:2], m[4:5], p[6:7]
    add_schoolbook(2, 3, 0, 2, 0, 2)  # n[2:3], m[0:2], p[0:2]
    add_schoolbook(2, 3, 0, 2, 2, 4)  # n[2:3], m[0:2], p[2:4]
    add_schoolbook(2, 3, 0, 2, 4, 6)  # n[2:3], m[0:2], p[4:6]
    add_schoolbook(2, 3, 0, 2, 6, 7)  # n[2:3], m[0:2], p[6:7]
    add_schoolbook(2, 3, 2, 4, 0, 2)  # n[2:3], m[2:4], p[0:2]
    add_schoolbook(2, 3, 2, 4, 2, 4)  # n[2:3], m[2:4], p[2:4]
    add_schoolbook(2, 3, 2, 4, 4, 6)  # n[2:3], m[2:4], p[4:6]
    add_schoolbook(2, 3, 2, 4, 6, 7)  # n[2:3], m[2:4], p[6:7]
    add_schoolbook(2, 3, 4, 5, 0, 2)  # n[2:3], m[4:5], p[0:2]
    add_schoolbook(2, 3, 4, 5, 2, 4)  # n[2:3], m[4:5], p[2:4]
    add_schoolbook(2, 3, 4, 5, 4, 6)  # n[2:3], m[4:5], p[4:6]
    add_schoolbook(2, 3, 4, 5, 6, 7)  # n[2:3], m[4:5], p[6:7]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result['U'][0])
    
    assert rank == 99, f"Expected rank 99, got {rank}"
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
