import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 99 decomposition
    # 6 Strassen blocks (rank 7 each) + 57 Schoolbook terms (rank 1 each)
    rank = 99
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
        
        # U indices
        u00 = i0 * m + j0
        u01 = i0 * m + j1
        u10 = i1 * m + j0
        u11 = i1 * m + j1
        
        # V indices
        v00 = j0 * p + k0
        v01 = j0 * p + k1
        v10 = j1 * p + k0
        v11 = j1 * p + k1
        
        # W indices
        w00 = k0 * n + i0
        w01 = k1 * n + i0
        w10 = k0 * n + i1
        w11 = k1 * n + i1
        
        # M1
        U[u00][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M2
        U[u10][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        col += 1
        
        # M3
        U[u00][col] = 1
        V[v01][col] = 1
        V[v11][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M4
        U[u11][col] = 1
        V[v10][col] = 1
        V[v00][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # M5
        U[u00][col] = 1
        U[u01][col] = 1
        V[v11][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        col += 1
        
        # M6
        U[u10][col] = 1
        U[u00][col] = -1
        V[v00][col] = 1
        V[v01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M7
        U[u01][col] = 1
        U[u11][col] = -1
        V[v10][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        col += 1

    # Define blocks
    # n: [0,2), [2,3)
    # m: [0,2), [2,4), [4,5)
    # p: [0,2), [2,4), [4,6), [6,7)
    
    # Block 1: n[0:2], m[0:2], p[0:2] -> Strassen
    add_strassen(0, 0, 0)
    
    # Block 2: n[0:2], m[0:2], p[2:4] -> Strassen
    add_strassen(0, 0, 2)
    
    # Block 3: n[0:2], m[0:2], p[4:6] -> Strassen
    add_strassen(0, 0, 4)
    
    # Block 4: n[0:2], m[0:2], p[6:7] -> Schoolbook
    add_schoolbook(0, 2, 0, 2, 6, 7)
    
    # Block 5: n[0:2], m[2:4], p[0:2] -> Strassen
    add_strassen(0, 2, 0)
    
    # Block 6: n[0:2], m[2:4], p[2:4] -> Strassen
    add_strassen(0, 2, 2)
    
    # Block 7: n[0:2], m[2:4], p[4:6] -> Strassen
    add_strassen(0, 2, 4)
    
    # Block 8: n[0:2], m[2:4], p[6:7] -> Schoolbook
    add_schoolbook(0, 2, 2, 4, 6, 7)
    
    # Block 9: n[0:2], m[4:5], p[0:2] -> Schoolbook
    add_schoolbook(0, 2, 4, 5, 0, 2)
    
    # Block 10: n[0:2], m[4:5], p[2:4] -> Schoolbook
    add_schoolbook(0, 2, 4, 5, 2, 4)
    
    # Block 11: n[0:2], m[4:5], p[4:6] -> Schoolbook
    add_schoolbook(0, 2, 4, 5, 4, 6)
    
    # Block 12: n[0:2], m[4:5], p[6:7] -> Schoolbook
    add_schoolbook(0, 2, 4, 5, 6, 7)
    
    # Block 13: n[2:3], m[0:2], p[0:2] -> Schoolbook
    add_schoolbook(2, 3, 0, 2, 0, 2)
    
    # Block 14: n[2:3], m[0:2], p[2:4] -> Schoolbook
    add_schoolbook(2, 3, 0, 2, 2, 4)
    
    # Block 15: n[2:3], m[0:2], p[4:6] -> Schoolbook
    add_schoolbook(2, 3, 0, 2, 4, 6)
    
    # Block 16: n[2:3], m[0:2], p[6:7] -> Schoolbook
    add_schoolbook(2, 3, 0, 2, 6, 7)
    
    # Block 17: n[2:3], m[2:4], p[0:2] -> Schoolbook
    add_schoolbook(2, 3, 2, 4, 0, 2)
    
    # Block 18: n[2:3], m[2:4], p[2:4] -> Schoolbook
    add_schoolbook(2, 3, 2, 4, 2, 4)
    
    # Block 19: n[2:3], m[2:4], p[4:6] -> Schoolbook
    add_schoolbook(2, 3, 2, 4, 4, 6)
    
    # Block 20: n[2:3], m[2:4], p[6:7] -> Schoolbook
    add_schoolbook(2, 3, 2, 4, 6, 7)
    
    # Block 21: n[2:3], m[4:5], p[0:2] -> Schoolbook
    add_schoolbook(2, 3, 4, 5, 0, 2)
    
    # Block 22: n[2:3], m[4:5], p[2:4] -> Schoolbook
    add_schoolbook(2, 3, 4, 5, 2, 4)
    
    # Block 23: n[2:3], m[4:5], p[4:6] -> Schoolbook
    add_schoolbook(2, 3, 4, 5, 4, 6)
    
    # Block 24: n[2:3], m[4:5], p[6:7] -> Schoolbook
    add_schoolbook(2, 3, 4, 5, 6, 7)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result['U'][0])
    
    assert rank == 99
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
