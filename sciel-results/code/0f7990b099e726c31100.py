"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe (120s wall-clock limit)
    
    n, m, p = DIMENSIONS
    
    def add_strassen(U, V, W, i0, j0, k0, col):
        # M1: (a11+a22)(b11+b22)
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2: (a21+a22)b11
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3: a11(b21-b22)
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4: (a11-a22)b22
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5: (a11+a12)b22
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6: (a21-a11)(b11+b12)
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7: (a12-a22)(b21+b22)
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col

    def add_schoolbook(U, V, W, i_range, j_range, k_range, col):
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col

    rank = 99
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 6 Strassen blocks: i in {0,1}, j in {0,1} or {2,3}, k in {0,1} or {2,3} or {4,5}
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            col = add_strassen(U, V, W, 0, j0, k0, col)
    
    # Remaining schoolbook terms
    # i in {0,1}, j in {0,1}, k in {6}
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook(U, V, W, [0, 1], [0, 1], [6], col)
    
    # i in {0,1}, j in {4}, k in {0..6}
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook(U, V, W, [0, 1], [4], range(7), col)
    
    # i in {2}, j in {0,1}, k in {0..6}
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook(U, V, W, [2], [0, 1], range(7), col)
    
    # i in {2}, j in {2,3}, k in {0..6}
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook(U, V, W, [2], [2, 3], range(7), col)
    
    # i in {2}, j in {4}, k in {0..6}
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook(U, V, W, [2], [4], range(7), col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
