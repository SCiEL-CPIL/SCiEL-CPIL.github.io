"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import random
import time

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _error(U, V, W, n, m, p, rank):
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                err += (s - 1.0) ** 2
    return err

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 135.0
    
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    
    for target_rank in [23, 22, 21]:
        if time.time() > deadline:
            break
        
        U, V, W = _schoolbook(n, m, p)
        U = [row[:target_rank] for row in U]
        V = [row[:target_rank] for row in V]
        W = [row[:target_rank] for row in W]
        
        best_err = _error(U, V, W, n, m, p, target_rank)
        
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            mat_choice = random.randint(0, 2)
            if mat_choice == 0:
                mat, nrows = U, n*m
            elif mat_choice == 1:
                mat, nrows = V, m*p
            else:
                mat, nrows = W, p*n
            
            r = random.randint(0, nrows - 1)
            c = random.randint(0, target_rank - 1)
            old_val = mat[r][c]
            new_val = old_val + random.gauss(0, 0.3)
            mat[r][c] = new_val
            
            new_err = _error(U, V, W, n, m, p, target_rank)
            
            if new_err < best_err:
                best_err = new_err
                if new_err < 1e-12:
                    return {"U": U, "V": V, "W": W}
            else:
                mat[r][c] = old_val
        
        if best_err < 1e-12:
            return {"U": U, "V": V, "W": W}
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
