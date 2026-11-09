"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 x 3x4 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try a quick search for rank 16-22
    # Use a very fast heuristic
    
    import random
    random.seed(42)
    
    for rank in range(16, 23):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Sparse random init
            U = [[random.choice([0, 0, 0, 0.5, -0.5, 1, -1]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([0, 0, 0, 0.5, -0.5, 1, -1]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([0, 0, 0, 0.5, -0.5, 1, -1]) for _ in range(rank)] for _ in range(p*n)]
            
            # Quick local search
            if _quick_local_search(U, V, W, n, m, p, rank, deadline):
                return {"U": U, "V": V, "W": W}
    
    return schoolbook()

def _quick_local_search(U, V, W, n, m, p, rank, deadline):
    """Very fast local search."""
    import random
    
    for step in range(500):
        if time.time() > deadline:
            return False
        
        # Check if valid
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Check half-integer
            for mat in [U, V, W]:
                for row in mat:
                    for v in row:
                        if abs(v*2 - round(v*2)) > 1e-9:
                            return False
            return True
        
        # Fix worst violation
        worst = None
        worst_err = -1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    err = abs(val - 1)
                    if err > worst_err:
                        worst_err = err
                        worst = (i, j, k)
        
        i, j, k = worst
        c = random.randint(0, rank-1)
        mat = random.choice(['U', 'V', 'W'])
        
        if mat == 'U':
            row = i*m+j
            old = U[row][c]
            for new in [0, 0.5, -0.5, 1, -1, 1.5, -1.5]:
                if new == old:
                    continue
                U[row][c] = new
                new_val = 0
                for cc in range(rank):
                    new_val += U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc]
                if abs(new_val - 1) < worst_err:
                    break
                U[row][c] = old
        elif mat == 'V':
            row = j*p+k
            old = V[row][c]
            for new in [0, 0.5, -0.5, 1, -1, 1.5, -1.5]:
                if new == old:
                    continue
                V[row][c] = new
                new_val = 0
                for cc in range(rank):
                    new_val += U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc]
                if abs(new_val - 1) < worst_err:
                    break
                V[row][c] = old
        else:
            row = k*n+i
            old = W[row][c]
            for new in [0, 0.5, -0.5, 1, -1, 1.5, -1.5]:
                if new == old:
                    continue
                W[row][c] = new
                new_val = 0
                for cc in range(rank):
                    new_val += U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc]
                if abs(new_val - 1) < worst_err:
                    break
                W[row][c] = old
    
    return False
# EVOLVE-BLOCK-END
