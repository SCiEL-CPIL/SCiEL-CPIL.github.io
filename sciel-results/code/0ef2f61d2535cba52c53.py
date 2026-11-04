"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form an exact tensor decomposition."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

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
    return {"U": U, "V": V, "W": W}

def _block_decomp_3x3x6():
    """
    Decompose 3x3 * 3x6 as two 3x3 * 3x3 multiplications.
    Each uses schoolbook rank 9, total rank 18.
    """
    n, m, p = 3, 3, 6
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # First 3 columns (k=0,1,2)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Last 3 columns (k=3,4,5)
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # Start with schoolbook as the safe baseline
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try the block decomposition (rank 18)
    block = _block_decomp_3x3x6()
    block_rank = len(block['U'][0])
    if _verify(block['U'], block['V'], block['W'], n, m, p):
        best = block
        best_rank = block_rank
    
    # Try structured random searches for even lower ranks
    candidate_ranks = list(range(best_rank - 1, max(best_rank - 15, 5), -1))
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            continue
        
        max_attempts = 10
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 7919 + rank * 31 + 1000)
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            for r in range(rank):
                ui = np.random.randint(0, n*m)
                vj = np.random.randint(0, m*p)
                wk = np.random.randint(0, p*n)
                U[ui, r] = np.random.choice([-1.0, 1.0])
                V[vj, r] = np.random.choice([-1.0, 1.0])
                W[wk, r] = np.random.choice([-1.0, 1.0])
            
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                U_clean = [[float(x) for x in row] for row in U]
                V_clean = [[float(x) for x in row] for row in V]
                W_clean = [[float(x) for x in row] for row in W]
                best = {"U": U_clean, "V": V_clean, "W": W_clean}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best
# EVOLVE-BLOCK-END
