import time
import random

# EVOLVE-BLOCK-START

def _check_decomposition(U, V, W, n, m, p):
    """Check that the decomposition is exact for random inputs."""
    rank = len(U[0])
    for seed in range(10):
        random.seed(seed)
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        AB = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        X = [[0] * n for _ in range(rank)]
        for r in range(rank):
            for i in range(n):
                s = 0
                for j in range(m):
                    s += U[i * m + j][r] * A[i][j]
                X[r][i] = s
        
        Y = [[0] * p for _ in range(rank)]
        for r in range(rank):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += V[j * p + k][r] * B[j][k]
                Y[r][k] = s
        
        C = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += W[k * n + i][r] * X[r][i] * Y[r][k]
                C[i][k] = s
        
        if C != AB:
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x6():
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    U_s = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    V_s = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    W_s = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 1, 0, 0],   # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]
    
    for block in range(3):
        for s in range(7):
            r = block * 7 + s
            for ij in range(4):
                U[ij][r] = U_s[s][ij]
            for j in range(2):
                for k in range(2):
                    v_idx = j * 6 + (block * 2 + k)
                    u_idx = j * 2 + k
                    V[v_idx][r] = V_s[s][u_idx]
            for k in range(2):
                for i in range(2):
                    w_idx = (block * 2 + k) * 2 + i
                    w_s_idx = k * 2 + i
                    W[w_idx][r] = W_s[s][w_s_idx]
                    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, target_rank):
    """
    Try to find a rank reduction by identifying linear dependencies among the columns.
    We look for a set of columns that can be eliminated by expressing them as linear combinations
    of others, using the constraint that the tensor decomposition must remain exact.
    
    For the 2x2x6 case with 3 Strassen blocks, we try to merge the M1 terms from each block.
    The M1 term in each block contributes to the same (i,j) pattern in U, but different (j,k) in V.
    If we can find a linear combination that cancels out, we can reduce the rank.
    """
    n, m, p = 2, 2, 6
    current_rank = len(U[0])
    
    if current_rank <= target_rank:
        return {"U": U, "V": V, "W": W}
    
    # Try to eliminate one column at a time by finding linear dependencies
    # This is a heuristic approach - we look for columns that are linearly dependent
    # in the tensor sense
    
    # For now, this is a placeholder. A proper implementation would use
    # numerical linear algebra to find the minimal rank decomposition.
    # Given the time constraints, we'll just return the current decomposition.
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 2, 2, 6
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try Strassen-based decomposition first
    result = _strassen_2x2x6()
    if _check_decomposition(result["U"], result["V"], result["W"], n, m, p):
        rank = len(result["U"][0])
        return result
    
    # Fall back to schoolbook
    result = _schoolbook(n, m, p)
    return result

# EVOLVE-BLOCK-END
