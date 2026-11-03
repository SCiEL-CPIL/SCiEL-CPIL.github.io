import time
import numpy as np
import random

DIMENSIONS = [3, 3, 7]
_DEADLINE_SECONDS = 120  # safety margin before 150s kill


def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x7 matrix.

    Returns:
        A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
        U has shape (n*m, rank)
        V has shape (m*p, rank)
        W has shape (p*n, rank)
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank_trivial = n * m * p
    
    # Trivial decomposition
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() - start_time > _DEADLINE_SECONDS - 10:
                    return {"U": U_trivial, "V": V_trivial, "W": W_trivial}
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
                
    # Target ranks
    target_ranks = [55, 50, 45]
    
    # Precompute target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1
                
    for R in target_ranks:
        if time.time() - start_time > _DEADLINE_SECONDS - 30:
            break
            
        for attempt in range(10):
            if time.time() - start_time > _DEADLINE_SECONDS - 10:
                break
                
            U = np.random.randint(-2, 3, size=(n*m, R))
            V = np.random.randint(-2, 3, size=(m*p, R))
            W = np.random.randint(-2, 3, size=(p*n, R))
            
            best_err = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(200):
                if time.time() - start_time > _DEADLINE_SECONDS - 10:
                    break
                    
                T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.sum((T_approx - T_target) ** 2)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                if best_err == 0:
                    break
                
                # Perturb
                mat = random.choice([U, V, W])
                idx = random.randrange(mat.size)
                mat.flat[idx] += random.choice([-1, 1])
                
            if best_err == 0:
                return {
                    "U": best_U.tolist(),
                    "V": best_V.tolist(),
                    "W": best_W.tolist()
                }

    return {"U": U_trivial, "V": V_trivial, "W": W_trivial}
