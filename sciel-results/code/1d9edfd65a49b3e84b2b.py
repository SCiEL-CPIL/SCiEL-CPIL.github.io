import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Strassen coefficients for 2x2 * 2x2
    strassen_terms = [
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]
    
    def strassen_56():
        """8 independent 2x2 x 2x2 Strassen blocks, rank 56."""
        rank = 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        for a in range(2):  # A block column: 2a, 2a+1
            for b in range(4):  # B column block: 2b, 2b+1
                for t in range(7):
                    term = strassen_terms[t]
                    for (i_prime, j_prime), coeff in term['x'].items():
                        u_idx = i_prime * m + (2*a + j_prime)
                        U[u_idx, col] += coeff
                    for (j_prime, k_prime), coeff in term['y'].items():
                        v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                        V[v_idx, col] += coeff
                    for (i_prime, k_prime), coeff in term['z'].items():
                        w_idx = (2*b + k_prime) * n + i_prime
                        W[w_idx, col] += coeff
                    col += 1
        
        return U, V, W, rank
    
    def verify(U, V, W, rank, atol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :rank] * V[j*p+k, :rank] * W[k*n+i, :rank])
                    if not np.isclose(val, 1.0, atol=atol):
                        return False
        return True
    
    # Start with Strassen (rank 56)
    U_best, V_best, W_best, rank_best = strassen_56()
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()
    
    # Try to reduce rank by finding linear dependencies among the 56 terms
    # or by finding a more efficient block decomposition
    # The key observation: M_{2,4,8} = sum over 8 blocks of M_{2,2,2}
    # Each block has rank 7, total 56.
    # Can we share structure between blocks?
    
    # Try rank 48: remove 8 terms if they are linearly dependent
    def try_reduce_rank(target_rank):
        if target_rank >= rank_best:
            return None
        if time.time() > deadline - 15:
            return None
        
        U = U_best[:, :target_rank].copy()
        V = V_best[:, :target_rank].copy()
        W = W_best[:, :target_rank].copy()
        
        if verify(U, V, W, target_rank):
            return U, V, W, target_rank
        return None
    
    # Try various target ranks
    for target in [50, 48, 46, 44, 42, 40, 38, 36, 34, 32]:
        if time.time() > deadline - 15:
            break
        result = try_reduce_rank(target)
        if result is not None:
            U_best, V_best, W_best, rank_best = result
            break
    
    # Final verification
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
