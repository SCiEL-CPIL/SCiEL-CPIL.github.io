"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import random
import itertools

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor.
    T[a, b, c] where a indexes (i,j) for A, b indexes (k,l) for B, c indexes (r = l*n+i) for C
    T[a,b,c] = 1 if j==k else 0
    """
    tensor = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(m):
                    if j == k:
                        a = i * m + j
                        b = k * m + l
                        c = l * n + i
                        tensor[a, b, c] = 1
    return tensor

def _verify_decomposition(U, V, W, tensor, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == tensor."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    recon = np.zeros_like(tensor, dtype=np.float64)
    for r in range(rank):
        recon += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    return np.max(np.abs(recon - tensor)) < tol

def _schoolbook_decomposition(n, m, p):
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

def _try_rank_reduction(n, m, p, start_rank, deadline):
    """Try to reduce rank from start_rank by searching over small integer/half-integer entries."""
    tensor = _build_tensor(n, m, p)
    best_rank = start_rank
    best_result = _schoolbook_decomposition(n, m, p)
    
    # Try ranks from start_rank-1 down to a reasonable lower bound
    # For 2x5 x 5x5, known lower bounds suggest rank >= 42
    lower_bound = max(42, (n*m + m*p + p*n - 2) // 2)  # Strassen-like lower bound
    
    for target_rank in range(start_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
        
        # Try random search for this rank
        found = _random_search_for_rank(n, m, p, target_rank, tensor, deadline)
        if found is not None:
            best_rank = target_rank
            best_result = found
            break  # Found a valid decomposition, move to lower rank
    
    return best_rank, best_result

def _random_search_for_rank(n, m, p, target_rank, tensor, deadline, max_iters=1000):
    """Random search for a rank-r decomposition with integer/half-integer entries."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Values to try: integers and half-integers in a small range
    values = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Generate random U, V, W with small integer/half-integer entries
        U = [[random.choice(values) for _ in range(target_rank)] for _ in range(n_m)]
        V = [[random.choice(values) for _ in range(target_rank)] for _ in range(m_p)]
        W = [[random.choice(values) for _ in range(target_rank)] for _ in range(p_n)]
        
        # Convert to float for verification
        U_f = np.array([[float(x) for x in row] for row in U])
        V_f = np.array([[float(x) for x in row] for row in V])
        W_f = np.array([[float(x) for x in row] for row in W])
        
        recon = np.zeros((n_m, m_p, p_n), dtype=np.float64)
        for r in range(target_rank):
            recon += np.einsum('a,b,c->abc', U_f[:, r], V_f[:, r], W_f[:, r])
        
        if np.max(np.abs(recon - tensor)) < 1e-9:
            # Found a valid decomposition - convert back to fractions
            return {"U": U, "V": V, "W": W}
    
    return None

def _structured_rank_search(n, m, p, deadline):
    """Try structured approaches to find lower rank decompositions."""
    tensor = _build_tensor(n, m, p)
    schoolbook = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_result = schoolbook
    
    # Try merging terms in schoolbook decomposition
    # The key insight: some terms might be linearly dependent
    
    # For now, try a simple approach: check if we can reduce by 1
    # This is a placeholder for more sophisticated structured search
    
    return best_rank, best_result

def construct():
    deadline = time.time() + 145  # Leave 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best_rank = n * m * p
    best_result = _schoolbook_decomposition(n, m, p)
    
    # Try to reduce rank
    if time.time() < deadline - 10:
        new_rank, new_result = _try_rank_reduction(n, m, p, best_rank, deadline - 10)
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = new_result
    
    return best_result

# EVOLVE-BLOCK-END
