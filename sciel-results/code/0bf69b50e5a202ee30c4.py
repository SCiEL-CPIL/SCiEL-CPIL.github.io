"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    Uses Strassen's algorithm as a structural basis (rank 35) and attempts
    global rank reduction via integer relation search.
    """
    n, m, p = DIMENSIONS
    
    # Start with rank-35 Strassen-based decomposition
    U, V, W = _construct_strassen_rank35()
    rank = len(U[0])
    
    # Attempt global rank reduction with time guard
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for return
    
    U, V, W, rank = _reduce_rank(U, V, W, deadline)
    
    return {"U": U, "V": V, "W": W}

def _construct_strassen_rank35():
    """Constructs rank-35 decomposition using Strassen's algorithm for each 2x2 block."""
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    strassen_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    w_contrib = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            U[0][col] = strassen_A[r][0]
            U[1][col] = strassen_A[r][1]
            U[2][col] = strassen_A[r][2]
            U[3][col] = strassen_A[r][3]
            
            k0 = 2 * b
            k1 = 2 * b + 1
            V[0 * p + k0][col] = strassen_B[r][0]
            V[0 * p + k1][col] = strassen_B[r][1]
            V[1 * p + k0][col] = strassen_B[r][2]
            V[1 * p + k1][col] = strassen_B[r][3]
            
            W[(2 * b) * 2 + 0][col] = w_contrib[r][0]
            W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]
            W[(2 * b) * 2 + 1][col] = w_contrib[r][2]
            W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]
    
    return U, V, W

def _reduce_rank(U, V, W, deadline):
    """Greedy rank reduction by dropping terms that are redundant, plus PSLQ-based search."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # First try simple greedy removal
    while rank > 20:
        if time.time() > deadline - 10:
            break
        
        reduced = False
        for drop_idx in range(rank):
            if time.time() > deadline - 10:
                break
            
            U_new = [row[:drop_idx] + row[drop_idx+1:] for row in U]
            V_new = [row[:drop_idx] + row[drop_idx+1:] for row in V]
            W_new = [row[:drop_idx] + row[drop_idx+1:] for row in W]
            
            if _verify_fast(U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                reduced = True
                break
        
        if not reduced:
            break
    
    # Try PSLQ-based rank reduction: search for linear dependencies among rank-1 terms
    if time.time() < deadline - 5:
        U, V, W, rank = _pslq_reduce(U, V, W, deadline)
    
    return U, V, W, rank

def _pslq_reduce(U, V, W, deadline):
    """
    Uses PSLQ algorithm to find integer relations among the rank-1 terms.
    If we find a relation, we can potentially eliminate one term.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # For each pair of terms (r1, r2), check if their sum/difference can be
    # represented as a single rank-1 term
    # This is a heuristic search
    
    # Convert to numpy for faster operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to find if any term can be expressed as a linear combination of others
    # by checking if the tensor is in the span of the remaining terms
    
    improved = False
    for drop_idx in range(rank):
        if time.time() > deadline - 3:
            break
        
        # Check if term drop_idx is in the span of all other terms
        if _is_term_in_span(U_np, V_np, W_np, drop_idx):
            U_np = np.delete(U_np, drop_idx, axis=1)
            V_np = np.delete(V_np, drop_idx, axis=1)
            W_np = np.delete(W_np, drop_idx, axis=1)
            rank -= 1
            improved = True
            break
    
    if improved:
        # Convert back to lists
        U = U_np.tolist()
        V = V_np.tolist()
        W = W_np.tolist()
        # Round to integers if close
        U = [[round(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U]
        V = [[round(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V]
        W = [[round(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W]
    
    return U, V, W, rank

def _is_term_in_span(U_np, V_np, W_np, drop_idx):
    """
    Checks if the rank-1 term at index drop_idx is in the span of all other terms.
    This is done by checking if the tensor T_drop is in the column space of
    the matrix formed by the other tensors.
    """
    n, m, p = DIMENSIONS
    rank = U_np.shape[1]
    
    if rank <= 1:
        return False
    
    # Create indices for all other terms
    other_indices = [i for i in range(rank) if i != drop_idx]
    
    # The tensor for term r is: T_r[i, j, k] = U[i, r] * V[j, r] * W[k, r]
    # We want to check if T_drop = sum_{r in other} c_r * T_r
    
    # This is a linear system. Let's flatten the tensors.
    # Dimension of each tensor: n * m * p = 2 * 2 * 10 = 40
    
    # Create the tensor for the dropped term
    T_drop = np.zeros(n * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i * m + j) * p + k
                T_drop[idx] = U_np[i * m + j, drop_idx] * V_np[j * p + k, drop_idx] * W_np[k * n + i, drop_idx]
    
    # Create the matrix of other tensors
    A = np.zeros((n * m * p, len(other_indices)))
    for col, r in enumerate(other_indices):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + k
                    A[idx, col] = U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
    
    # Solve the least squares problem: A * c = T_drop
    # Check if the residual is close to zero
    try:
        c, residuals, rank_A, s = np.linalg.lstsq(A, T_drop, rcond=None)
        residual = np.linalg.norm(A @ c - T_drop)
        norm_T = np.linalg.norm(T_drop)
        
        if norm_T > 1e-10:
            return residual / norm_T < 1e-8
        else:
            # T_drop is zero, so it's trivially in the span
            return True
    except:
        return False

def _verify_fast(U, V, W):
    """Verify decomposition using numpy for speed."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for i_prime in range(n):
            for k in range(p):
                for k_prime in range(p):
                    M = np.zeros((m, m))
                    for j in range(m):
                        for j_prime in range(m):
                            M[j, j_prime] = np.sum(
                                W_np[k*n+i] * U_np[i_prime*m + j] * V_np[j_prime*p + k_prime]
                            )
                    
                    if i == i_prime and k == k_prime:
                        if not np.allclose(M, np.eye(m)):
                            return False
                    else:
                        if not np.allclose(M, np.zeros((m, m))):
                            return False
    return True

def _verify_decomposition(U, V, W):
    """
    Verifies that the tensor decomposition exactly computes matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for i_prime in range(n):
            for k in range(p):
                for k_prime in range(p):
                    M = np.zeros((m, m))
                    for r in range(rank):
                        w_val = W[k * n + i][r]
                        if w_val == 0:
                            continue
                        for j in range(m):
                            u_val = U[i_prime * m + j][r]
                            if u_val == 0:
                                continue
                            for j_prime in range(m):
                                v_val = V[j_prime * p + k_prime][r]
                                if v_val == 0:
                                    continue
                                M[j, j_prime] += w_val * u_val * v_val
                    
                    if i == i_prime and k == k_prime:
                        if not np.allclose(M, np.eye(m)):
                            return False
                    else:
                        if not np.allclose(M, np.zeros((m, m))):
                            return False
    return True

# EVOLVE-BLOCK-END
