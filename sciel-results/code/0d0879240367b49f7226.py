import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy broadcasting."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Exact verification using Fraction."""
    rank = U.shape[1]
    U_f = [[Fraction(U[i, c]) for c in range(rank)] for i in range(n * m)]
    V_f = [[Fraction(V[j, c]) for c in range(rank)] for j in range(m * p)]
    W_f = [[Fraction(W[k, c]) for c in range(rank)] for k in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U_f[i * m + j][c] * V_f[j * p + k][c] * W_f[k * n + i][c]
                if total != Fraction(1):
                    return False
    return True

def _try_rank_rational(n, m, p, target_rank, deadline, max_cols=20):
    """Try to find a rational decomposition of given rank by solving linear system.
    
    We fix a subset of entries to 1 and solve for the rest.
    """
    # Total unknowns: (n*m + m*p + p*n) * target_rank
    total_unknowns = (n * m + m * p + p * n) * target_rank
    # Total equations: n * m * p
    total_eqs = n * m * p
    
    if total_unknowns < total_eqs:
        return None
    
    # We need to fix enough entries to make the system square or overdetermined
    # Actually, we have more unknowns than equations, so the system is underdetermined.
    # We can set some variables to 0 or 1 to reduce the search space.
    
    # Strategy: Try to find a solution where most entries are 0 or 1.
    # This is equivalent to finding a combinatorial design.
    
    # For now, let's try a random sparse pattern and solve.
    # We'll use a simple approach: random initialization with many zeros.
    
    np.random.seed(int(time.time() * 1000) % 10000)
    
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
        
        # Create sparse random matrices
        density = 0.1
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        for i in range(n * m):
            for c in range(target_rank):
                if np.random.random() < density:
                    U[i, c] = 1
        for j in range(m * p):
            for c in range(target_rank):
                if np.random.random() < density:
                    V[j, c] = 1
        for k in range(p * n):
            for c in range(target_rank):
                if np.random.random() < density:
                    W[k, c] = 1
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
            return U, V, W
    
    return None

def _try_known_decompositions(n, m, p, deadline):
    """Try known efficient decompositions for small matrix multiplication."""
    
    # For 4x5x5, the schoolbook rank is 100.
    # Known results suggest rank can be lower, but exact minimal rank is not well-known.
    # Let's try some structured patterns.
    
    # Pattern 1: Use Strassen-like ideas
    # For 2x2x2, Strassen uses rank 7 instead of 8.
    # For larger matrices, we can try block decomposition.
    
    # Try a block decomposition where we treat the 4x5x5 as blocks
    # This is complex, so let's try a simpler approach: random search with verification.
    
    candidate_ranks = [99, 98, 97, 96, 95, 94, 93, 92, 91, 90]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_rational(n, m, p, rank, deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W, rank
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school
    
    # Try to find better decompositions
    result = _try_known_decompositions(n, m, p, deadline)
    if result is not None:
        U_try, V_try, W_try, rank_try = result
        if rank_try < best_rank:
            best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
            best_rank = rank_try
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
