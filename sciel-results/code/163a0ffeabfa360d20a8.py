"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    We check that this equals 1 for all i,j,k (since T is the matrix multiplication tensor).
    """
    rank = len(U[0])
    # Build tensor T of shape (n, m, p)
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = np.array([U[i*m+j][c] for i in range(n) for j in range(m)]).reshape(n, m)
        vc = np.array([V[j*p+k][c] for j in range(m) for k in range(p)]).reshape(m, p)
        wc = np.array([W[k*n+i][c] for k in range(p) for i in range(n)]).reshape(p, n)
        # Outer product structure: T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    # Check if T is all ones
    return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Exact verification using Fraction to avoid floating point issues."""
    rank = len(U[0])
    # We need to check that for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    # And that the decomposition is correct.
    
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(U[i][c]) for c in range(rank)] for i in range(n*m)]
    V_frac = [[Fraction(V[j][c]) for c in range(rank)] for j in range(m*p)]
    W_frac = [[Fraction(W[k][c]) for c in range(rank)] for k in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                if s != 1:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _try_strong_rank_decomposition():
    """
    Try to find a rank-minimal decomposition using the known algebraic structure.
    For 3x4x7 matrix multiplication, we try ranks from the theoretical lower bound up.
    The lower bound is max(n*m, m*p, p*n) = max(12, 28, 21) = 28.
    We try to find a decomposition of rank r for r from 28 upwards.
    """
    n, m, p = DIMENSIONS
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower_bound to n*m*p
    for r in range(lower_bound, n*m*p + 1):
        # For each rank, try to find a valid decomposition
        # This is a complex search, so we use a structured approach
        
        # Initialize with random values
        np.random.seed(42 + r)
        for trial in range(50):
            # Try a structured initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Round to nearest half-integer
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            # Verify
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                # Convert to list of lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_iterative_truncation():
    """
    Use iterative truncation of the tensor unfolding to find a low-rank approximation,
    then refine to integer/half-integer solutions.
    """
    n, m, p = DIMENSIONS
    rank_full = n * m * p
    
    # Start with schoolbook
    schoolbook = _schoolbook()
    
    # Try to reduce rank by finding linear dependencies
    # This is a heuristic approach
    
    # For now, return schoolbook if no better is found
    return schoolbook

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 7 matrix.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook decomposition as a fallback
    schoolbook = _schoolbook()
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try to find a better decomposition
    # Strategy 1: Try structured low-rank approaches
    if time.time() < deadline:
        result = _try_strong_rank_decomposition()
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    # Strategy 2: Try iterative truncation
    if time.time() < deadline:
        result = _try_iterative_truncation()
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    return best_result

# EVOLVE-BLOCK-END
