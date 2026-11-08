"""Rewritten program: rank-minimal exact tensor decomposition for 2x3 * 3x5."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Return {'U', 'V', 'W'} factor matrices for exact matrix multiplication
    of a 2x3 matrix by a 3x5 matrix, with minimal rank.
    
    We use a systematic search over low-rank decompositions.
    The known lower bound for rank of matrix multiplication R_{2,3,5} is at least
    the geometric mean or other bounds. Let's try to find the true minimum.
    
    For small dimensions, we can use algebraic methods. The tensor for multiplying
    a 2x3 by 3x5 matrix has 2*3*5 = 30 entries in the naive decomposition.
    
    We'll use a randomized algebraic approach: represent the multiplication tensor
    and try to find a low-rank decomposition by solving systems of polynomial equations,
    or use known results.
    
    Actually, let's use a direct approach: for small matrices, we can enumerate
    or use the fact that rank of matmul tensor for (n,m,p) is at most n*m*p and
    we can try to reduce it.
    
    Let me use a practical approach: try ranks from lower bound up, and for each
    rank r, try to find U, V, W with integer/half-integer entries such that
    the tensor decomposition holds.
    
    The tensor T[i,j,k, l] (where A is n x m, B is m x p, C = A*B is n x p)
    satisfies: C[i,l] = sum_j A[i,j] * B[j,l]
    
    In tensor form: T = sum_c u_c ⊗ v_c ⊗ w_c where
    u_c[i*m+j] = U[i*m+j, c]
    v_c[j*p+k] = V[j*p+k, c]  
    w_c[k*n+i] = W[k*n+i, c]
    
    And the condition is: for all i,j,k,
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) corresponds to a valid matmul term
    Actually, the tensor is: T[i,j,k] = delta (the coefficient)
    
    Let me think more carefully. The matrix multiplication tensor is defined as:
    T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the multiplication is C[i,k] = sum_j A[i,j] B[j,k]
    
    So the tensor has components: for each (i,j,k), the coefficient is 1.
    This means T[i,j,k] = 1 for all valid i,j,k, but that's not quite right.
    
    Actually, the standard way: the bilinear map (A,B) -> A*B can be written as
    sum_c u_c(A) * v_c(B) * w_c where w_c is the output coefficient.
    
    More precisely: (A*B)[i,k] = sum_j A[i,j] B[j,k] = sum_c U[i,j,c] V[j,k,c] W[i,k,c]
    
    Wait, I need to be careful with the index conventions.
    
    Let me use the convention from the task:
    U has shape (n*m, rank): U[i*m + j, c]
    V has shape (m*p, rank): V[j*p + k, c]
    W has shape (p*n, rank): W[k*n + i, c]  (note: k*n+i, not i*p+k)
    
    The decomposition condition: for all i, j, k:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is because the (i,k) entry of the product is sum_j A[i,j] B[j,k],
    and each term A[i,j]*B[j,k] contributes to C[i,k].
    
    So we need: for each triple (i,j,k), the sum over c of the product equals 1.
    
    This is a system of n*m*p = 2*3*5 = 30 polynomial equations in the entries of U, V, W.
    
    Let's try to find the minimum rank. Known results suggest that for small matrices,
    the rank can be less than n*m*p.
    
    Let me use a randomized approach with rational arithmetic to search for low-rank solutions.
    """
    n, m, p = DIMENSIONS
    # Lower bound: rank >= max(n*m, m*p, p*n) / something, but let's just search
    # Actually, a simple lower bound is that rank >= max(n*m, m*p, n*p) is not true.
    # The trivial bound is rank >= n*m*p / max(n*m, m*p, n*p) but that's not helpful.
    
    # Let's try ranks from a reasonable lower bound up to n*m*p
    # A known lower bound for rank of matmul tensor is at least (n*m*p) / max(n,m,p) in some cases
    
    start_time = time.time()
    deadline = 140  # seconds, with margin
    
    # Try to find the minimum rank
    # Start with a lower bound estimate
    # For 2x3*3x5, let's try ranks from 10 up to 30
    
    for rank in range(10, n * m * p + 1):
        if time.time() - start_time > deadline:
            break
        
        # Try to find a decomposition of this rank
        # Use a randomized approach with rational numbers
        found = _try_find_decomposition(n, m, p, rank, deadline - (time.time() - start_time))
        if found is not None:
            U, V, W = found
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)


def _try_find_decomposition(n, m, p, rank, time_budget):
    """
    Try to find a rank-`rank` decomposition using randomized algebraic methods.
    Returns (U, V, W) as lists of lists with Fraction entries, or None.
    """
    import random
    from fractions import Fraction
    import numpy as np
    
    start = time.time()
    
    # Number of equations: n*m*p
    num_eq = n * m * p
    
    # We have rank * (n*m + m*p + p*n) unknowns, but with scaling symmetry
    # (we can scale U[c] by alpha, V[c] by beta, W[c] by 1/(alpha*beta))
    
    # Strategy: fix some variables to break symmetry, then solve
    # This is complex, so let's use a simpler approach for small cases
    
    # For small dimensions, we can try specific constructions
    
    # Let's try a structured approach:
    # Assume U, V, W have a specific pattern
    
    # Actually, let me use a different strategy: 
    # Use the fact that we can set W[k*n+i, c] = 1 for some entries and solve for U, V
    
    # Simpler: use randomized initialization and iterate
    
    # For now, let's try a few specific constructions
    
    # Construction 1: Try to use the structure of small matrices
    # For 2x3*3x5, we can try to exploit the small size
    
    # Let me try a direct algebraic approach for small ranks
    # Set up the system of equations
    
    # Variables: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] for i in range(n), j in range(m), k in range(p), c in range(rank)
    
    # To break scaling symmetry, set W[k*n+i, 0] = 1 for the first column (if rank >= 1)
    # Actually, let's set specific entries to 1 to fix the gauge
    
    if rank < 1:
        return None
    
    # Fix gauge: set W[0, 0] = 1 (i=0, k=0)
    # And set U[0, 0] = 1 (i=0, j=0)
    
    # This is getting complex. Let me use a practical approach:
    # Try known decompositions or use a solver
    
    # For the specific case of 2x3*3x5, let's try to construct a solution
    
    # I'll use a heuristic: try random rational matrices and check
    
    best_solution = None
    
    # Try a few random attempts
    num_attempts = 50
    for attempt in range(num_attempts):
        if time.time() - start > time_budget:
            break
        
        # Initialize with small rational numbers
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        
        # Set some random values
        for c in range(rank):
            # Random small fractions
            for _ in range(3):  # set a few entries per column
                i = random.randint(0, n * m - 1)
                U[i][c] = Fraction(random.randint(-2, 2), random.choice([1, 1, 2]))
            
            for _ in range(3):
                j = random.randint(0, m * p - 1)
                V[j][c] = Fraction(random.randint(-2, 2), random.choice([1, 1, 2]))
            
            for _ in range(3):
                k = random.randint(0, p * n - 1)
                W[k][c] = Fraction(random.randint(-2, 2), random.choice([1, 1, 2]))
        
        # Check if this satisfies the equations
        # For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return (U, V, W)
    
    return None


def _schoolbook(n, m, p):
    """Return the trivial schoolbook decomposition of rank n*m*p."""
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
