"""Matrix multiplication 2x3x5 rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 5]
n, m, p = DIMENSIONS
# Tensor indices: T[i,j,k] = 1 if we multiply (i,j) of A by (j,k) of B -> C[i,k]
# Decomposition: C[i,k] = sum_c U[i*m+j_c, c] * V[j_c*p+k, c] * W[k*n+i, c]
# Wait, the interface is U (n*m, rank), V (m*p, rank), W (p*n, rank)
# U index: i*m + j? Or i + m*j?
# The task says: "output factor follows the official trace convention W[k*n+i,r]"
# This implies W is indexed by (k, i) as k*n + i.
# Usually U is indexed by (i, j) and V by (j, k).
# Let's assume U[i*m + j, c], V[j*p + k, c], W[k*n + i, c].
# Then C[i,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
# But j is summed over? No, in tensor decomposition of matrix mult,
# T_{i,j,k} = sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c ?
# Actually, the standard tensor for matrix multiplication M_{n,m,p} is defined by
# T[i, j, k] = 1 if we are computing C[i,k] = sum_j A[i,j] B[j,k].
# The tensor decomposition is T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c].
# The interface flattens these:
# U has shape (n*m, rank). Index u = i*m + j? Or i + m*j?
# Let's check the schoolbook example in the prompt:
# U[i * m + j][column] = 1
# V[j * p + k][column] = 1
# W[k * n + i][column] = 1
# So U index is i*m+j, V index is j*p+k, W index is k*n+i.
# This matches my assumption.

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # 1. Try to find a decomposition of rank 28 (known bound or lower?)
    # Schoolbook rank is 2*3*5 = 30.
    # Known lower bound for 2x3x5 is ?
    # Let's try to search for rank 28, 29, etc.
    # We can use a random search + local optimization (Gauss-Newton or similar)
    # or just random search with exact rational arithmetic for small ranks.
    
    # Strategy:
    # 1. Random initialization of U, V, W with small integers.
    # 2. Check if sum_c U[:,c] V[:,c] W[:,c] == T (tensor product)
    #    The tensor T is sparse: T[i,j,k] = 1 for all i,j,k? No.
    #    T[i,j,k] is the coefficient of A[i,j] B[j,k] in the product.
    #    So T[i,j,k] = 1 for all i in 0..1, j in 0..2, k in 0..4.
    #    Wait, is that right?
    #    C[i,k] = sum_j A[i,j] B[j,k].
    #    The tensor T has indices (i,j,k) and T[i,j,k] = 1.
    #    Yes, it's the all-ones tensor of size 2x3x5.
    
    # So we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # This is a system of nonlinear equations.
    # Rank 30 is trivial.
    # Can we do rank 29?
    # Can we do rank 28?
    
    # Let's try a simple random search for rank 28, 29, 30.
    # Since the numbers are small, we can use integers.
    # We can also try to use the known decomposition for 2x3x3 (rank 7) and extend?
    # 2x3x3 rank is 7 (Strassen-like).
    # 2x3x5 might be 7 + 2*2? No.
    
    # Let's just do a random search for rank 28.
    # If it fails, try 29, then 30.
    
    def verify(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
        # We can compute the tensor product and check.
        # For speed, we can check a few random indices first, then all.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def random_init(rank, max_val=2):
        U = [[random.randint(-max_val, max_val) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.randint(-max_val, max_val) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.randint(-max_val, max_val) for _ in range(rank)] for _ in range(p*n)]
        return U, V, W

    # Try ranks from 28 upwards
    for rank in range(28, 31):
        if time.time() > deadline:
            break
        # Try a few random attempts
        attempts = 1000 if rank == 28 else 100
        for _ in range(attempts):
            if time.time() > deadline:
                break
            U, V, W = random_init(rank)
            if verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
        
        # If random fails, try a structured search?
        # Or just move to next rank.
        
    # Fallback to schoolbook rank 30
    rank = 30
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
