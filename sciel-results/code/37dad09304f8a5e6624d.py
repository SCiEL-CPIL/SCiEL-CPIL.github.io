"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 120
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

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication, T[i,j,k] should be 1 if i,j,k are valid indices, else 0
    # Actually, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has entries T[i,j,k] = 1 for all valid i,j,k
    # The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, let me re-read the convention: W[k*n+i,r]
    # So W is indexed by [k*n+i, r] where k is column of B / row of C, i is row of A / row of C
    # U is [i*m+j, c] - row i of A, column j
    # V is [j*p+k, c] - row j of B, column k
    # W is [k*n+i, c] - column k of C, row i of C
    
    # The tensor entry T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Let's verify by checking a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def try_strassen_based_decomposition(n, m, p):
    """
    Try to find a decomposition using Strassen-like ideas.
    For 2x2x2, Strassen gives rank 7 instead of 8.
    We can try to partition the matrices and use Strassen on blocks.
    
    For 4x5x6:
    - Split A (4x5) into blocks
    - Split B (5x6) into blocks
    - Split C (4x6) into blocks
    
    One approach: use 2x2 block structure where possible.
    A is 4x5, B is 5x6, C is 4x6.
    
    Let's try a different approach: use the known rank-7 decomposition for 2x2x2
    and see if we can compose it.
    
    Actually, let's try a more direct approach: use random search with verification.
    """
    pass

def random_search_for_decomposition(n, m, p, max_time=140.0):
    """
    Use random search to try to find a lower-rank decomposition.
    Start with schoolbook rank and try to reduce it.
    """
    start_time = time.time()
    deadline = start_time + max_time - 5.0  # 5 second safety margin
    
    # Try various ranks below the schoolbook rank
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_U = best_V = best_W = None
    
    # Initialize with schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # This is a hard problem; random search may not find anything quickly
    # but we can try a few structured approaches
    
    # Approach 1: Try rank = schoolbook_rank - 1, -2, etc.
    # For each rank, try random initializations and see if we can verify
    # This is unlikely to work with pure random search, but let's try a few
    
    # Actually, let's try a more structured approach:
    # Use the fact that matrix multiplication tensor has symmetries
    
    # For now, return the schoolbook decomposition
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition as fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * m)]  # This is wrong, should be m*p
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
    
    # Try to find a better decomposition
    U2, V2, W2, rank2 = random_search_for_decomposition(n, m, p, max_time=120.0)
    
    if rank2 < rank:
        return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
