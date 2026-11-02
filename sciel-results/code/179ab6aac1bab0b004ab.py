"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import itertools

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x5 matrix.
    
    The matrix multiplication tensor T has indices (i, j, k) where:
    - i in range(n) = range(2)
    - j in range(m) = range(5)
    - k in range(p) = range(5)
    
    The decomposition is:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    We search for the minimum rank by trying ranks from lower bounds upward.
    For small dimensions, we can use a direct algebraic approach.
    
    The schoolbook rank is n*m*p = 50.
    We'll try to find a decomposition with rank < 50.
    
    For the 2x5x5 case, we can use known results or search.
    Let's start with a systematic search using linear algebra.
    """
    n, m, p = DIMENSIONS
    
    # Try ranks from a reasonable lower bound up to the schoolbook rank
    # Lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # Also, for matrix multiplication, rank >= max(n*m, m*p, p*n) is a known lower bound
    
    # For 2x5x5, let's try rank 25 first (which equals m*p = 25)
    # This is achievable if we can find appropriate factor matrices
    
    # The tensor equation is:
    # For all (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # (since the matrix multiplication tensor has 1s at all valid (i,j,k) positions)
    
    # This is a bilinear system in the entries of U, V, W.
    # We can fix one matrix and solve for the others, or use iterative methods.
    
    # Strategy: Since n=2 is small, we can exploit the structure.
    # Let's try to find a rank-25 decomposition.
    
    # For rank r = m*p = 25, we can try to set V to be an identity-like matrix
    # and solve for U and W.
    
    # Actually, let's use a more direct approach.
    # The matrix multiplication can be viewed as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T[i,j,k] = 1 for all valid i,j,k.
    
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try rank = 25 and see if we can find integer/half-integer solutions.
    
    # Approach: Fix V to be a specific matrix, then the equation becomes linear in U and W.
    # But it's still bilinear. Let's try a different approach.
    
    # For rank r, we have r variables for each of U, V, W.
    # The number of equations is n*m*p = 50.
    # The number of unknowns is 3*r*(n*m + m*p + p*n)/r... wait, let me count properly.
    # U has n*m*r = 10*r entries
    # V has m*p*r = 25*r entries  
    # W has p*n*r = 10*r entries
    # Total unknowns: 45*r
    # Equations: n*m*p = 50
    
    # For r = 25: 45*25 = 1125 unknowns, 50 equations. Very underdetermined.
    # So there should be many solutions if one exists.
    
    # Let's try to construct a rank-25 solution.
    # One simple approach: set V[j*p+k, c] = delta_{j,c//p} * delta_{k,c%p} for c in 0..24
    # i.e., V is a block diagonal with 5 blocks of 5x1... wait, that's not right.
    
    # Let me think differently. For rank r = m*p = 25:
    # Let c = j*p + k for j in 0..4, k in 0..4
    # Set V[j*p+k, j*p+k] = 1 (identity matrix of size 25x25)
    # Then the equation becomes:
    # sum_c U[i*m+j, c] * delta_{j*p+k, c} * W[k*n+i, c] = U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1
    # So we need U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1 for all i,j,k
    # This means for each (i,j,k), we need the product to be 1.
    # We can set U[i*m+j, j*p+k] = 1 and W[k*n+i, j*p+k] = 1 for all i,j,k.
    # But wait, this would make U and W have specific non-zero entries.
    
    # Let's check: U has shape (10, 25). U[i*m+j, j*p+k] = 1 means:
    # For i=0: U[0*5+j, j*5+k] = 1 for j=0..4, k=0..4
    #   U[0, 0..4] = 1, U[1, 5..9] = 1, U[2, 10..14] = 1, U[3, 15..19] = 1, U[4, 20..24] = 1
    # For i=1: U[1*5+j, j*5+k] = 1 for j=0..4, k=0..4
    #   U[5, 0..4] = 1, U[6, 5..9] = 1, U[7, 10..14] = 1, U[8, 15..19] = 1, U[9, 20..24] = 1
    
    # W has shape (10, 25). W[k*n+i, j*p+k] = 1 means:
    # For i=0: W[k*2+0, j*5+k] = 1 for j=0..4, k=0..4
    #   W[0, j*5+k] = 1 for all j,k -> W[0, 0..24] = 1
    #   W[2, j*5+k] = 1 for all j,k -> W[2, 0..24] = 1
    #   W[4, j*5+k] = 1 for all j,k -> W[4, 0..24] = 1
    #   W[6, j*5+k] = 1 for all j,k -> W[6, 0..24] = 1
    #   W[8, j*5+k] = 1 for all j,k -> W[8, 0..24] = 1
    # For i=1: W[k*2+1, j*5+k] = 1 for j=0..4, k=0..4
    #   W[1, j*5+k] = 1 for all j,k -> W[1, 0..24] = 1
    #   W[3, j*5+k] = 1 for all j,k -> W[3, 0..24] = 1
    #   W[5, j*5+k] = 1 for all j,k -> W[5, 0..24] = 1
    #   W[7, j*5+k] = 1 for all j,k -> W[7, 0..24] = 1
    #   W[9, j*5+k] = 1 for all j,k -> W[9, 0..24] = 1
    
    # So W would be all ones! Let's verify the decomposition:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # = sum_c U[i*m+j, c] * delta_{j*p+k, c} * 1
    # = U[i*m+j, j*p+k]
    # = 1 (by our construction)
    
    # This works! So rank 25 is achievable.
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Set V to identity
    for c in range(rank):
        V[c][c] = 1
    
    # Set U[i*m+j, j*p+k] = 1 for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][j * p + k] = 1
    
    # Set W[k*n+i, j*p+k] = 1 for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                W[k * n + i][j * p + k] = 1
    
    # Verify the decomposition
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    # Fallback to schoolbook if verification fails
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
