"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition search."""
DIMENSIONS = [4, 5, 5]

from fractions import Fraction
import itertools
import random
import sys

def construct():
    n, m, p = DIMENSIONS
    # Trivial schoolbook as fallback
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using random search with exact verification
    # The tensor T[i,j,k] = 1 if we multiply (i,j) element by (j,k) element
    # We need: T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product sum)
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    
    # Known: rank >= max(n*m, m*p, p*n) is not quite right; 
    # Lower bound: rank >= max(n*p, m*p, n*m) / something... 
    # Actually for matrix multiplication, rank >= n*p (from fixing j and looking at i,k)
    # More carefully: the tensor has slices that are rank m, so rank >= m * max(n,p) roughly
    # For 4x5x5: lower bound is at least 20 (5*4), possibly higher
    
    # Let's try random search for ranks from a reasonable lower bound up to trivial
    # Use exact rational arithmetic for verification
    
    def verify_decomposition(U, V, W, rank):
        """Check if sum_r U[:,r]*V[:,r]*W[:,r] equals the matrix multiplication tensor."""
        # T[i*m+j, j*p+k] should equal 1 for the (i,j,k) term
        # Actually T is a 3D tensor: T[i,j,k] = 1
        # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    # Random search: generate random rational matrices and check
    # This is expensive, so we limit the search
    
    best = None
    # Try a few ranks below the trivial one
    # Theoretical lower bound for 4x5x5: at least 20 (from 5*4)
    # Let's try ranks from 20 to min(60, rank_trivial - 1)
    # But random search over rationals is hard. Let's use a structured approach.
    
    # Alternative: use the fact that small matrix multiplications have known optimal ranks
    # For 4x4x4, optimal is 49 (Strassen)
    # For 4x5x5, we don't have a well-known optimal, but let's try to find something
    
    # Given time constraints, let's do a limited random search with small integer entries
    random.seed(42)
    
    # Try ranks from a lower bound up to trivial-1, but limit to feasible range
    # Lower bound: rank >= ceil(n*p) = 20, also rank >= ceil(m*p)=25, rank >= ceil(n*m)=20
    # So rank >= 25
    lower_bound = max(n * p, m * p, n * m)  # = 25
    upper_bound = min(rank_trivial - 1, 50)  # Don't go too high
    
    # For each candidate rank, do a few random trials
    for rank in range(lower_bound, upper_bound + 1):
        for trial in range(50):
            # Generate random matrices with small integer entries
            U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(p * n)]
            
            if verify_decomposition(U, V, W, rank):
                best = (rank, U, V, W)
                break
        if best is not None:
            break
    
    if best is not None:
        rank, U, V, W = best
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to trivial
        return {"U": U_trivial, "V": V_trivial, "W": W_trivial}
