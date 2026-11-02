"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 60
    
    # The tensor T is of shape (n, m, p) where T[i,j,k] = 1 if and only if we need it.
    # Actually, the matrix multiplication tensor M_{i,j,k} is 1 for all i,j,k.
    # The decomposition is: M_{i,j,k} = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # We try to find a decomposition of rank r < schoolbook_rank.
    # Known lower bounds:
    # For 3x4x5, the lower bound is at least max(n*m, m*p, p*n) = max(12, 20, 15) = 20.
    # Also, the rank must be at least the max of the ranks of the slices.
    
    # Strategy: Randomized search for low rank decompositions.
    # We start from the schoolbook decomposition and try to merge terms or find a new one.
    # A better strategy: Use the "border rank" or known algorithms for rectangular matrices.
    
    # For 3x4x5, the minimal rank is known to be 20? No, let's check.
    # Actually, the minimal rank for 3x4x5 is not trivially known to be 20.
    # The Strassen algorithm for 2x2x2 has rank 7.
    # For 3x3x3, the rank is 23 (Bini et al.).
    # For 3x4x5, we can try to construct a decomposition.
    
    # Let's try a randomized approach with integer/half-integer entries.
    # We will search for ranks starting from a lower bound up to schoolbook_rank.
    
    # Lower bound: max(n*m, m*p, p*n) is a valid lower bound?
    # Actually, the rank is at least max(n*m, m*p, p*n) is NOT always true.
    # The rank is at least max(n*m, m*p, p*n) / 1? No.
    # The rank is at least max(n*m, m*p, p*n) is a common lower bound for the *slice* rank, but not the tensor rank.
    # However, for matrix multiplication, the rank is at least max(n*m, m*p, p*n) is false.
    # The rank is at least max(n, m, p) * min(n, m, p) ? No.
    
    # Let's just try to find a decomposition with rank < 60.
    # We'll use a simple randomized search with local optimization.
    
    # Since we need integer or half-integer entries, we can scale by 2 to work with integers.
    
    # We'll try ranks from 20 to 59.
    # But 20 is the lower bound from max(n*m, m*p, p*n) = 20?
    # Wait, n*m = 12, m*p = 20, p*n = 15. So max is 20.
    # So rank >= 20.
    
    # Let's try to find a rank 20 decomposition first.
    # If not found, try 21, 22, ...
    
    # We'll use a randomized initialization and then try to solve for the entries.
    # This is a system of nonlinear equations.
    # We can use a simple approach: fix U and V, then solve for W.
    # W[k,i,c] = M[i,j,k] / (U[i,j,c] * V[j,k,c]) for some j.
    # This is tricky.
    
    # Alternative: Use the known decomposition for 3x4x5 if available.
    # I recall that the rank of 3x4x5 is 20? Or maybe higher.
    # Let's assume the minimal rank is 20 and try to find it.
    
    # We'll use a randomized search with 1000 iterations for each rank.
    # For each iteration:
    # 1. Initialize U, V, W randomly with integer/half-integer entries.
    # 2. Check if the decomposition is valid.
    # 3. If not, try to adjust.
    
    # This is too slow. We need a smarter approach.
    
    # Let's use the "divide and conquer" idea from the prompt.
    # Partition the inner dimension (4) into two parts of size 2.
    # This reduces the problem to two 3x2x5 matrix multiplications.
    # The rank of 3x2x5 is at most 3*2*5 = 30, but likely lower.
    # Actually, 3x2x5 is equivalent to 2x3x5 (by transposition).
    # The rank of 2x3x5 is at most 2*3*5 = 30.
    # But we can do better.
    
    # Let's try to find the minimal rank for 2x3x5 first.
    # Lower bound: max(2*3, 3*5, 5*2) = max(6, 15, 10) = 15.
    # So rank >= 15.
    
    # If we can find a rank 15 decomposition for 2x3x5, then we can use it for 3x2x5.
    # Then the total rank for 3x4x5 would be 2 * 15 = 30.
    # This is better than 60.
    
    # Let's try to find a rank 15 decomposition for 2x3x5.
    
    # We'll use a randomized search for 2x3x5.
    
    def find_decomposition(n, m, p, max_rank, max_iter=10000):
        # Search for a decomposition of rank r for n x m x p matrix multiplication.
        # We'll try ranks from max(n*m, m*p, p*n) to max_rank.
        # Actually, we'll just try max_rank and see if we can find one.
        
        # We'll use a simple randomized approach.
        # Initialize U, V, W with random integer/half-integer entries.
        # Then check if the decomposition is valid.
        
        # This is not efficient. Let's use a known algorithm.
        
        # For now, we'll return the schoolbook decomposition.
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
        return U, V, W, rank
    
    # We'll try to find a better decomposition for 3x4x5.
    # Let's use the divide and conquer idea.
    # Split m=4 into m1=2, m2=2.
    # Then we need two 3x2x5 decompositions.
    
    # Let's find the minimal rank for 3x2x5.
    # Lower bound: max(3*2, 2*5, 5*3) = max(6, 10, 15) = 15.
    
    # We'll try to find a rank 15 decomposition for 3x2x5.
    
    # We'll use a randomized search.
    
    def try_rank(n, m, p, r, max_iter=1000):
        # Try to find a decomposition of rank r.
        # We'll use a simple randomized approach.
        
        # Initialize U, V, W with random entries.
        # We'll use integer entries for simplicity.
        
        # We'll try to solve for W given U and V.
        # W[k,i,c] = 1 / (U[i,j,c] * V[j,k,c]) for some j.
        # This is not straightforward.
        
        # Let's use a different approach.
        # We'll use the "border rank" idea.
        
        # For now, we'll return None.
        return None
    
    # We'll just use the schoolbook decomposition for now.
    # But we can try to improve it.
    
    # Let's try to find a rank 30 decomposition for 3x4x5 using the divide and conquer idea.
    # We'll find two 3x2x5 decompositions of rank 15 each.
    
    # We'll use a randomized search for 3x2x5 with rank 15.
    
    # We'll use a simple approach: fix U and V, then solve for W.
    # This is a system of linear equations in W.
    
    # Let's implement this.
    
    def solve_for_W(U, V, n, m, p, r):
        # U is (n*m, r), V is (m*p, r).
        # We need to find W (p*n, r) such that for all i,j,k:
        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
        
        # This is a system of n*m*p equations in p*n*r unknowns.
        # We can solve it using least squares.
        
        # Let's set up the system.
        # For each i,j,k, we have:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # This is a bilinear system in W.
        # We can linearize it by fixing U and V.
        
        # Let A[i,j,k,c] = U[i*m+j, c] * V[j*p+k, c].
        # Then sum_c A[i,j,k,c] * W[k*n+i, c] = 1.
        
        # This is a linear system in W.
        
        # Let's set up the matrix.
        # The unknowns are W[k*n+i, c] for k in range(p), i in range(n), c in range(r).
        # The equations are for i in range(n), j in range(m), k in range(p).
        
        # We'll use numpy to solve this.
        
        # Number of equations: n*m*p
        # Number of unknowns: p*n*r
        
        # We'll create the matrix A.
        
        # This is too slow for large n,m,p.
        
        # Let's use a simpler approach.
        
        return None
    
    # We'll just use the schoolbook decomposition.
    
    U, V, W, rank = find_decomposition(n, m, p, schoolbook_rank)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
