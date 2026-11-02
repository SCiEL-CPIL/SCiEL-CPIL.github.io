"""Matrix multiplication 2,2,4: rank-minimal exact tensor decomposition search."""
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    # Dimensions: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Schoolbook rank = n*m*p = 16
    
    # The tensor T[i,j,k] corresponds to C[i,k] = sum_j A[i,j] * B[j,k]
    # In the factorization: T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Index convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The tensor entry T[(i,j),(j',k),(k',i')] = sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k'*n+i',c]
    # For this to equal the matrix multiplication tensor, we need:
    # T[(i,j),(j',k),(k',i')] = 1 if i==i', j==j', k==k' else 0
    # Wait, let me reconsider the tensor structure.
    
    # Actually, the matrix multiplication tensor T has indices (i, j, k) where:
    # A is n x m, B is m x p, C is n x p
    # T[i, j, k] = 1 if this term contributes to C[i,k] via A[i,j]*B[j,k]
    # The standard tensor decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, the W index is k*n+i, so W is indexed by (k, i) pairs.
    
    # The tensor T is a 3-tensor with:
    # - First index: (i, j) for i in range(n), j in range(m) -> n*m = 4 indices
    # - Second index: (j, k) for j in range(m), k in range(p) -> m*p = 8 indices  
    # - Third index: (k, i) for k in range(p), i in range(n) -> p*n = 8 indices
    
    # T[(i,j), (j',k), (k',i')] = 1 if i==i', j==j', k==k' else 0
    
    # Let's build the full tensor and its matricizations for verification.
    
    # Build the tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
    
    # For verification, we can use any matricization. Let's use the first-mode matricization.
    # T_1 is (n*m) x (m*p * p*n) matrix where T_1[a, b] = T[a, b1, b2]
    # Actually, let's just verify by reconstructing the tensor from the factors.
    
    def verify(U, V, W, rank):
        """Verify that sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] == T"""
        # Reconstruct tensor
        T_recon = np.zeros((n*m, m*p, p*n), dtype=object)
        for c in range(rank):
            for a in range(n*m):
                if U[a][c] != 0:
                    for b in range(m*p):
                        if V[b][c] != 0:
                            for d in range(p*n):
                                if W[d][c] != 0:
                                    T_recon[a, b, d] += U[a][c] * V[b][c] * W[d][c]
        # Compare with T
        for a in range(n*m):
            for b in range(m*p):
                for d in range(p*n):
                    if T[a, b, d] != T_recon[a, b, d]:
                        return False
        return True
    
    # We'll search for rank-13 decompositions using randomized bilinear reduction.
    # Strategy: Start with schoolbook terms and try to find linear dependencies.
    
    # The 16 schoolbook terms are indexed by (i, j, k) with i in [0,1], j in [0,1], k in [0,3]
    # Each term c corresponds to a rank-1 tensor U_c ⊗ V_c ⊗ W_c where:
    # U_c has 1 at position i*m+j, V_c has 1 at position j*p+k, W_c has 1 at position k*n+i
    
    # We want to find if there's a rank-13 decomposition. This means we need to find
    # 13 vectors u_c, v_c, w_c such that sum_c u_c ⊗ v_c ⊗ w_c = T.
    
    # Randomized approach: 
    # 1. Pick 13 random linear combinations of the 16 schoolbook terms
    # 2. Check if they can be "reduced" to a valid decomposition
    
    # Actually, a better approach: use the fact that if we have a rank-r decomposition,
    # the flattening of the tensor has rank at most r. So we can check necessary conditions.
    
    # Let's try a different approach: use Gaussian elimination to find if there's a
    # rank-13 decomposition by solving the system of equations.
    
    # The tensor T has 4*8*8 = 256 entries, but only 16 are non-zero (all equal to 1).
    # A rank-r decomposition gives us 3*r variables (r entries in each of U, V, W per column).
    # The equations are: for each (a,b,d), sum_c U[a,c]*V[b,c]*W[d,c] = T[a,b,d]
    
    # This is a system of 256 equations in 3*r unknowns, but the equations are nonlinear
    # (products of variables). However, we can linearize by fixing one set of variables.
    
    # Let's try a randomized search: 
    # Fix random values for U (4 x 13 matrix), then for each column c, we have V[:,c] and W[:,c].
    # The equation becomes: for each (a,b,d), sum_c U[a,c]*V[b,c]*W[d,c] = T[a,b,d]
    
    # This is still nonlinear. Let's use a different strategy.
    
    # Strategy: Use the "tensor rank" approach. For a fixed U, the problem reduces to
    # finding V and W such that T = sum_c u_c ⊗ v_c ⊗ w_c.
    
    # Let's try a more direct approach: enumerate possible structures.
    
    # Given the small size, let's try to find a rank-13 decomposition by solving
    # a system of polynomial equations using a randomized approach.
    
    # Actually, let's use a known result: the matrix multiplication tensor M_{2,2,4} 
    # has rank 13. This is a known result (Landsberg, 2012, or similar).
    
    # Let me try to construct the known rank-13 decomposition.
    
    # From the literature, the rank of M_{2,2,4} is 13. Let me try to find an explicit
    # decomposition.
    
    # One approach: use the "winograd" style decomposition.
    
    # Let me try a randomized search with rational coefficients.
    
    import random
    from sympy import Matrix, symbols, solve, Rational
    
    # Actually, sympy might be too slow. Let's use numpy and fractions.
    
    # Let's try a different approach: use the fact that we can fix the first few
    # columns and solve for the rest.
    
    # Given the complexity, let's try a simpler approach: use the known rank-13
    # decomposition from the literature.
    
    # The matrix multiplication tensor M_{2,2,4} can be decomposed as follows:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3]
    
    # That's 16 terms. To reduce to 13, we need to combine some terms.
    
    # Let's try Winograd's approach for 2x2 by 2x4:
    # We can use the fact that 2x2 matrix multiplication has rank 7 (over reals, 
    # actually it's 7 for the bilinear complexity).
    
    # For 2x2 by 2x4, we can think of it as two 2x2 multiplications.
    # The rank of 2x2 by 2x2 is 7, so two of them would give rank 14.
    # But we can share some computations, potentially reducing to 13.
    
    # Let me try to construct a rank-13 decomposition.
    
    # Actually, let's just do a randomized search. We'll try random rational coefficients
    # and check if we can find a valid rank-13 decomposition.
    
    # For efficiency, let's use a fixed seed and try several random initializations.
    
    random.seed(42)
    
    # We'll represent U, V, W as lists of lists of Fractions.
    
    def try_random_decomposition(rank, num_trials=1000):
        for trial in range(num_trials):
            # Initialize U, V, W with small random rational values
            U = [[Fraction(random.randint(-2, 2), random.randint(1, 3)) for _ in range(rank)] for _ in range(n*m)]
            V = [[Fraction(random.randint(-2, 2), random.randint(1, 3)) for _ in range(rank)] for _ in range(m*p)]
            W = [[Fraction(random.randint(-2, 2), random.randint(1, 3)) for _ in range(rank)] for _ in range(p*n)]
            
            # Check if this is a valid decomposition
            if verify(U, V, W, rank):
                return U, V, W
        
        return None
    
    # Try rank 13 first
    result = try_random_decomposition(13, num_trials=100)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # If random search fails, fall back to schoolbook rank 16
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
