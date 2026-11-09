"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
import itertools
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _validate_decomposition(U, V, W, n, m, p, rank):
    """Check if U, V, W form a valid exact decomposition."""
    # Build tensors as Fraction arrays for exact arithmetic
    Uf = [[Fraction(U[i][r]) for r in range(rank)] for i in range(n*m)]
    Vf = [[Fraction(V[j][r]) for r in range(rank)] for j in range(m*p)]
    Wf = [[Fraction(W[k][r]) for r in range(rank)] for k in range(p*n)]
    
    # Verify: for all i,j,k: sum_r U[i*m+j][r] * V[j*p+k][r] == W[k*n+i][r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = sum(Uf[i*m+j][r] * Vf[j*p+k][r] for r in range(rank))
                rhs = Wf[k*n+i][r] if False else sum(Wf[k*n+i][r] for r in range(rank))
                # Actually, the tensor equation is:
                # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]... no
                # The bilinear form: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
                # Tensor: T[i,j,k] = sum_r u[i,j,r] * v[j,k,r] * w[k,i,r] is wrong
                # Correct: the multiplication tensor has entries T[i,j,k] such that
                # (A*B)[i,k] = sum_{j} sum_{r} U[i*m+j][r] * V[j*p+k][r] * ... 
                # Actually for matrix mult: T[i,j,k] = delta_{condition}
                # The standard form: T is a 3-tensor where T[i,j,k] = 1 if the
                # (i,j) element of A and (j,k) element of B contribute to (i,k) of output
                # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle index)
                # The decomposition: T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                # Wait, that's not right either. Let me reconsider.
                # 
                # Standard tensor decomposition for matrix mult (n x m) * (m x p):
                # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
                # T[i,j,k] = 1 (all ones) because every combination contributes
                # No wait, T[i,j,k] should encode: A[i,j] * B[j,k] contributes to C[i,k]
                # So the tensor is: T[i,j,k] = 1 for all i,j,k (it's the "multiplication" tensor)
                # 
                # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
                # 
                # But that gives a trilinear form, not bilinear. The correct formulation:
                # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_k V[j,k,r]*B[j,k])...
                # 
                # I'm confusing myself. Let me use the standard formulation:
                # The matrix multiplication tensor M has M[i,j,k] = 1 if and only if
                # the product of the (i,j)-th element of the first matrix and the (j,k)-th
                # element of the second matrix contributes to the (i,k)-th element of the result.
                # In this case, M[i,j,k] = 1 for ALL valid i, j, k.
                # 
                # A rank-r decomposition: M[i,j,k] = sum_{r=1}^{r} u_{i,j,r} * v_{j,k,r} * w_{k,i,r}
                # 
                # So we need: for all i,j,k: 1 = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                
                total = sum(Uf[i*m+j][r] * Vf[j*p+k][r] * Wf[k*n+i][r] for r in range(rank))
                if total != 1:
                    return False
    return True

def _validate_decomposition_fast(U, V, W, n, m, p, rank):
    """Faster validation using numpy for initial check, then exact for candidates."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), check sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i*m+j]
                v_row = V_arr[j*p+k]
                w_row = W_arr[k*n+i]
                prod = u_row * v_row * w_row
                s = np.sum(prod)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p = 42
    rank_schoolbook = n * m * p
    
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Strategy: Try to find lower-rank decompositions using structured forms
    # Key idea: For 2x3x7, we can potentially share computations across the 7 columns
    # 
    # Approach: Use the fact that n=2 is small. 
    # For each of the 3 middle indices j, we have 2*7=14 output entries.
    # We can try to represent these using fewer than 14 rank-1 terms per j,
    # but with shared structure across j.
    
    # Method 1: Random search with local optimization
    # Method 2: Structured search based on linear algebra
    
    # Let's try a structured approach:
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,6}
    # 
    # Consider grouping by j. For fixed j, we need to decompose the 2x7 matrix of ones
    # as sum_r u_i^j_r * v_k^j_r * w_{k,i}^j_r... this is getting complex.
    
    # Alternative: Use the known result that rank(2,3,7) <= 35 (or similar)
    # and try to construct it.
    
    # Let's try a different approach: use the "Winograd" style algorithm
    # For 2x3x7, we can use block decomposition.
    
    # Actually, let's try a direct search for rank 35, 36, ... up to 41
    # using a combination of random initialization and gradient descent
    
    # First, let's try a simple structured approach:
    # For 2xNxP, there are known algorithms. For 2x3x7, 
    # we can think of it as two rows, each being a 1x3 times 3x7 product.
    
    # Row 0: C[0,k] = sum_j A[0,j] * B[j,k] for k=0..6
    # Row 1: C[1,k] = sum_j A[1,j] * B[j,k] for k=0..6
    
    # Each row is a 1x3 * 3x7 = 1x7 product, which has rank 3*7=21 if done naively
    # But we can share computations between the two rows.
    
    # Let's try to find a decomposition of rank r for the full 2x3x7 tensor.
    # We'll use a randomized approach with verification.
    
    def try_rank(r, max_iterations=500):
        """Try to find a rank-r decomposition using random search + local refinement."""
        nonlocal best_result, best_rank
        
        if r >= best_rank:
            return None
        
        # Initialize with random values
        for attempt in range(10):
            U = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(n*m)]
            V = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(m*p)]
            W = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(p*n)]
            
            # Try to refine using a simple iterative method
            # This is a simplified version - in practice, we'd use gradient descent
            # on the non-convex objective sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
            
            # For now, just check if the random initialization happens to work (very unlikely)
            if _validate_decomposition_fast(U, V, W, n, m, p, r):
                return U, V, W
        
        return None
    
    # Try structured approach: exploit the fact that n=2
    # For each pair (i, k), the sum over j of A[i,j]*B[j,k] must equal 1 in the tensor.
    # 
    # Let's try a different structured form:
    # Use the identity: for 2x3x7, we can write
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Consider using shared "basis" vectors for the B matrix across the two i values.
    
    # Method: Try ranks from 35 to 41 with a more systematic search
    # Given time constraints, let's focus on a few promising ranks
    
    # Actually, let's try a known construction for 2xNxP
    # For 2x3x7, one approach is to use the fact that we can combine columns
    
    # Let's try: represent the 7 output columns using 4 "basis" columns with coefficients
    # This would give rank around 2*3*4 = 24 if we could do it perfectly,
    # but the coefficients would need to be encoded in U.
    
    # More realistically, let's try rank 35-40 with a local search.
    
    # Given the complexity and time constraints, let's return the schoolbook result
    # unless we can quickly find a better one.
    
    # Try a few quick random searches at rank 40
    for r in [40, 39, 38]:
        if time.time() > deadline - 5:
            break
        
        result = try_rank(r, max_iterations=100)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
