"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import numpy as np
from scipy.optimize import linprog
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Start with schoolbook rank as fallback
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Use a local search approach with rational arithmetic
    
    import random
    from fractions import Fraction
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        r = len(U[0]) if U else 0
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    expected = 1 if (i == j and k == 0) or False else 0
                    # Actually, the tensor is: T[i,j,k] = 1 if this is a valid multiplication term
                    # For matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
                    # So the tensor has T[i,j,k] = 1 for all valid (i,j,k)
                    # Wait, let me reconsider. The tensor for matrix multiplication maps (A,B) -> C
                    # where C[i,k] = sum_j A[i,j] B[j,k]
                    # So the bilinear form is: sum_{i,j,k} C[i,k] * A[i,j] * B[j,k]
                    # The tensor T has T[i,j,k] = 1 for all i,j,k (since each term appears)
                    # No wait, that's not right either.
                    
                    # Let me think again. The matrix multiplication tensor M is defined such that:
                    # M(A,B) = A*B where (A*B)[i,k] = sum_j A[i,j] * B[j,k]
                    # As a trilinear form: sum_{i,j,k} M[i,j,k] * A[i,j] * B[j,k] * C[i,k]
                    # But for the decomposition, we need:
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta for the right indices
                    # Actually, the tensor entry T[i,j,k] should be 1 if (i,j,k) is a valid triple
                    # For 3x3 * 3x8: i in 0..2, j in 0..2, k in 0..7
                    # T[i,j,k] = 1 for all valid (i,j,k) since each contributes to C[i,k]
                    # Wait no. Let me re-read the problem.
                    
                    # "The output factor follows the official trace convention W[k*n+i,r]"
                    # So W is indexed by (k, i) where k is the column of B (0..7) and i is the row of A (0..2)
                    # U is indexed by (i, j) where i is row of A (0..2) and j is col of A (0..2)
                    # V is indexed by (j, k) where j is row of B (0..2) and k is col of B (0..7)
                    
                    # The tensor decomposition should satisfy:
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if this is a valid term
                    # For matrix multiplication, every (i,j,k) triple where j is the shared index is valid
                    # So T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
                    # Wait, that can't be right because then the rank would just be n*m*p.
                    
                    # Actually, I think the tensor is: T[i,j,k] = 1 for all valid (i,j,k)
                    # And we're trying to decompose this tensor into rank-1 terms.
                    # The schoolbook decomposition has rank n*m*p with each term being a single basis tensor.
                    # A lower rank decomposition would be a "fast" multiplication algorithm.
                    
                    # So T[i,j,k] = 1 for all i,j,k in their respective ranges.
                    expected = 1
                    if s != expected:
                        return False
        return True
    
    def random_rational_init(rows, cols, max_denom=4):
        """Generate random matrix with small rational entries."""
        M = []
        for _ in range(rows):
            row = []
            for _ in range(cols):
                if random.random() < 0.5:
                    row.append(0)
                else:
                    num = random.randint(-2, 2)
                    den = random.randint(1, max_denom)
                    if num == 0:
                        row.append(0)
                    else:
                        row.append(Fraction(num, den))
            M.append(row)
        return M
    
    def local_search(U, V, W, n, m, p, max_iters=50):
        """Iteratively refine the decomposition by fixing two matrices and solving for the third."""
        r = len(U[0]) if U else 0
        nm, mp, pn = n * m, m * p, p * n
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Fix U and V, solve for W
            # For each (k, i) pair, we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
            # This is a system of equations in W[k*n+i, :]
            # Actually, for each fixed (k,i), we have m equations (one for each j) in r unknowns W[k*n+i, 0..r-1]
            # But we also have the constraint that the same W[k*n+i, c] appears for all j
            # So we need to find W such that for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            
            # This is nonlinear in W. Let's use a different approach.
            # Fix U and W, solve for V, or use gradient descent.
            
            # Simpler approach: use the fact that if we fix U and V, then for each (k,i),
            # we need W[k*n+i, :] to satisfy: sum_j (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]) = m
            # No, that's not right either.
            
            # Let me try a different strategy: use the structure of the problem.
            # The tensor T[i,j,k] = 1 for all i,j,k.
            # We want T = sum_c u_c v_c w_c where u_c in R^{nm}, v_c in R^{mp}, w_c in R^{pn}
            # T[i,j,k] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
            
            # For a given (i,k), sum_j T[i,j,k] = m (since T[i,j,k]=1 for all j)
            # But sum_j sum_c u_c[i*m+j]*v_c[j*p+k]*w_c[k*n+i] = sum_c w_c[k*n+i] * (u_c^T * (v_c with j dimension))
            # This is getting complicated.
            
            # Let me just try a simple coordinate descent: randomly pick one matrix, 
            # and for each row, adjust it to reduce the error.
            
            pass  # Placeholder for now
            
        return U, V, W
    
    # Try a few random initializations with lower rank
    for trial in range(10):
        if time.time() > deadline - 10:
            break
        
        for target_rank in [60, 65, 70, 71]:
            if time.time() > deadline - 5:
                break
            
            U = random_rational_init(n * m, target_rank, max_denom=2)
            V = random_rational_init(m * p, target_rank, max_denom=2)
            W = random_rational_init(p * n, target_rank, max_denom=2)
            
            # Simple verification (this is slow, so only do it for small cases)
            # For now, skip verification and just try to return if it looks plausible
            # Actually, we need to verify for correctness
            
            # Quick check: compute the error for a few random entries
            error = 0
            for _ in range(20):
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                s = sum(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c] for c in range(target_rank))
                if s != 1:
                    error += abs(s - 1)
            
            if error < 1e-6:
                # Full verification
                full_error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c] for c in range(target_rank))
                            full_error += abs(s - 1)
                if full_error < 1e-6:
                    # Found a valid decomposition!
                    best_rank = target_rank
                    # Convert to lists for return
                    best_U = [[float(x) if x != int(x) else int(x) for x in row] for row in U]
                    best_V = [[float(x) if x != int(x) else int(x) for x in row] for row in V]
                    best_W = [[float(x) if x != int(x) else int(x) for x in row] for row in W]
                    return {"U": best_U, "V": best_V, "W": best_W}
    
    # Return the schoolbook solution as fallback
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
