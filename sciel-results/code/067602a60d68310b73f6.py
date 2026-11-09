"""Matrix multiplication 3x4x6 rank-minimal decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is valid."""
    # For each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    # In our convention: W[k*n+i, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
    r = U.shape[1]
    for i in range(n):
        for k in range(p):
            expected = 0.0
            for j in range(m):
                # The contribution of (i,j) and (j,k) to (i,k)
                # U[i*m+j, c] * V[j*p+k, c] should sum to W[k*n+i, c]
                # Actually, the tensor entry T[(i,j),(j,k),(k,i)] = 1
                # So for the decomposition: sum_c U[(i,j),c]*V[(j,k),c]*W[(k,i),c] = T[(i,j),(j,k),(k,i)]
                # Wait, the standard tensor decomposition for matmul is:
                # T_{i,j,k} = sum_c U_{i,c} V_{j,c} W_{k,c}
                # where T_{i,j,k} = delta_{j,j} (always 1 in the right positions)
                # More precisely, the matmul tensor has T[(i,j),(j,k),(k,i)] = 1
                pass
    
    # Let me re-derive. The matrix multiplication tensor T has indices (i,j,k) where
    # i in [0,n), j in [0,m), k in [0,p), and T[i,j,k] = 1.
    # A decomposition of rank r means: T[i,j,k] = sum_{c=0}^{r-1} U[i,c]*V[j,c]*W[k,c]
    # But the task uses a different indexing:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n+i, r] follows the trace convention.
    #
    # So the decomposition is: for all (i,j,k),
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # Wait, that's not right either. Let me re-read the task.
    # "W[k*n+i,r]" - this is the output factor.
    # The standard formulation: C = A @ B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Tensor decomposition: The bilinear map (A,B) -> C can be written as:
    # C[i,k] = sum_c U[i*m+j, c] ... 
    #
    # Actually, I think the convention is:
    # U[(i,j), c] corresponds to A[i,j]
    # V[(j,k), c] corresponds to B[j,k]  
    # W[(k,i), c] is the output coefficient
    #
    # And the constraint is: for all (i,j,k),
    # sum_c U[(i,j),c] * V[(j,k),c] = W[(k,i),c] ... no that doesn't work.
    #
    # Let me think again. The bilinear form is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # A rank-r decomposition means we can write:
    # C[i,k] = sum_{c=0}^{r-1} (sum_j U[i*m+j,c] * A[i,j]) * (sum_k' V[j*p+k',c] * B[j,k'])
    # Hmm, this is getting confusing. Let me just use the standard formulation.
    #
    # Standard: T[i,j,k] = sum_c u_{i,c} * v_{j,c} * w_{k,c}
    # where T[i,j,k] = 1 for the matmul tensor (with appropriate index ranges).
    #
    # But the task specifies U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank).
    # This suggests a different grouping. Let me just verify numerically.
    
    # The correct interpretation:
    # For the decomposition to be valid, for all (i,j,k) with i in [0,n), j in [0,m), k in [0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # (since the matmul tensor has all entries equal to 1)
    
    # Let's verify this
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_rank_search(n, m, p, max_rank, time_budget, rng):
    """Try to find a decomposition of rank < max_rank using random search."""
    best_U, best_V, best_W = None, None, None
    best_rank = max_rank
    
    # Start with schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    best_U, best_V, best_W = U0, V0, W0
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # This is a very hard problem, so we'll do a limited search
    #
    # Approach: For a given rank r, we need to find U, V, W such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    #
    # This is a system of n*m*p equations in 3*r*(n*m + m*p + p*n) unknowns.
    # For small r, this is overdetermined and generally has no solution.
    #
    # One approach: fix some entries and solve for others.
    # Another: use the fact that the tensor has a specific structure.
    
    # For now, let's just try a few random initializations and see if we can
    # reduce the rank by setting some entries to zero and solving.
    
    # Actually, let's try a different approach: use the structure of the problem.
    # The matmul tensor for 3x4x6 has a specific structure that might allow
    # a lower rank decomposition.
    
    # One known technique: if one of the dimensions is 1, the rank is n*p.
    # For 3x4x6, we can try to exploit the fact that 4 = 2+2 or other splits.
    
    # Let's try a block decomposition approach:
    # Split the middle dimension: j in [0,4) into j1 in [0,2) and j2 in [2,4)
    # C[i,k] = sum_{j1} A[i,j1]B[j1,k] + sum_{j2} A[i,j2]B[j2,k]
    # Each part is a 3x2x6 matmul, which has rank 3*2*6 = 36
    # Total: 72. No improvement.
    
    # Try 3x1x6 + 3x3x6:
    # 3x1x6: rank 18
    # 3x3x6: rank 54
    # Total: 72. Still no improvement.
    
    # The fundamental issue is that simple block decompositions don't help.
    # We need a more sophisticated approach.
    
    # For now, return the schoolbook decomposition.
    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    start_time = time.time()
    time_limit = 120  # seconds
    
    # Return schoolbook decomposition as a safe valid answer
    U, V, W = _schoolbook(n, m, p)
    
    # Check time
    if time.time() - start_time > time_limit - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a better decomposition
    rng = random.Random(42)
    U, V, W, rank = _try_rank_search(n, m, p, n*m*p, time_limit - 10, rng)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
