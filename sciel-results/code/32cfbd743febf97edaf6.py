"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m) x rank
    V: (m*p) x rank
    W: (p*n) x rank
    Convention: W[k*n + i, r] is the coefficient for output (i, k) from input (j) at rank r.
    The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    For matrix multiplication, T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k].
    So we need: for each (i,k), sum_j A[i,j]*B[j,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k]) ... 
    Actually the standard bilinear form: C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k])? No.
    
    Let me re-read the convention. The task says:
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W[k*n+i, r] follows the official trace convention.
    
    The matrix multiplication tensor for C = A*B where A is n x m and B is m x p:
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    A rank-r decomposition gives:
    C[i,k] = sum_{r=0}^{r-1} (sum_{j=0}^{m-1} U[i*m+j, r] * A[i,j]) * (sum_{j=0}^{m-1} V[j*p+k, r] * B[j,k])
    
    Wait, that's not right either. Let me think about this more carefully.
    
    In a bilinear algorithm for matrix multiplication, we have:
    C = A * B
    where A is n x m, B is m x p.
    
    We define intermediate products:
    M_r = (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    No wait, the standard formulation is:
    
    For each rank component r, we compute:
    X_r = sum_{i,j} U[i*m+j, r] * A[i,j]  -- but this sums over all i,j, which doesn't make sense for a bilinear form.
    
    Let me reconsider. The correct formulation for a rank-r decomposition of matrix multiplication:
    
    We want C[i,k] = sum_{j} A[i,j] * B[j,k]
    
    A rank-r decomposition means:
    C[i,k] = sum_{r=0}^{r-1} alpha_r(i) * beta_r(k)
    where alpha_r(i) is a linear combination of entries of A, and beta_r(k) is a linear combination of entries of B.
    
    More precisely:
    alpha_r = sum_{i',j} U[i'*m+j, r] * A[i',j]
    beta_r = sum_{j',k'} V[j'*p+k', r] * B[j',k']
    
    Then C[i,k] = sum_r alpha_r * beta_r ... but that gives a scalar, not C[i,k].
    
    I think the correct formulation is:
    
    C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    Let me verify with the schoolbook case. In schoolbook, rank = n*m*p.
    For each triple (i,j,k), we have one rank component.
    U[i*m+j, r] = 1 if r corresponds to (i,j,k), else 0.
    V[j*p+k, r] = 1 if r corresponds to (i,j,k), else 0.
    W[k*n+i, r] = 1 if r corresponds to (i,j,k), else 0.
    
    Then: sum_j U[i*m+j, r] * A[i,j] = A[i,j] (when r corresponds to (i,j,k))
    sum_j V[j*p+k, r] * B[j,k] = B[j,k] (when r corresponds to (i,j,k))
    
    So C[i,k] = sum_{r} A[i,j_r] * B[j_r, k] * W[k*n+i, r]
    
    For the schoolbook case, W[k*n+i, r] = 1 only when r corresponds to (i,j,k).
    So C[i,k] = sum_{j} A[i,j] * B[j,k] * 1 = sum_j A[i,j]*B[j,k]. Correct!
    
    So the formula is:
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    This is a trilinear form in A, B, and the W coefficients.
    
    For verification, we can check this identity symbolically or by evaluating on random matrices.
    """
    # Check using random test matrices
    for _ in range(20):
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        C_expected = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        C_actual = [[0] * p for _ in range(n)]
        for r in range(rank):
            # Compute alpha_r = sum_j U[i*m+j, r] * A[i,j] for each i
            # But this depends on i... so we need to be careful.
            # Actually, the formula is:
            # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
            
            # Precompute for each r:
            pass
        
        # Let's just do it directly
        C_actual = [[0] * p for _ in range(n)]
        for r in range(rank):
            for i in range(n):
                for k in range(p):
                    if W[k*n + i][r] != 0:
                        alpha = sum(U[i*m + j][r] * A[i][j] for j in range(m))
                        beta = sum(V[j*p + k][r] * B[j][k] for j in range(m))
                        C_actual[i][k] += W[k*n + i][r] * alpha * beta
        
        if C_actual != C_expected:
            return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using structured approaches
    
    # Approach 1: Try known small ranks with random search over {-1, 0, 1} entries
    # The tensor for 3x4 * 4x4 matrix multiplication has dimension 3*4*4 = 48.
    # Lower bound on rank is max(n*m, m*p, p*n) = max(12, 16, 12) = 16.
    # Upper bound is 48 (schoolbook).
    # Known results: For small matrices, the rank is often close to the lower bound.
    # Let's try ranks from 16 upward.
    
    for target_rank in range(16, min(49, best_rank + 1)):
        if time.time() > deadline - 5:
            break
        
        # Try random search with {-1, 0, 1} entries
        max_attempts = 5000
        attempt = 0
        while attempt < max_attempts and time.time() < deadline - 5:
            attempt += 1
            
            # Generate random U, V, W with entries in {-1, 0, 1}
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Verify
            if _verify(U, V, W, n, m, p, target_rank):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        # If we found a valid decomposition at this rank, we can stop
        if best_rank == target_rank:
            break
    
    return best
# EVOLVE-BLOCK-END
