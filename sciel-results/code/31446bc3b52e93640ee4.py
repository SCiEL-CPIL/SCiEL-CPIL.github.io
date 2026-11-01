"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False, i, j, k, val
    return True, -1, -1, -1, 0.0

def _try_random_search(n, m, p, target_rank, max_iters=50000, seed=42):
    """Try to find a rank decomposition via random initialization and gradient-like updates."""
    rng = np.random.RandomState(seed)
    best_err = float('inf')
    best_solution = None
    
    for attempt in range(max_iters):
        U = rng.randn(n*m, target_rank)
        V = rng.randn(m*p, target_rank)
        W = rng.randn(p*n, target_rank)
        
        # Quick check: compute all tensor entries
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(target_rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    errors.append(abs(val - 1.0))
        
        max_err = max(errors)
        if max_err < best_err:
            best_err = max_err
            best_solution = (U.copy(), V.copy(), W.copy())
        
        if max_err < 1e-9:
            return U, V, W
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _construct_rank39(n, m, p):
    """
    Attempt to construct a rank-39 decomposition for 2x2 * 2x11.
    
    The key insight: for n=2, m=2, p=11, we have 2*2*11 = 44 schoolbook terms.
    We need to find linear dependencies to reduce the rank.
    
    Strategy: Use the structure that for fixed (i,k), the sum over j of A[i,j]*B[j,k] 
    must equal C[i,k]. With n=2, m=2, each C[i,k] is a sum of 2 products.
    
    Let's try to exploit the fact that we can share rank-1 terms across different (i,j,k)
    triples by using linear combinations.
    
    A known technique: if we can find that certain rank-1 tensors are linearly dependent,
    we can reduce the rank. For matmul, the tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (contribution).
    
    Actually, for standard matmul T[i,j,k] = 1 for all i,j,k (since every product term contributes).
    So T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k.
    
    This is the all-ones tensor of shape (n*m) x (m*p) x (p*n) where the indices are
    mapped as (i,j) -> i*m+j, (j,k) -> j*p+k, (k,i) -> k*n+i.
    
    Wait, this is NOT the all-ones tensor because the index mappings create a specific structure.
    Let me reconsider.
    
    The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, and 0 otherwise.
    But in the decomposition, we're looking at U[a,r]*V[b,r]*W[c,r] where a=i*m+j, b=j*p+k, c=k*n+i.
    The constraint is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1.
    And for "invalid" combinations (where the indices don't form a valid (i,j,k) triple), there's no constraint.
    
    So we only need to satisfy 2*2*11 = 44 constraints.
    """
    # Start with schoolbook and try to find linear dependencies
    U, V, W = _schoolbook(n, m, p)
    
    # Try to find a rank-reduced version by checking if we can express
    # some columns as linear combinations of others.
    
    # For each column r, the "signature" is (U[:,r], V[:,r], W[:,r]).
    # Two columns are linearly dependent if one is a scalar multiple of the other in all three.
    
    # Better approach: use the fact that we have 44 constraints and 3*(n*m+m*p+p*n)*rank unknowns.
    # For rank r, we have 3*(4+22+22)*r = 3*48*r = 144r unknowns, 44 constraints.
    # So for r >= 1, we're overdetermined in a different sense.
    
    # Let's try a more structured approach:
    # For n=2, m=2, p=11, the constraints are:
    # For i in {0,1}, j in {0,1}, k in {0,...,10}:
    #   sum_r U[i*2+j, r] * V[j*11+k, r] * W[k*2+i, r] = 1
    
    # This is a system of 44 equations. Let's try to find a solution with minimal rank.
    
    # Use a greedy approach: start with rank 1, check if feasible, increase rank.
    
    best_result = None
    for target_rank in range(30, 45):
        # Try random initializations
        for seed in range(200):
            rng = np.random.RandomState(seed)
            U = rng.randn(n*m, target_rank)
            V = rng.randn(m*p, target_rank)
            W = rng.randn(p*n, target_rank)
            
            # Build the constraint matrix
            # For each (i,j,k), the equation is: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is nonlinear in the variables. Use alternating least squares.
            
            success = False
            for iteration in range(100):
                # Fix V and W, solve for U
                for i in range(n):
                    for j in range(m):
                        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                        # This is a linear system in U[i*m+j, :]
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            for r in range(target_rank):
                                A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # Solve A * u = ones(p)
                        try:
                            u, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i*m+j, :] = u
                        except:
                            pass
                
                # Fix U and W, solve for V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n*m, target_rank))
                        for i in range(n):
                            for jj in range(m):
                                # We need: for each i, sum_r U[i*m+jj,r]*V[jj*p+k,r]*W[k*n+i,r] = 1
                                # But V[j*p+k,r] only appears when jj=j
                                # So for fixed j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i
                                A2 = np.zeros((n, target_rank))
                                for i in range(n):
                                    for r in range(target_rank):
                                        A2[i, r] = U[i*m+j, r] * W[k*n+i, r]
                                try:
                                    v, _, _, _ = np.linalg.lstsq(A2, np.ones(n), rcond=None)
                                    V[j*p+k, :] = v
                                except:
                                    pass
                
                # Fix U and V, solve for W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            for r in range(target_rank):
                                A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        try:
                            w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i, :] = w
                        except:
                            pass
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-10:
                    success = True
                    break
                
                if max_err < 1e-12:
                    break
            
            if success:
                best_result = (U, V, W, target_rank)
                break
        
        if best_result:
            break
    
    if best_result:
        return best_result[0], best_result[1], best_result[2]
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook as baseline
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The knowledge suggests rank 39 is achievable
    for target_rank in range(35, 44):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations with alternating least squares
        for seed in range(500):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(seed)
            U = rng.randn(n*m, target_rank)
            V = rng.randn(m*p, target_rank)
            W = rng.randn(p*n, target_rank)
            
            # Alternating least squares
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            for r in range(target_rank):
                                A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        try:
                            u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i*m+j, :] = u
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A2 = np.zeros((n, target_rank))
                        for i in range(n):
                            for r in range(target_rank):
                                A2[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        try:
                            v, _, _, _ = np.linalg.lstsq(A2, np.ones(n), rcond=None)
                            V[j*p+k, :] = v
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A3 = np.zeros((m, target_rank))
                        for j in range(m):
                            for r in range(target_rank):
                                A3[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        try:
                            w, _, _, _ = np.linalg.lstsq(A3, np.ones(m), rcond=None)
                            W[k*n+i, :] = w
                        except:
                            pass
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-10:
                    # Found a valid decomposition
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    break
                if max_err < 1e-12:
                    break
            
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
    
    # Convert to list format
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
