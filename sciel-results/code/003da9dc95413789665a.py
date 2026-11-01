"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k] for (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
    should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Actually, the standard tensor decomposition for matrix multiplication:
    T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    where (AB)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_r sum_j U[i,j,r] A_{i,j} V[j,k,r] B_{j,k} W[k,i,r]
    
    Wait, let me re-read the task. The output factor follows W[k*n+i, r].
    So the decomposition is:
    (AB)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Hmm, that's not quite right either. Let me think more carefully.
    
    The matrix multiplication tensor T has indices (i, j, k) where T[i,j,k] = 1 if we're computing (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    
    A rank-r decomposition gives:
    T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    But the task specifies the shapes:
    U: (n*m, rank) - indexed by (i, j)
    V: (m*p, rank) - indexed by (j, k)  
    W: (p*n, rank) - indexed by (k, i)
    
    And W[k*n+i, r] suggests W is indexed as W[k, i] flattened to k*n+i.
    
    So the decomposition should be:
    (AB)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, but that would mean:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This doesn't look right because the sum over r is outside the sum over j.
    
    Let me reconsider. The standard form is:
    (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    A rank-r decomposition of the multiplication tensor means we can write:
    (AB)_{i,k} = sum_r (sum_j U[i,j,r] A_{i,j}) * (sum_k' V[j,k',r] B_{j,k'}) * W[k,i,r]
    
    No, that's not right either. Let me look at this more carefully.
    
    The matrix multiplication tensor T is a 3-tensor with T[i,j,k] = 1 for the "multiplication" operation.
    A rank-r decomposition is:
    T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    This means:
    (AB)_{i,k} = sum_j T[i,j,k] A_{i,j} B_{j,k} = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] A_{i,j} B_{j,k}
    = sum_r W[k,i,r] (sum_j U[i,j,r] A_{i,j}) (sum_j V[j,k,r] B_{j,k})
    
    But wait, that's summing over j in both factors, which would give a product of two sums, not what we want.
    
    I think I'm confusing myself. Let me look at the schoolbook example:
    U[i*m+j, r] = 1 for r corresponding to (i,j,k)
    V[j*p+k, r] = 1 for r corresponding to (i,j,k)
    W[k*n+i, r] = 1 for r corresponding to (i,j,k)
    
    So for each (i,j,k), there's a rank-1 term. The sum over r gives:
    sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    = sum_{(i',j',k') : r=(i',j',k')} [i'=i, j'=j] * [j'=j, k'=k] * [k'=k, i'=i]
    = 1 if (i,j,k) matches, 0 otherwise
    
    So T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    And the matrix product is:
    (AB)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_j T[i,j,k] A_{i,j} B_{j,k}
    = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A_{i,j} B_{j,k}
    
    This is the correct interpretation. So to verify, I need to check that for all (i,j,k):
    T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    where T[i,j,k] = 1 for all valid (i,j,k).
    
    """
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(n, m, p, target_rank, time_limit=10):
    """Try to find a decomposition of the given rank using random initialization and optimization.
    
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    
    # Initialize with random values
    np.random.seed(42)
    
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    # We need to minimize sum_{i,j,k} (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1)^2
    # This is a non-convex problem. Let's try a simple gradient descent approach.
    
    # Actually, let's try a different approach: use the fact that we can fix one matrix
    # and solve for the others. But this is still non-convex.
    
    # For now, let's just return the schoolbook solution if target_rank >= n*m*p
    if target_rank >= n * m * p:
        return _schoolbook(n, m, p)
    
    # Try a few random restarts
    for restart in range(5):
        if time.time() - start_time > time_limit:
            break
        
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Simple iterative refinement
        for iteration in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    for r in range(target_rank):
                        # Gradient of sum_k (sum_r' U[i*m+j,r'] V[j*p+k,r'] W[k*n+i,r'] - 1)^2 wrt U[i*m+j,r]
                        grad = 0
                        for k in range(p):
                            inner = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                            grad += 2 * (inner - 1) * V[j*p+k][r] * W[k*n+i][r]
                        U[i*m+j][r] -= 0.01 * grad
            
            # Check convergence
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-6:
                # Convert to list of lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    for target_rank in range(20, best_rank):
        if time.time() > deadline - 30:
            break
        
        result = _try_rank(n, m, p, target_rank, time_limit=10)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
