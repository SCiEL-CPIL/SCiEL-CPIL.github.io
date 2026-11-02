"""Matrix multiplication 2,4,5 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 40).
    2. Use a block-Strassen-inspired approach to get a better base.
    3. Try to reduce rank by finding linear dependencies among the columns.
    """
    deadline = time.time() + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # The tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n + i, r] follows the trace convention
    
    # Schoolbook: rank = n*m*p = 40
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k)
    
    # Let's try a smarter approach using the structure
    # For n=2, we can use Strassen-like decomposition on the 2x2 block structure
    # But here we have 2x4 * 4x5, so let's think differently
    
    # The key insight: we need to find matrices U, V, W such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    # This is equivalent to: for each (i,k), sum_j (U[:,r] outer V[:,r]) contracted appropriately = 1
    
    # More precisely, the tensor is T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k] = 1 for all i,j,k
    
    # Let's use a random search with verification
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        # For each (i,j,k), check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W, rank):
        """Faster verification using numpy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # For each r, we have outer product contributions
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        
        # For each (i,j,k), sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        # This is a tensor contraction
        # Let's compute it step by step
        # First, for each (i,j,k): sum_r U_3d[i,j,r] * V_3d[j,k,r]
        # Then multiply by W_3d[k,i,r] and sum over r
        
        # Actually, let's just do it directly with einsum
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # = einsum('ijr,jkr,kr->ik', U_3d, V_3d, W_3d) -- wait, W is indexed by [k,i,r]
        # W_3d has shape (p, n, rank) = (k, i, r)
        # So we want: sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        
        result = np.einsum('ijr,jkr,ki r->ik', U_3d, V_3d, W_3d)
        
        # Check if result is all 1s
        if not np.allclose(result, np.ones((n, p)), atol=1e-9):
            return False
        return True
    
    # Start with schoolbook
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
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by finding linear dependencies
    # The columns of U, V, W together define the decomposition
    # If we can express one column as a linear combination of others, we can reduce rank
    
    # Let's try a more systematic approach:
    # Randomly generate candidate decompositions and try to minimize rank
    
    # Actually, let's think about this differently.
    # The tensor T[i,j,k] = 1 for all i,j,k is a rank-1 tensor in the (i,j,k) space? No.
    # T[i,j,k] = 1 means the matrix multiplication tensor is the all-ones tensor.
    
    # Wait, no. The matrix multiplication tensor for n x m * m x p -> n x p is:
    # T[i, j, k] where i is row of A (and C), j is column of A (and row of B), k is column of B (and C)
    # The tensor is T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all valid (i,j,k).
    
    # The rank of this tensor is what we're trying to minimize.
    
    # For small dimensions, known results:
    # 2x4 * 4x5: let's see if we can do better than 40.
    
    # Let's try a random search approach
    np.random.seed(42)
    
    def random_decomposition(rank):
        """Generate a random decomposition of given rank."""
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        return U.tolist(), V.tolist(), W.tolist()
    
    def try_reduce_rank(start_rank, target_rank, max_iterations=1000):
        """Try to find a decomposition of target_rank by modifying a start_rank decomposition."""
        # This is a heuristic approach
        # We'll try random perturbations and check if we can maintain correctness while reducing rank
        
        # Actually, let's use a different approach:
        # Solve for the decomposition using linear algebra
        # The constraint is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k)
        
        # This is a system of n*m*p equations in (n*m + m*p + p*n) * rank variables
        # For rank r, we have (n*m + m*p + p*n) * r variables and n*m*p equations
        # n*m*p = 40, n*m + m*p + p*n = 8 + 20 + 10 = 38
        # So for r=1: 38 vars, 40 eqs (underdetermined)
        # For r=2: 76 vars, 40 eqs (overdetermined, likely solvable)
        
        # Let's try to solve for small ranks
        for r in range(1, start_rank + 1):
            if time.time() > deadline - 10:
                break
            # Try to find a solution of rank r
            # Use a random initialization and optimize
            # This is a non-convex problem, so we'll use random restarts
            
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break
                    
                U = np.random.randn(n * m, r) * 0.1
                V = np.random.randn(m * p, r) * 0.1
                W = np.random.randn(p * n, r) * 0.1
                
                # Use gradient descent to minimize the error
                # Error = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                
                lr = 0.01
                for it in range(100):
                    if time.time() > deadline - 10:
                        break
                    
                    # Compute gradients
                    U_3d = U.reshape(n, m, r)
                    V_3d = V.reshape(m, p, r)
                    W_3d = W.reshape(p, n, r)
                    
                    # Error for each (i,j,k)
                    # e[i,j,k] = sum_r U_3d[i,j,r]*V_3d[j,k,r]*W_3d[k,i,r] - 1
                    # This is expensive to compute directly, let's use einsum
                    # sum_r U_3d[i,j,r]*V_3d[j,k,r]*W_3d[k,i,r]
                    # = einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                    # Wait, W_3d is (p, n, r) = (k, i, r), so W_3d[k,i,r]
                    # So we want einsum('ijr,jkr,ki r->ijk', U_3d, V_3d, W_3d)
                    
                    prod = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                    err = prod - 1.0
                    
                    # Gradients
                    # d/dU[i,j,r] = sum_{k} err[i,j,k] * V_3d[j,k,r] * W_3d[k,i,r]
                    # = einsum('ijk,jkr,ki r->ijr', err, V_3d, W_3d) -- wait, need to be careful
                    # d/dU[i,j,r] = sum_k err[i,j,k] * V_3d[j,k,r] * W_3d[k,i,r]
                    # = einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d)
                    grad_U = np.einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d)
                    
                    # d/dV[j,k,r] = sum_i err[i,j,k] * U_3d[i,j,r] * W_3d[k,i,r]
                    # = einsum('ijk,ijr,kir->jkr', err, U_3d, W_3d)
                    grad_V = np.einsum('ijk,ijr,kir->jkr', err, U_3d, W_3d)
                    
                    # d/dW[k,i,r] = sum_j err[i,j,k] * U_3d[i,j,r] * V_3d[j,k,r]
                    # = einsum('ijk,ijr,jkr->kir', err, U_3d, V_3d)
                    grad_W = np.einsum('ijk,ijr,jkr->kir', err, U_3d, V_3d)
                    
                    # Update
                    U -= lr * grad_U.reshape(n * m, r)
                    V -= lr * grad_V.reshape(m * p, r)
                    W -= lr * grad_W.reshape(p * n, r)
                    
                    # Check convergence
                    if np.max(np.abs(err)) < 1e-8:
                        break
                
                # Check if we found a valid decomposition
                U_3d = U.reshape(n, m, r)
                V_3d = V.reshape(m, p, r)
                W_3d = W.reshape(p, n, r)
                prod = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                if np.max(np.abs(prod - 1.0)) < 1e-8:
                    # Found a valid decomposition of rank r
                    return U.tolist(), V.tolist(), W.tolist(), r
        
        return None
    
    # Try to find a lower rank decomposition
    result = try_reduce_rank(rank, rank, max_iterations=100)
    if result is not None:
        U_new, V_new, W_new, new_rank = result
        if new_rank < best_rank:
            best_rank = new_rank
            best_U, best_V, best_W = U_new, V_new, W_new
    
    # Return the best result found
    return {"U": best_U, "V": best_V, "W": best_W}
