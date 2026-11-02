"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    # Build the multiplication tensor T[i,j,k] where T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C
    # Actually, the tensor for matrix multiplication C = A*B has entries:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i,j,k) and T[i,j,k] = 1 for all valid i,j,k
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, let me re-read the convention.
    
    # The output factor follows W[k*n+i, r]
    # U has shape (n*m, rank), indexed by (i, j) -> i*m + j
    # V has shape (m*p, rank), indexed by (j, k) -> j*p + k
    # W has shape (p*n, rank), indexed by (k, i) -> k*n + i
    
    # The tensor T for matrix multiplication C = A @ B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k
    
    # Let's verify by checking all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, rank, tol=1e-9):
    """Fast verification using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a batched dot product
    # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # For fixed (i,j,k): dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
    # Wait, it's sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # = dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
    
    # Let's do this efficiently:
    # For each i, j, k, we need dot product of U[i*m+j, :] with element-wise product of V[j*p+k, :] and W[k*n+i, :]
    
    # Alternative: think of it as a 3-tensor contraction
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's build it step by step
    # For each r, we have a rank-1 tensor: U[:, r] outer V[:, r] outer W[:, r] (with appropriate indexing)
    # Actually, let's just do it directly for small dimensions
    
    total = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        # U_col = U[:, r] has shape (n*m,)
        # V_col = V[:, r] has shape (m*p,)
        # W_col = W[:, r] has shape (p*n,)
        Uc = U_arr[:, r].reshape(n, m)
        Vc = V_arr[:, r].reshape(m, p)
        Wc = W_arr[:, r].reshape(p, n)
        # For this r, contribution to T[i,j,k] is Uc[i,j] * Vc[j,k] * Wc[k,i]
        # So we need: for all i,j,k: Uc[i,j] * Vc[j,k] * Wc[k,i]
        # This is like a tensor product with a specific permutation
        
        # Let's compute: for each i, j, k: Uc[i,j] * Vc[j,k] * Wc[k,i]
        # We can do: first Uc[i,j] * Vc[j,k] for all i,j,k -> this is like Uc @ Vc but we need to keep j
        # Actually, let's just loop over i for small n
        
        for i in range(n):
            # Uc[i, :] has shape (m,)
            # Wc[:, i] has shape (p,)
            # We need for all j,k: Uc[i,j] * Vc[j,k] * Wc[k,i]
            # = Uc[i, :] outer Vc * Wc[:, i] (element-wise)
            # Uc[i,j] * Vc[j,k] is the (j,k) entry of Uc[i,:] (as row) times Vc... 
            # Actually: sum over nothing, just Uc[i,j] * Vc[j,k] * Wc[k,i]
            # For fixed i: Uc[i, :] is a vector of length m, Wc[:, i] is a vector of length p
            # The contribution to T[i, :, :] is: Uc[i, :, None] * Vc * Wc[:, i][None, :]
            # Wait: Uc[i,j] * Vc[j,k] * Wc[k,i]
            # = (Uc[i, j] * 1) * Vc[j, k] * (Wc[k, i])
            # For fixed i, this is: outer(Uc[i, :], Wc[:, i]) * Vc element-wise
            # outer(Uc[i, :], Wc[:, i]) has shape (m, p), entry (j,k) = Uc[i,j] * Wc[k,i]
            # Then multiply element-wise by Vc[j,k]
            outer = np.outer(Uc[i, :], Wc[:, i])  # shape (m, p), entry (j,k) = Uc[i,j]*Wc[k,i]
            total[i, :, :] += outer * Vc
    return np.allclose(total, 1.0, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Start with schoolbook
    best_U = [[0] * schoolbook_rank for _ in range(n * m)]
    best_V = [[0] * schoolbook_rank for _ in range(m * p)]
    best_W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    
    # Try to find a lower-rank decomposition
    # Strategy: use random initialization and gradient descent on the tensor equations
    # Or: try known lower bounds and use optimization
    
    # The tensor rank for 2x5x5 matrix multiplication
    # Known: rank(2x5x5) is between 2*5*5/2=25 (trivial lower bound from slice rank) and 50 (schoolbook)
    # Actually, slice rank lower bound: for C = A*B with A: n x m, B: m x p
    # The slice rank of the multiplication tensor is at least max(n*m, m*p, n*p) = max(10, 25, 10) = 25
    # So rank >= 25
    
    # Let's try to find a decomposition of rank 25 or nearby
    # Use random initialization and try to solve the system
    
    np.random.seed(42)
    
    # Try ranks from 25 upward
    for target_rank in range(25, schoolbook_rank + 1):
        if time.time() > deadline - 30:
            break
            
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Scale to help convergence
            U /= np.sqrt(target_rank)
            V /= np.sqrt(target_rank)
            W /= np.sqrt(target_rank)
            
            # Gradient descent to minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            # Use simple SGD
            
            lr = 0.01
            num_iters = 5000
            for it in range(num_iters):
                if time.time() > deadline - 5:
                    break
                    
                # Compute gradient
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # d/dU[i*m+j,r] = (T[i,j,k] - 1) * V[j*p+k,r] * W[k*n+i,r]
                # Similarly for V and W
                
                # Compute T
                T = np.zeros((n, m, p), dtype=float)
                for r in range(target_rank):
                    Uc = U[:, r].reshape(n, m)
                    Vc = V[:, r].reshape(m, p)
                    Wc = W[:, r].reshape(p, n)
                    for i in range(n):
                        outer = np.outer(Uc[i, :], Wc[:, i])
                        T[i, :, :] += outer * Vc
                
                error = T - 1.0
                if np.allclose(error, 0, atol=1e-8):
                    # Found a solution!
                    # Convert to lists
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    if _verify(U_list, V_list, W_list, n, m, p, target_rank, tol=1e-7):
                        best_rank = target_rank
                        best_U = U_list
                        best_V = V_list
                        best_W = W_list
                        return {"U": best_U, "V": best_V, "W": best_W}
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = error[i, j, k]
                            for r in range(target_rank):
                                grad_U[i*m+j, r] += e * V[j*p+k, r] * W[k*n+i, r]
                                grad_V[j*p+k, r] += e * U[i*m+j, r] * W[k*n+i, r]
                                grad_W[k*n+i, r] += e * U[i*m+j, r] * V[j*p+k, r]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Decay learning rate
                lr = max(0.0001, 0.01 * (0.999 ** it))
    
    # If we found a better rank, return it; otherwise return schoolbook
    if best_rank < schoolbook_rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
