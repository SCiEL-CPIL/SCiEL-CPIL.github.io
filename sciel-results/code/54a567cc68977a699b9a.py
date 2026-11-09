"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _to_list(M):
    # Convert to list of lists with integer or half-integer entries
    M = np.round(M * 2) / 2
    return [[float(x) for x in row] for row in M]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that the decomposition exactly matches the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the multiplication tensor T[i,j,k] = delta_{i,k} * delta_{j,j} (trivially 1 for valid indices)
    # Actually, the tensor is: T[i, j, k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] = 1 (since the multiplication tensor has 1s at all (i,j,k) positions)
    # Wait, let me reconsider. The tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i, j, k] = 1 for all valid i, j, k (it's the identity in the "j" dimension pairing).
    # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # Let's verify by checking a few random instances
    for _ in range(20):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_decomposed = np.zeros((n, p))
        for c in range(rank):
            u_vec = U[:, c].reshape(n, m)
            v_vec = V[:, c].reshape(m, p)
            w_vec = W[:, c].reshape(p, n)
            # The contribution to C[i,k] from term c is:
            # sum_j u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
                    C_decomposed[i, k] += s
        if not np.allclose(C_decomposed, C_expected, atol=tol):
            return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Faster verification using tensor operations."""
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # The decomposition gives: C[i,k] = sum_j sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Let's compute this for all i, j, k and check it equals delta (the multiplication tensor)
    # Actually the tensor T[i,j,k] should equal 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k])
    # So we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Compute the left side for all (i,j,k)
    # L[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # This is a contraction over c of three tensors.
    # Let's do it efficiently: for each c, form outer product and sum
    L = np.zeros((n, m, p))
    for c in range(rank):
        uc = U_r[:, :, c]  # (n, m)
        vc = V_r[:, :, c]  # (m, p)
        wc = W_r[:, :, c]  # (p, n)
        # L[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
        # This is: for each i,j,k: uc[i,j] * vc[j,k] * wc[k,i]
        # = einsum('ij,jk,ki->ijk', uc, vc, wc)
        L += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
    
    # The target tensor is all ones
    target = np.ones((n, m, p))
    return np.allclose(L, target, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find a lower-rank decomposition using randomized approaches
    # Strategy: use random initializations and iterative refinement (ALS-like or gradient-free)
    # For small matrices, we can try various ranks
    
    # Known bounds: for 3x5x6, the tensor rank is at most 90 (schoolbook)
    # Lower bounds: max(n*p, m*p, n*m) = max(18, 30, 15) = 30
    # So rank is between 30 and 90
    
    # Try to find a decomposition with rank < 90
    # Use a simple approach: random rank-1 terms and greedy/iterative refinement
    
    for target_rank in range(30, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try several random initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            U_r = np.random.randn(n * m, target_rank)
            V_r = np.random.randn(m * p, target_rank)
            W_r = np.random.randn(p * n, target_rank)
            
            # Iterate: fix two, solve for the third (ALS-like)
            converged = False
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each column c, U[:,c] should satisfy
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
                # This is nonlinear, so use a simple gradient step or alternating least squares approximation
                
                # Simple approach: compute current error and do a small update
                L = np.zeros((n, m, p))
                for c in range(target_rank):
                    uc = U_r[:, c].reshape(n, m)
                    vc = V_r[:, c].reshape(m, p)
                    wc = W_r[:, c].reshape(p, n)
                    L += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
                
                error = L - 1.0
                err_norm = np.linalg.norm(error)
                
                if err_norm < 1e-8:
                    converged = True
                    break
                
                # Gradient step (simplified)
                lr = 0.01
                for c in range(target_rank):
                    uc = U_r[:, c].reshape(n, m)
                    vc = V_r[:, c].reshape(m, p)
                    wc = W_r[:, c].reshape(p, n)
                    
                    # Gradient of error w.r.t. uc[i,j] = sum_k vc[j,k] * wc[k,i] * sign stuff
                    # Simplified: just do a small random perturbation weighted by error
                    grad_u = np.zeros((n, m))
                    grad_v = np.zeros((m, p))
                    grad_w = np.zeros((p, n))
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                grad_u[i, j] += error[i, j, k] * vc[j, k] * wc[k, i]
                                grad_v[j, k] += error[i, j, k] * uc[i, j] * wc[k, i]
                                grad_w[k, i] += error[i, j, k] * uc[i, j] * vc[j, k]
                    
                    U_r[:, c] -= lr * grad_u.flatten()
                    V_r[:, c] -= lr * grad_v.flatten()
                    W_r[:, c] -= lr * grad_w.flatten()
                
                if it % 20 == 0 and err_norm < 1e-4:
                    break
            
            if converged or err_norm < 1e-4:
                if _verify_fast(U_r, V_r, W_r, n, m, p, tol=1e-5):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    break
        
        if best_rank == target_rank:
            break
    
    # Convert to list format
    U_out = _to_list(best_U)
    V_out = _to_list(best_V)
    W_out = _to_list(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
