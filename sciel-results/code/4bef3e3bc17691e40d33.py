"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Convention: W[k*n+i, r]
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find minimum rank decomposition
    # Start with schoolbook and try to reduce
    
    def make_decomposition(rank, U=None, V=None, W=None):
        """Create a decomposition with given rank."""
        if U is None:
            U = np.zeros((n*m, rank))
        if V is None:
            V = np.zeros((m*p, rank))
        if W is None:
            W = np.zeros((p*n, rank))
        return U, V, W
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition is exact."""
        rank = U.shape[1]
        for c in range(rank):
            pass
        # Compute the reconstructed tensor
        # T_recon[a, b, c_idx] = sum_r U[a, r] * V[b, r] * W[c_idx, r]
        # This is expensive, so let's be smart
        # For each (a, b), check if the resulting vector matches T[a, b, :]
        
        # More efficient: for each r, compute outer product
        # T_recon = sum_r U[:, r, None] * V[None, :, r] * W[None, None, r]
        # But this is O(rank * n*m * m*p * p*n) which is 64*8*32*16 = 262144, fine
        
        # Actually, let's just verify the constraint directly
        # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
        
        # Vectorized: 
        # For each r, the contribution to T is U[:, r] outer V[:, r] outer W[:, r]
        # Sum over r
        
        # Let's do it mode by mode for efficiency
        # T_recon[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # Reshape approach:
        # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        # But this might be slow for large rank
        
        # Alternative: for each c, T_recon[:, :, c] = (U * W[c, :][None, :]) @ V.T
        # i.e., T_recon[:, :, c] = U @ diag(W[c]) @ V.T
        
        errors = []
        for c_idx in range(p*n):
            # T_recon[:, :, c_idx] = U @ diag(W[c_idx]) @ V.T
            # = (U * W[c_idx][None, :]) @ V.T
            recon = (U * W[c_idx][None, :]) @ V.T
            target = T[:, :, c_idx]
            err = np.max(np.abs(recon - target))
            if err > tol:
                errors.append((c_idx, err))
        
        if errors:
            return False, errors
        return True, []
    
    # Start with schoolbook (rank 64)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Verify schoolbook
    valid, _ = check_decomposition(U_sb, V_sb, W_sb)
    assert valid, "Schoolbook decomposition should be valid"
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try to find lower rank using numerical optimization
    # Strategy: try ranks from low to high, use random initialization and gradient descent
    
    from scipy.optimize import minimize
    
    def flatten_params(U, V, W):
        """Flatten U, V, W into a single vector."""
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unflatten_params(params):
        """Unflatten params back to U, V, W."""
        u_size = n*m
        v_size = m*p
        w_size = p*n
        u_flat = params[:u_size]
        v_flat = params[u_size:u_size+v_size]
        w_flat = params[u_size+v_size:u_size+v_size+w_size]
        rank = u_size // (n*m)  # This doesn't work, need to pass rank
        return None  # We'll handle this differently
    
    # Better approach: parameterize by rank
    def try_rank_optimization(target_rank, max_iter=500, tol=1e-8):
        """Try to find a decomposition of given rank using optimization."""
        if time.time() > deadline - 5:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective_and_grad(params):
            u_size = n*m
            v_size = m*p
            w_size = p*n
            U = params[:u_size].reshape(n*m, target_rank)
            V = params[u_size:u_size+v_size].reshape(m*p, target_rank)
            W = params[u_size+v_size:].reshape(p*n, target_rank)
            
            # Compute error
            # T_recon[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Error = ||T_recon - T||_F^2
            
            # Efficient computation:
            # For each c: T_recon[:, :, c] = (U * W[c][None, :]) @ V.T
            # error_c = ||(U * W[c][None, :]) @ V.T - T[:, :, c]||_F^2
            
            total_error = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for c_idx in range(p*n):
                # recon = (U * W[c_idx][None, :]) @ V.T
                UW = U * W[c_idx][None, :]  # (n*m, target_rank)
                recon = UW @ V.T  # (n*m, m*p)
                diff = recon - T[:, :, c_idx]  # (n*m, m*p)
                total_error += np.sum(diff**2)
                
                # Gradients:
                # d/dU: diff @ V * W[c_idx][None, :]
                # d/dV: (U * W[c_idx][None, :]).T @ diff
                # d/dW[c_idx]: (U * V.T) * ... hmm, let me think
                
                # recon = (U * W[c][None,:]) @ V.T
                # Let M = U * W[c][None,:]
                # recon = M @ V.T
                # d(recon)/dU = diag(W[c]) @ V.T  ... no
                
                # Actually: recon[a,b] = sum_r U[a,r] * W[c,r] * V[b,r]
                # d(recon[a,b])/d(U[a',r']) = delta(a,a') * W[c,r'] * V[b,r']
                # d(recon[a,b])/d(V[b',r']) = delta(b,b') * U[a,r'] * W[c,r']
                # d(recon[a,b])/d(W[c,r']) = U[a,r'] * V[b,r']
                
                # grad_U[a, r'] = sum_b diff[a,b] * W[c,r'] * V[b,r']
                #                = diff @ (W[c][None,:] * V).T  ... 
                # Let me be more careful:
                # grad_U[a, r'] = sum_b diff[a,b] * W[c,r'] * V[b,r']
                #               = W[c,r'] * sum_b diff[a,b] * V[b,r']
                #               = W[c,r'] * (diff @ V)[:, r']  ... no
                # sum_b diff[a,b] * V[b,r'] = (diff @ V) [a, r']? No, V is (m*p, target_rank)
                # (diff @ V)[a, r'] = sum_b diff[a,b] * V[b, r']  -- but b ranges over m*p, and V has m*p rows. Yes!
                
                # So: grad_U[:, r'] = W[c, r'] * (diff @ V)[:, r']
                # i.e., grad_U = (diff @ V) * W[c][None, :]
                
                # grad_V[b, r'] = sum_a diff[a,b] * U[a,r'] * W[c,r']
                #               = W[c,r'] * sum_a diff[a,b] * U[a,r']
                #               = W[c,r'] * (diff.T @ U)[b, r']
                # i.e., grad_V = (diff.T @ U) * W[c][None, :]
                
                # grad_W[c, r'] = sum_{a,b} diff[a,b] * U[a,r'] * V[b,r']
                #               = sum_a sum_b diff[a,b] * U[a,r'] * V[b,r']
                #               = (U.T @ diff @ V)[r', r']  ... no
                # Let me define: for fixed r', grad_W[c, r'] = sum_{a,b} diff[a,b] * U[a,r'] * V[b,r']
                # = sum_a U[a,r'] * (sum_b diff[a,b] * V[b,r'])
                # = sum_a U[a,r'] * (diff @ V)[a, r']
                # = (U * (diff @ V)).sum()  ... no, that's not right either
                
                # Actually: grad_W[c, r'] = sum_{a,b} diff[a,b] * U[a,r'] * V[b,r']
                # Let M = diff @ V  (shape: n*m x target_rank)
                # M[a, r'] = sum_b diff[a,b] * V[b,r']
                # grad_W[c, r'] = sum_a U[a,r'] * M[a, r']
                # = np.sum(U * M)
                
                dV = diff @ V  # (n*m, target_rank)
                grad_U += dV * W[c_idx][None, :]
                
                dU = diff.T @ U  # (m*p, target_rank)
                grad_V += dU * W[c_idx][None, :]
                
                # grad_W[c_idx, r'] = sum_{a,b} diff[a,b] * U[a,r'] * V[b,r']
                # = np.sum(diff * np.outer(U[:, r'], V[:, r'])) for each r'
                # = np.sum(U[:, r'][None, :] * V[:, r'][None, :] * diff) for each r'
                # Let's compute: for each r', grad_W[c_idx, r'] = np.sum(diff * (U[:, r'] * V[:, r']))
                # Vectorized: grad_W[c_idx, :] = np.sum(diff[None, :, :] * U[:, :, None] * V[:, None, :], axis=(1,2))
                # That's expensive. Alternative:
                # grad_W[c_idx, r'] = sum_a U[a,r'] * (diff @ V)[a, r'] = sum_a U[a,r'] * dV[a,r']
                # = np.sum(U * dV)
                
                grad_W[c_idx, :] = np.sum(U * dV)
            
            total_error *= 0.5  # For gradient descent convenience
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return total_error, grad
        
        # Use L-BFGS-B
        result = minimize(objective_and_grad, params, jac=True, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.success or result.fun < tol:
            params_opt = result.x
            U_opt = params_opt[:n*m].reshape(n*m, target_rank)
            V_opt = params_opt[n*m:n*m+m*p].reshape(m*p, target_rank)
            W_opt = params_opt[n*m+m*p:].reshape(p*n, target_rank)
            
            valid, _ = check_decomposition(U_opt, V_opt, W_opt)
            if valid:
                return U_opt, V_opt, W_opt
        
        return None
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(8, 32, 16) = 32
    # But actually the tensor rank lower bound is different
    
    # Let's try a range of ranks
    for target_rank in range(32, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_optimization(target_rank, max_iter=200)
        if result is not None:
            U_opt, V_opt, W_opt = result
            best_U, best_V, best_W = U_opt, V_opt, W_opt
            best_rank = target_rank
            break  # Found a valid decomposition, stop
    
    # If we found a lower rank, use it; otherwise use schoolbook
    if best_rank < rank_schoolbook:
        U, V, W = best_U, best_V, best_W
    else:
        U, V, W = U_sb, V_sb, W_sb
    
    # Convert to lists with integer/half-integer entries
    # Round to nearest half-integer
    def to_half_integer_list(matrix):
        """Convert numpy matrix to list of lists with half-integer entries."""
        result = []
        for row in matrix:
            row_list = []
            for val in row:
                # Round to nearest 0.5
                rounded = round(val * 2) / 2
                if abs(rounded - round(rounded)) < 1e-9:
                    val_out = int(round(rounded))
                else:
                    val_out = rounded
                row_list.append(val_out)
            result.append(row_list)
        return result
    
    return {"U": to_half_integer_list(U), "V": to_half_integer_list(V), "W": to_half_integer_list(W)}
# EVOLVE-BLOCK-END
