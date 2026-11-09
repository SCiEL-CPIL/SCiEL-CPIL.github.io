"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 40
    
    # Deadline guard
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0
    
    def check_deadline():
        return time.time() + safety_margin > deadline
    
    # Build the target tensor T[i*m+j, j*p+k, k*n+i] for matrix mult
    # For 2x4x5: A is 2x4, W is 4x5, result is 2x5
    # T has shape (2*4, 4*5, 5*2) = (8, 20, 10)
    # T[i*4+j, j*5+k, k*2+i] = 1 for i in [0,1], j in [0,3], k in [0,4]
    
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m + j, j*p + k, k*n + i] = 1.0
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook as fallback
    U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    def try_find_decomposition(rank, max_restarts=50, max_steps=5000):
        """Try to find a rank-rank decomposition using gradient descent with momentum."""
        if check_deadline():
            return None
        
        # Try different learning rate schedules and initializations
        for restart in range(max_restarts):
            if check_deadline():
                return None
            
            # Use different seeds for diversity
            seed = rank * 10000 + restart * 7 + 42
            np.random.seed(seed)
            
            # Initialize with small random values
            U_r = np.random.randn(n*m, rank) * 0.5
            V_r = np.random.randn(m*p, rank) * 0.5
            W_r = np.random.randn(p*n, rank) * 0.5
            
            # Momentum variables
            vel_U = np.zeros_like(U_r)
            vel_V = np.zeros_like(V_r)
            vel_W = np.zeros_like(W_r)
            
            # Learning rate with decay
            base_lr = 0.05
            momentum = 0.9
            
            for step in range(max_steps):
                if check_deadline():
                    return None
                
                # Compute reconstruction efficiently
                recon = np.zeros_like(target)
                for r in range(rank):
                    recon += np.einsum('a,b,c->abc', U_r[:, r], V_r[:, r], W_r[:, r])
                
                err = recon - target
                loss = np.sum(err**2)
                
                if loss < 1e-12:
                    break
                
                # Learning rate decay
                lr = base_lr * (0.995 ** step)
                
                # Compute gradients
                grad_U = np.zeros_like(U_r)
                grad_V = np.zeros_like(V_r)
                grad_W = np.zeros_like(W_r)
                
                for r in range(rank):
                    # d/dU[r] = (err * V[r] outer W[r])
                    err_contracted_b = np.einsum('abc,b->ac', err, V_r[:, r])
                    grad_U[:, r] = err_contracted_b * W_r[:, r]
                    
                    err_contracted_a = np.einsum('abc,a->bc', err, U_r[:, r])
                    grad_V[:, r] = err_contracted_a * W_r[:, r]
                    
                    err_contracted_c = np.einsum('abc,c->ab', err, W_r[:, r])
                    grad_W[:, r] = err_contracted_c * V_r[:, r]
                
                # Update with momentum
                vel_U = momentum * vel_U - lr * grad_U
                vel_V = momentum * vel_V - lr * grad_V
                vel_W = momentum * vel_W - lr * grad_W
                
                U_r += vel_U
                V_r += vel_V
                W_r += vel_W
                
                # Early stopping if loss is very small
                if step % 200 == 0 and loss < 1e-8:
                    break
            
            # Check if we got a good solution
            recon = np.zeros_like(target)
            for r in range(rank):
                recon += np.einsum('a,b,c->abc', U_r[:, r], V_r[:, r], W_r[:, r])
            
            max_err = np.max(np.abs(recon - target))
            if max_err < 1e-6:
                # Try to find exact rational representation
                result = _rationalize(U_r, V_r, W_r, target, rank)
                if result is not None:
                    return result
        
        return None
    
    def _rationalize(U, V, W, target, rank):
        """Try to find exact rational/integer/half-integer representation."""
        # First, try to round to half-integers directly
        for denom in [1, 2, 4, 8, 16, 32, 64]:
            U_s = np.round(U * denom) / denom
            V_s = np.round(V * denom) / denom
            W_s = np.round(W * denom) / denom
            
            # Check if this gives exact decomposition
            recon_s = np.zeros_like(target)
            for r in range(rank):
                recon_s += np.einsum('a,b,c->abc', U_s[:, r], V_s[:, r], W_s[:, r])
            
            if np.max(np.abs(recon_s - target)) < 1e-10:
                # Verify with exact arithmetic using fractions
                if _verify_exact(U_s, V_s, W_s, target, rank, denom):
                    return (U_s, V_s, W_s)
        
        # If simple rounding fails, try more sophisticated rational approximation
        # Use continued fractions or similar
        return None
    
    def _verify_exact(U, V, W, target, rank, denom):
        """Verify exactness using Fraction arithmetic."""
        try:
            # Convert to fractions
            U_frac = np.zeros((n*m, rank), dtype=object)
            V_frac = np.zeros((m*p, rank), dtype=object)
            W_frac = np.zeros((p*n, rank), dtype=object)
            
            for a in range(n*m):
                for r in range(rank):
                    U_frac[a, r] = Fraction(int(round(U[a, r] * denom)), denom)
            for b in range(m*p):
                for r in range(rank):
                    V_frac[b, r] = Fraction(int(round(V[b, r] * denom)), denom)
            for c in range(p*n):
                for r in range(rank):
                    W_frac[c, r] = Fraction(int(round(W[c, r] * denom)), denom)
            
            # Verify the tensor decomposition exactly
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Target value at (i*m+j, j*p+k, k*n+i) should be 1
                        a = i*m + j
                        b = j*p + k
                        c = k*n + i
                        
                        # Compute sum over r of U[a,r] * V[b,r] * W[c,r]
                        total = Fraction(0, 1)
                        for r in range(rank):
                            total += U_frac[a, r] * V_frac[b, r] * W_frac[c, r]
                        
                        if total != 1:
                            return False
                        
                        # Also check that other entries are 0
                        # (This is expensive, so we'll skip for now and rely on the numeric check)
            
            return True
        except:
            return False
    
    # Try ranks from low to high
    # Based on known results, the rank for 2x4x5 matrix multiplication might be around 32-38
    # Let's try a range of ranks
    for rank in range(28, rank_schoolbook + 1):
        if check_deadline():
            break
        
        result = try_find_decomposition(rank)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = rank
            best_U = U_r
            best_V = V_r
            best_W = W_r
            break  # Found the minimal rank, stop
    
    # Convert to list-of-lists format with integer/half-integer entries
    if best_U is not None and best_rank < rank_schoolbook:
        # Ensure entries are integer or half-integer
        def clean_matrix(mat):
            # Round to nearest half-integer
            mat_rounded = np.round(mat * 2) / 2
            return mat_rounded.tolist()
        
        return {"U": clean_matrix(best_U), "V": clean_matrix(best_V), "W": clean_matrix(best_W)}
    else:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
