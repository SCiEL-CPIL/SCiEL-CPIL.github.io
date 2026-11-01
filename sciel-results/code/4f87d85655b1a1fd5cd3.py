"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    total_eqs = n * m * p  # 64 equations
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Build the linear system matrices once for efficiency
    # For solving U: A_U[eq_idx, i*m+j*rank + r] = V[j*p+k, r] * W[k*n+i, r]
    # For solving W: A_W[eq_idx, k*n+i*rank + r] = U[i*m+j, r] * V[j*p+k, r]
    # These change with rank, so we build them per rank
    
    # Try ranks from a reasonable lower bound up to 63
    # Theoretical lower bound for matrix multiplication tensor rank is at least max(n*m, m*p, p*n)
    # For 2x4x8: max(8, 32, 16) = 32
    # Known results: rank of 2x4x8 is at most 64 (schoolbook)
    # Let's try from 32 upward
    
    for target_rank in range(32, 64):
        if time.time() > deadline:
            break
        
        # Multiple random initializations
        max_inits = 30
        for init in range(max_inits):
            if time.time() > deadline:
                break
            
            seed_val = init * 10000 + target_rank * 100 + 42
            np.random.seed(seed_val)
            
            # Initialize with structured patterns that might help convergence
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # Scale to reasonable magnitude
            U /= np.sqrt(target_rank)
            V /= np.sqrt(target_rank)
            W /= np.sqrt(target_rank)
            
            # Alternating minimization with linear solves
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Fix V and W, solve for U (linear)
                A_U = np.zeros((total_eqs, nm * target_rank))
                b_vec = np.ones(total_eqs)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(target_rank):
                                A_U[eq_idx, (i*m+j)*target_rank + r] = V[j*p+k, r] * W[k*n+i, r]
                            eq_idx += 1
                
                if nm * target_rank >= total_eqs:
                    U_new, residuals, rank_A, sv = np.linalg.lstsq(A_U, b_vec, rcond=None)
                    U = U_new.reshape(nm, target_rank)
                
                # Fix U and V, solve for W (linear)
                A_W = np.zeros((total_eqs, pn * target_rank))
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(target_rank):
                                A_W[eq_idx, (k*n+i)*target_rank + r] = U[i*m+j, r] * V[j*p+k, r]
                            eq_idx += 1
                
                if pn * target_rank >= total_eqs:
                    W_new, residuals, rank_A, sv = np.linalg.lstsq(A_W, b_vec, rcond=None)
                    W = W_new.reshape(pn, target_rank)
                
                # Fix U and W, solve for V (linear)
                A_V = np.zeros((total_eqs, mp * target_rank))
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(target_rank):
                                A_V[eq_idx, (j*p+k)*target_rank + r] = U[i*m+j, r] * W[k*n+i, r]
                            eq_idx += 1
                
                if mp * target_rank >= total_eqs:
                    V_new, residuals, rank_A, sv = np.linalg.lstsq(A_V, b_vec, rcond=None)
                    V = V_new.reshape(mp, target_rank)
                
                # Check residual
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            residual += (val - 1.0)**2
                
                if residual < 1e-6:
                    # Found a valid decomposition!
                    best_U, best_V, best_W, best_rank = U, V, W, target_rank
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                
                # Early stopping if not improving
                if it > 20 and residual > 1e2:
                    break
    
    # Return the best we found (schoolbook fallback)
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
