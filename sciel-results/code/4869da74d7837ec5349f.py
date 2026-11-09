"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Build the matrix multiplication tensor
    # T[i,j,k] = 1 for all valid (i,j,k)
    
    # Index helpers
    def idx_u(i, j):
        return i * m + j
    
    def idx_v(j, k):
        return j * p + k
    
    def idx_w(k, i):
        return k * n + i
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[idx_u(i, j), col] = 1.0
                V_sb[idx_v(j, k), col] = 1.0
                W_sb[idx_w(k, i), col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # Greedy rank minimization
    # Try to find a decomposition with lower rank
    # Strategy: iteratively add rank-1 components that cover the most remaining terms
    
    # For exact decomposition with integer/half-integer entries,
    # we need to be careful
    
    # Try candidate ranks from lower bound up to schoolbook
    lower_bound = max(n * m, m * p, p * n)  # = 24
    
    # Check if we have time for search
    elapsed = time.time() - start_time
    if elapsed < 100:  # Still have time
        # Try a few candidate ranks
        for candidate_rank in range(lower_bound, schoolbook_rank + 1, 6):
            if time.time() - start_time > deadline - 30:
                break
            
            # Try to construct a decomposition with this rank
            # Using a greedy heuristic
            
            # Initialize
            U = np.zeros((n * m, candidate_rank))
            V = np.zeros((m * p, candidate_rank))
            W = np.zeros((p * n, candidate_rank))
            
            # Residual tensor (what still needs to be satisfied)
            residual = np.ones((n, m, p))
            
            # Greedy: at each step, find the rank-1 component that
            # maximizes the number of satisfied terms
            
            for step in range(candidate_rank):
                if time.time() - start_time > deadline - 10:
                    break
                
                # Find best rank-1 component
                best_score = -1
                best_u, best_v, best_w = None, None, None
                
                # Try all possible simple rank-1 components
                # A rank-1 component has u[i,j], v[j,k], w[k,i]
                # For simplicity, try binary or small integer values
                
                # This is computationally expensive, so limit search
                # Try a few random candidates
                
                np.random.seed(42 + step)
                for trial in range(100):
                    if time.time() - start_time > deadline - 5:
                        break
                    
                    # Random initialization
                    u = np.random.choice([0, 1, 0.5], size=(n * m,))
                    v = np.random.choice([0, 1, 0.5], size=(m * p,))
                    w = np.random.choice([0, 1, 0.5], size=(p * n,))
                    
                    # Compute how many terms this component satisfies
                    # A term (i,j,k) is satisfied if u[i,j]*v[j,k]*w[k,i] matches residual
                    score = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = u[idx_u(i,j)] * v[idx_v(j,k)] * w[idx_w(k,i)]
                                if abs(val - residual[i,j,k]) < 1e-9:
                                    score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
                
                if best_u is None:
                    break
                
                # Add this component
                U[:, step] = best_u
                V[:, step] = best_v
                W[:, step] = best_w
                
                # Update residual
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residual[i,j,k] -= best_u[idx_u(i,j)] * best_v[idx_v(j,k)] * best_w[idx_w(k,i)]
                
                # Check if we've satisfied all terms
                if np.allclose(residual, 0, atol=1e-9):
                    if step + 1 < best_rank:
                        best_rank = step + 1
                        best_U = U[:, :step+1].copy()
                        best_V = V[:, :step+1].copy()
                        best_W = W[:, :step+1].copy()
                    break
        
        # Check if we have a valid decomposition
        if best_rank < schoolbook_rank:
            # Verify
            verified = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(best_U[idx_u(i,j), :] * best_V[idx_v(j,k), :] * best_W[idx_w(k,i), :])
                        if abs(val - 1.0) > 1e-9:
                            verified = False
                            break
                if not verified:
                    break
            
            if verified:
                U_list = best_U.tolist()
                V_list = best_V.tolist()
                W_list = best_W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook decomposition
    U_list = U_sb.tolist()
    V_list = V_sb.tolist()
    W_list = W_sb.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
