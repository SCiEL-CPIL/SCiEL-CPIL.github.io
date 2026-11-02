"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Build the target multiplication tensor
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # For matrix mult A(n,m) * B(m,p) = C(n,p): C[i,k] = sum_j A[i,j]B[j,k]
    # The tensor is T[i*m+j, j*p+k, k*n+i] = 1 (for valid i,j,k)
    # Note: output index c = k*n + i (trace convention W[k*n+i])
    
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U ⊗ V ⊗ W reconstructs the multiplication tensor exactly."""
        if isinstance(U, list):
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
        else:
            U_arr = U.astype(float)
            V_arr = V.astype(float)
            W_arr = W.astype(float)
        
        r = U_arr.shape[1]
        T_recon = np.zeros((n*m, m*p, p*n))
        for t in range(r):
            T_recon += np.einsum('a,b,c->abc', U_arr[:,t], V_arr[:,t], W_arr[:,t])
        
        if np.allclose(T_recon, T_target, atol=tol, rtol=0):
            return True
        return False
    
    def schoolbook_decomposition():
        """Return the standard rank-54 schoolbook decomposition."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Start with schoolbook as baseline
    U_school, V_school, W_school, school_rank = schoolbook_decomposition()
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = school_rank
    
    # Strategy: Try to find a lower-rank decomposition using iterative refinement
    # Approach 1: Try rank 40-53 with random initialization and optimization
    
    def try_rank_decomposition(target_rank, num_trials=3, max_iter=200):
        """Try to find a rank-target_rank decomposition using iterative refinement."""
        if target_rank >= best_rank:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(trial * 1000 + 42)
            
            # Initialize with small random values
            U_t = np.random.randn(n*m, target_rank) * 0.1
            V_t = np.random.randn(m*p, target_rank) * 0.1
            W_t = np.random.randn(p*n, target_rank) * 0.1
            
            # Iterative refinement: update one factor at a time
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                # U[:,t] should minimize ||T - sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t]||
                # For each t: U[:,t] = (sum over a,b,c of T[a,b,c] * V[b,t] * W[c,t]) / (||V[:,t]||^2 * ||W[:,t]||^2)
                for t in range(target_rank):
                    vw = np.outer(V_t[:,t], W_t[:,t])
                    U_new_t = np.einsum('abc,bc->a', T_target, vw)
                    denom = np.dot(V_t[:,t], V_t[:,t]) * np.dot(W_t[:,t], W_t[:,t])
                    if denom > 1e-15:
                        U_t[:,t] = U_new_t / denom
                
                # Update V
                for t in range(target_rank):
                    uw = np.outer(U_t[:,t], W_t[:,t])
                    V_new_t = np.einsum('abc,ac->b', T_target, uw)
                    denom = np.dot(U_t[:,t], U_t[:,t]) * np.dot(W_t[:,t], W_t[:,t])
                    if denom > 1e-15:
                        V_t[:,t] = V_new_t / denom
                
                # Update W
                for t in range(target_rank):
                    uv = np.outer(U_t[:,t], V_t[:,t])
                    W_new_t = np.einsum('abc,ab->c', T_target, uv)
                    denom = np.dot(U_t[:,t], U_t[:,t]) * np.dot(V_t[:,t], V_t[:,t])
                    if denom > 1e-15:
                        W_t[:,t] = W_new_t / denom
                
                # Check convergence
                if it % 20 == 19:
                    T_recon = np.zeros((n*m, m*p, p*n))
                    for t in range(target_rank):
                        T_recon += np.einsum('a,b,c->abc', U_t[:,t], V_t[:,t], W_t[:,t])
                    error = np.linalg.norm(T_recon - T_target)
                    if error < 1e-6:
                        return U_t, V_t, W_t, target_rank
        
        return None
    
    # Try ranks from 45 down to 40
    for target_rank in range(45, 39, -1):
        if time.time() > deadline - 30:
            break
        
        result = try_rank_decomposition(target_rank, num_trials=2, max_iter=100)
        if result is not None:
            U_t, V_t, W_t, r = result
            if check_decomposition(U_t, V_t, W_t, tol=1e-6):
                best_U, best_V, best_W = U_t, V_t, W_t
                best_rank = r
                break
    
    # Strategy 2: Try a structured decomposition based on block structure
    # Split the 3x6 matrix into two 3x3 blocks and use a shared A factor
    # This might allow rank reduction below 2*23=46
    
    if time.time() < deadline - 20 and best_rank >= 46:
        # Try to find a decomposition with rank 44 or 45 using a more structured approach
        # Use the fact that the same A matrix is used for both halves
        
        for target_rank in range(45, 43, -1):
            if time.time() > deadline - 15:
                break
            
            # Initialize with a structured pattern
            U_t = np.zeros((n*m, target_rank))
            V_t = np.zeros((m*p, target_rank))
            W_t = np.zeros((p*n, target_rank))
            
            # First half: for k=0,1,2 (first 3 columns of B)
            # Second half: for k=3,4,5 (last 3 columns of B)
            
            # Use schoolbook-like initialization but with some sharing
            for col in range(target_rank):
                if col < 27:
                    # First 3x3 block
                    i = col // 9
                    j = (col // 3) % 3
                    k = col % 3
                    U_t[i*m+j, col] = 1.0
                    V_t[j*p+k, col] = 1.0
                    W_t[k*n+i, col] = 1.0
                elif col < 54 and col < target_rank:
                    # Second 3x3 block
                    col2 = col - 27
                    i = col2 // 9
                    j = (col2 // 3) % 3
                    k = col2 % 3 + 3
                    U_t[i*m+j, col] = 1.0
                    V_t[j*p+k, col] = 1.0
                    W_t[k*n+i, col] = 1.0
            
            # Add some random perturbation to break symmetry
            np.random.seed(123)
            U_t += np.random.randn(n*m, target_rank) * 0.01
            V_t += np.random.randn(m*p, target_rank) * 0.01
            W_t += np.random.randn(p*n, target_rank) * 0.01
            
            # Refine
            for it in range(50):
                if time.time() > deadline - 15:
                    break
                
                for t in range(target_rank):
                    vw = np.outer(V_t[:,t], W_t[:,t])
                    U_new_t = np.einsum('abc,bc->a', T_target, vw)
                    denom = np.dot(V_t[:,t], V_t[:,t]) * np.dot(W_t[:,t], W_t[:,t])
                    if denom > 1e-15:
                        U_t[:,t] = U_new_t / denom
                
                for t in range(target_rank):
                    uw = np.outer(U_t[:,t], W_t[:,t])
                    V_new_t = np.einsum('abc,ac->b', T_target, uw)
                    denom = np.dot(U_t[:,t], U_t[:,t]) * np.dot(W_t[:,t], W_t[:,t])
                    if denom > 1e-15:
                        V_t[:,t] = V_new_t / denom
                
                for t in range(target_rank):
                    uv = np.outer(U_t[:,t], V_t[:,t])
                    W_new_t = np.einsum('abc,ab->c', T_target, uv)
                    denom = np.dot(U_t[:,t], U_t[:,t]) * np.dot(V_t[:,t], V_t[:,t])
                    if denom > 1e-15:
                        W_t[:,t] = W_new_t / denom
            
            # Check
            if check_decomposition(U_t, V_t, W_t, tol=1e-6):
                best_U, best_V, best_W = U_t, V_t, W_t
                best_rank = target_rank
                break
    
    # Convert to lists with integer/half-integer entries if possible
    def to_fraction_list(arr):
        """Convert numpy array to list of Fractions (integers or half-integers)."""
        result = []
        for row in arr:
            result_row = []
            for val in row:
                # Round to nearest half-integer
                val_rounded = round(val * 2) / 2
                if abs(val - val_rounded) < 1e-6:
                    result_row.append(Fraction(val_rounded))
                else:
                    # Keep as float if not close to half-integer
                    result_row.append(float(val))
            result.append(result_row)
        return result
    
    # For the return, use the best decomposition found
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
