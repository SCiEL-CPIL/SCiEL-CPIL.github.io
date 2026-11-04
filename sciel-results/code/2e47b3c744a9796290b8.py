"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135.0  # 15s safety margin before 150s limit

    # Helper to create schoolbook decomposition
    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    # Helper to verify decomposition exactly using rational arithmetic
    def verify_exact(U, V, W, n, m, p, tol=1e-10):
        """Verify that the decomposition is exact."""
        # Check all 1-entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check all 0-entries (full check for correctness)
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i, j = divmod(a, m)
                    j2, k = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    if not (j == j2 and i == i2 and k == k2):
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        if abs(val) > tol:
                            return False
        
        return True

    # Helper to compute tensor error (Frobenius norm of difference)
    def tensor_error(U, V, W, n, m, p):
        """Compute the error in the tensor decomposition."""
        # Build the target tensor
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        # Compute the approximated tensor
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        
        return np.linalg.norm(T_approx - T)

    # Helper to convert to fraction list
    def to_fraction_list(matrix):
        result = []
        for row in matrix:
            frac_row = []
            for val in row:
                if abs(val) < 1e-10:
                    frac_row.append(0)
                else:
                    for denom in [1, 2, 4, 8, 16, 32, 64]:
                        numer = round(val * denom)
                        if abs(val - numer / denom) < 1e-6:
                            frac_row.append(numer / denom)
                            break
                    else:
                        frac_row.append(val)
            result.append(frac_row)
        return result

    # Initialize with schoolbook
    U, V, W, rank = make_schoolbook(n, m, p)
    best_result = (U.copy(), V.copy(), W.copy())
    best_rank = rank

    # Strategy: Use structured decomposition exploiting the 2x5x6 structure
    # The key insight: we can decompose the 5x6 multiplication using smaller blocks
    
    # For 2x5x6:
    # A (2x5) * B (5x6) = C (2x6)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can try to find decompositions with rank < 60
    
    # Approach 1: Try to use the structure where n=2 is small
    # For each (i,k) pair in the output, we need to compute sum_j A[i,j]*B[j,k]
    # This is like 2x6 = 12 output entries, each being a dot product of length 5
    
    # Let's try random initialization + gradient-based optimization with better initialization
    # and more careful handling of the discrete structure
    
    # First, try to find a decomposition using the fact that we can split the 5x6 product
    # into smaller parts
    
    # Try ranks from 30 down to 20
    for target_rank in range(35, 15, -1):
        if time.time() > deadline - 30:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Multiple random restarts for each rank
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(target_rank * 137 + restart * 7)
            
            # Random initialization with structured bias
            U_r = np.random.randn(n * m, target_rank) * 0.3
            V_r = np.random.randn(m * p, target_rank) * 0.3
            W_r = np.random.randn(p * n, target_rank) * 0.3
            
            # Gradient-based optimization with adaptive learning rate
            lr = 0.05
            momentum = 0.9
            vel_U = np.zeros_like(U_r)
            vel_V = np.zeros_like(V_r)
            vel_W = np.zeros_like(W_r)
            
            # Precompute target tensor
            T_target = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i * m + j, j * p + k, k * n + i] = 1.0
            
            best_err = float('inf')
            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
            
            for step in range(300):
                if time.time() > deadline - 5:
                    break
                
                # Compute T_approx
                T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                
                # Compute error
                error = T_approx - T_target
                err = np.linalg.norm(error)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                
                if err < 1e-8:
                    break
                
                # Compute gradients
                grad_U = np.einsum('abc,br,cr->ar', error, V_r, W_r)
                grad_V = np.einsum('abc,ar,cr->br', error, U_r, W_r)
                grad_W = np.einsum('abc,ar,br->cr', error, U_r, V_r)
                
                # Update with momentum
                vel_U = momentum * vel_U - lr * grad_U
                vel_V = momentum * vel_V - lr * grad_V
                vel_W = momentum * vel_W - lr * grad_W
                
                U_r += vel_U
                V_r += vel_V
                W_r += vel_W
                
                # Adaptive learning rate
                if step % 50 == 0 and step > 0:
                    lr *= 0.7
            
            # Check if we found a valid decomposition
            err = tensor_error(best_U, best_V, best_W, n, m, p)
            if err < 1e-6:
                # Verify exactly
                if verify_exact(best_U, best_V, best_W, n, m, p, tol=1e-5):
                    best_rank = target_rank
                    best_result = (best_U.copy(), best_V.copy(), best_W.copy())
                    break
        
        if best_rank <= target_rank:
            break
    
    # If we haven't found a better decomposition, try a more structured approach
    # Using the fact that n=2, we can try to exploit this
    
    if best_rank == n * m * p and time.time() < deadline - 20:
        # Try to use a decomposition based on splitting the 5x6 product
        # into 2x3 and 3x3 blocks or similar
        
        # For now, try sign-based random search for small ranks
        for target_rank in range(30, 20, -1):
            if time.time() > deadline - 10:
                break
            
            if target_rank >= best_rank:
                continue
            
            np.random.seed(target_rank * 31 + 7)
            for trial in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Use entries from {-1, 0, 1}
                U_pattern = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
                V_pattern = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
                W_pattern = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
                
                T_approx = np.einsum('ar,br,cr->abc', U_pattern, V_pattern, W_pattern)
                
                T_target = np.zeros((n * m, m * p, p * n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_target[i * m + j, j * p + k, k * n + i] = 1.0
                
                if np.allclose(T_approx, T_target, atol=1e-10):
                    if verify_exact(U_pattern, V_pattern, W_pattern, n, m, p, tol=1e-10):
                        best_rank = target_rank
                        best_result = (U_pattern.copy(), V_pattern.copy(), W_pattern.copy())
                        break
            
            if best_rank < n * m * p:
                break
    
    # Final polish: try to clean up the solution
    if best_rank < n * m * p:
        U_final, V_final, W_final = best_result
        # Try to snap to rational values
        for mat in [U_final, V_final, W_final]:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    for denom in [1, 2, 4, 8, 16]:
                        numer = round(mat[i, j] * denom)
                        if abs(mat[i, j] - numer / denom) < 1e-4:
                            mat[i, j] = numer / denom
                            break
    
    U_final, V_final, W_final = best_result
    return {"U": to_fraction_list(U_final), "V": to_fraction_list(V_final), "W": to_fraction_list(W_final)}

def tensor_error_target(n, m, p):
    """Create the target multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T
# EVOLVE-BLOCK-END
