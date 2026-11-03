"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n=3, m=3, p=8, tol=1e-9):
    """Check if the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_decomposition_from_coefficients(coeffs, n=3, m=3, p=8):
    """
    Given a flat array of coefficients for rank r, build U, V, W.
    The tensor is T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    We parameterize each rank-1 term by its (i,j,k) support pattern.
    Actually, we just store the coefficients directly.
    """
    r = len(coeffs)
    U = [[0.0]*r for _ in range(n*m)]
    V = [[0.0]*r for _ in range(m*p)]
    W = [[0.0]*r for _ in range(p*n)]
    
    for c in range(r):
        # We need to extract U, V, W values from coeffs
        # This is tricky - let's use a different approach
        pass
    return U, V, W

def _random_rank1_term(n=3, m=3, p=8, scale=1.0):
    """Generate a random rank-1 tensor term."""
    # A rank-1 term is defined by vectors u (n*m), v (m*p), w (p*n)
    # T[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    u = np.zeros(n*m)
    v = np.zeros(m*p)
    w = np.zeros(p*n)
    
    # Randomly choose non-zero entries
    # For efficiency, pick a few random positions
    num_nonzero_u = random.randint(1, 4)
    num_nonzero_v = random.randint(1, 4)
    num_nonzero_w = random.randint(1, 4)
    
    for _ in range(num_nonzero_u):
        idx = random.randint(0, n*m-1)
        u[idx] = random.choice([-1, 1]) * random.uniform(0.5, 1.0) * scale
    for _ in range(num_nonzero_v):
        idx = random.randint(0, m*p-1)
        v[idx] = random.choice([-1, 1]) * random.uniform(0.5, 1.0) * scale
    for _ in range(num_nonzero_w):
        idx = random.randint(0, p*n-1)
        w[idx] = random.choice([-1, 1]) * random.uniform(0.5, 1.0) * scale
    
    return u, v, w

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook
    rank_school = n * m * p
    U = [[0.0] * rank_school for _ in range(n * m)]
    V = [[0.0] * rank_school for _ in range(m * p)]
    W = [[0.0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    
    best_rank = rank_school
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try to reduce rank by finding linear dependencies
    # Use randomized approach: try to find rank r < n*m*p decompositions
    
    # For 3x3x8, known lower bound is at least max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # Upper bound from schoolbook is 72
    # Let's try to find better decompositions
    
    # Approach: Use the fact that we can combine terms
    # Try rank values from 24 up to 71, and for each, try random initializations + local optimization
    
    # Actually, let's use a more structured approach.
    # The matrix multiplication tensor has symmetry properties.
    
    # For now, try a simple greedy reduction:
    # Check if any column in the schoolbook decomposition is linearly dependent
    
    # Better approach: Use tensor rank algorithms
    # For small dimensions, we can try exhaustive search over small rank structures
    
    # Let's try to find a decomposition of rank r by solving a system of equations
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a bilinear (actually trilinear) system.
    # We can fix two factors and solve for the third.
    
    # Strategy: Try ranks from lower bound upward, use random initialization and Newton's method
    
    def try_rank(r, max_iters=50, tol=1e-8):
        """Try to find a decomposition of rank r using random init + local optimization."""
        if time.time() > deadline - 10:
            return None
        
        # Random initialization
        U_arr = np.random.randn(n*m, r) * 0.5
        V_arr = np.random.randn(m*p, r) * 0.5
        W_arr = np.random.randn(p*n, r) * 0.5
        
        # Normalize to avoid scale issues
        for c in range(r):
            norm_u = np.linalg.norm(U_arr[:, c])
            norm_v = np.linalg.norm(V_arr[:, c])
            norm_w = np.linalg.norm(W_arr[:, c])
            if norm_u > 0 and norm_v > 0 and norm_w > 0:
                scale = (norm_u * norm_v * norm_w) ** (1.0/3.0)
                U_arr[:, c] /= scale
                V_arr[:, c] /= scale
                W_arr[:, c] /= scale
        
        # Objective: minimize sum over (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        def objective_and_grad(U_arr, V_arr, W_arr):
            obj = 0.0
            grad_U = np.zeros_like(U_arr)
            grad_V = np.zeros_like(V_arr)
            grad_W = np.zeros_like(W_arr)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                        err = s - 1.0
                        obj += err * err
                        for c in range(r):
                            grad_U[i*m+j, c] += 2 * err * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                            grad_V[j*p+k, c] += 2 * err * U_arr[i*m+j, c] * W_arr[k*n+i, c]
                            grad_W[k*n+i, c] += 2 * err * U_arr[i*m+j, c] * V_arr[j*p+k, c]
            
            return obj, grad_U, grad_V, grad_W
        
        # Gradient descent with adaptive step size
        lr = 0.01
        best_obj = float('inf')
        best_U_arr = U_arr.copy()
        best_V_arr = V_arr.copy()
        best_W_arr = W_arr.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            obj, gU, gV, gW = objective_and_grad(U_arr, V_arr, W_arr)
            
            if obj < best_obj:
                best_obj = obj
                best_U_arr = U_arr.copy()
                best_V_arr = V_arr.copy()
                best_W_arr = W_arr.copy()
            
            if obj < tol:
                break
            
            # Update
            U_arr -= lr * gU
            V_arr -= lr * gV
            W_arr -= lr * gW
            
            # Adaptive learning rate
            new_obj, _, _, _ = objective_and_grad(U_arr, V_arr, W_arr)
            if new_obj > obj * 1.1:
                lr *= 0.5
                U_arr = U_arr + lr * gU  # revert
                V_arr = V_arr + lr * gV
                W_arr = W_arr + lr * gW
            elif new_obj < obj * 0.9:
                lr *= 1.2
            else:
                lr *= 0.8
        
        if best_obj < tol:
            # Convert to list format
            U_list = best_U_arr.tolist()
            V_list = best_V_arr.tolist()
            W_list = best_W_arr.tolist()
            
            # Try to round to integers or half-integers
            U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
            V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
            W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
            
            if _verify(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                return U_rounded, V_rounded, W_rounded
            
            # Try integer rounding
            U_rounded = [[round(x) for x in row] for row in U_list]
            V_rounded = [[round(x) for x in row] for row in V_list]
            W_rounded = [[round(x) for x in row] for row in W_list]
            
            if _verify(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try ranks from 24 upward
    for r in range(24, min(rank_school, 40)):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iters=30)
        if result is not None:
            U_found, V_found, W_found = result
            best_rank = r
            best_U = U_found
            best_V = V_found
            best_W = W_found
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
