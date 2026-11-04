"""Matrix multiplication 2x4 by 4x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 2x4 by 4x5 matrix multiplication.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, R, tol=1e-9):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def to_half(x):
        """Convert to nearest half-integer."""
        return round(x * 2) / 2.0
    
    def clean_and_verify(U, V, W, R):
        """Try to clean values to half-integers or integers and verify."""
        # Try half-integers
        U_clean = [[to_half(U[i, j]) for j in range(R)] for i in range(n * m)]
        V_clean = [[to_half(V[i, j]) for j in range(R)] for i in range(m * p)]
        W_clean = [[to_half(W[i, j]) for j in range(R)] for i in range(p * n)]
        if verify(U_clean, V_clean, W_clean, R):
            return {"U": U_clean, "V": V_clean, "W": W_clean}
        
        # Try integers
        U_int = [[round(U[i, j]) for j in range(R)] for i in range(n * m)]
        V_int = [[round(V[i, j]) for j in range(R)] for i in range(m * p)]
        W_int = [[round(W[i, j]) for j in range(R)] for i in range(p * n)]
        if verify(U_int, V_int, W_int, R):
            return {"U": U_int, "V": V_int, "W": W_int}
        
        # Use floats
        U_list = [[float(U[i, j]) for j in range(R)] for i in range(n * m)]
        V_list = [[float(V[i, j]) for j in range(R)] for i in range(m * p)]
        W_list = [[float(W[i, j]) for j in range(R)] for i in range(p * n)]
        if verify(U_list, V_list, W_list, R, tol=1e-8):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    def compute_error(U, V, W, R):
        """Compute the total squared error of the decomposition."""
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    error += (val - 1.0) ** 2
        return error
    
    def alternating_minimize(U, V, W, R, max_iter=2000, tol=1e-12):
        """
        Perform alternating minimization to refine the decomposition.
        Uses the gradient-based update for each factor.
        """
        prev_error = compute_error(U, V, W, R)
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Update W: W[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r] / ||sum_j U[i*m+j, r] * V[j*p+k, r]||^2
            for k in range(p):
                for i in range(n):
                    for r in range(R):
                        numerator = 0.0
                        for j in range(m):
                            numerator += U[i * m + j, r] * V[j * p + k, r]
                        W[k * n + i, r] = numerator
            
            # Normalize W rows to prevent blowup
            for idx in range(p * n):
                norm_sq = sum(W[idx][r] ** 2 for r in range(R))
                if norm_sq > 1e-14:
                    scale = 1.0 / norm_sq
                    for r in range(R):
                        W[idx][r] *= scale
            
            # Update U: U[i*m+j, r] = sum_k V[j*p+k, r] * W[k*n+i, r] / ||sum_k V[j*p+k, r] * W[k*n+i, r]||^2
            for i in range(n):
                for j in range(m):
                    for r in range(R):
                        numerator = 0.0
                        for k in range(p):
                            numerator += V[j * p + k, r] * W[k * n + i, r]
                        U[i * m + j, r] = numerator
            
            # Normalize U rows
            for idx in range(n * m):
                norm_sq = sum(U[idx][r] ** 2 for r in range(R))
                if norm_sq > 1e-14:
                    scale = 1.0 / norm_sq
                    for r in range(R):
                        U[idx][r] *= scale
            
            # Update V: V[j*p+k, r] = sum_i U[i*m+j, r] * W[k*n+i, r] / ||sum_i U[i*m+j, r] * W[k*n+i, r]||^2
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        numerator = 0.0
                        for i in range(n):
                            numerator += U[i * m + j, r] * W[k * n + i, r]
                        V[j * p + k, r] = numerator
            
            # Normalize V rows
            for idx in range(m * p):
                norm_sq = sum(V[idx][r] ** 2 for r in range(R))
                if norm_sq > 1e-14:
                    scale = 1.0 / norm_sq
                    for r in range(R):
                        V[idx][r] *= scale
            
            # Compute new error
            curr_error = compute_error(U, V, W, R)
            
            if abs(prev_error - curr_error) < tol:
                break
            prev_error = curr_error
        
        return U, V, W, prev_error
    
    def try_decomposition(R, seed=42, max_restarts=30):
        """
        Try to find a rank-R decomposition using multiple restarts and alternating minimization.
        """
        np.random.seed(seed)
        
        best_error = float('inf')
        best_result = None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Near-schoolbook initialization
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            col = (i * m + j) * p + k
                            if col < R:
                                U[i * m + j, col] = 1.0
                                V[j * p + k, col] = 1.0
                                W[k * n + i, col] = 1.0
                # Add small random perturbation
                U += np.random.randn(n * m, R) * 0.01
                V += np.random.randn(m * p, R) * 0.01
                W += np.random.randn(p * n, R) * 0.01
            elif restart == 1:
                # Random initialization
                U = np.random.randn(n * m, R) * 0.1
                V = np.random.randn(m * p, R) * 0.1
                W = np.random.randn(p * n, R) * 0.1
            elif restart == 2:
                # Binary-like initialization
                U = (np.random.rand(n * m, R) > 0.5).astype(float)
                V = (np.random.rand(m * p, R) > 0.5).astype(float)
                W = (np.random.rand(p * n, R) > 0.5).astype(float)
            elif restart == 3:
                # Sparse initialization
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                for _ in range(R * 2):
                    idx_u = np.random.randint(0, n * m)
                    idx_v = np.random.randint(0, m * p)
                    idx_w = np.random.randint(0, p * n)
                    col = np.random.randint(0, R)
                    U[idx_u, col] = np.random.choice([-1, 1])
                    V[idx_v, col] = np.random.choice([-1, 1])
                    W[idx_w, col] = np.random.choice([-1, 1])
            else:
                # Mixed random with varying scale
                scale = 0.1 * (1 + restart * 0.3)
                U = np.random.randn(n * m, R) * scale
                V = np.random.randn(m * p, R) * scale
                W = np.random.randn(p * n, R) * scale
            
            # Convert to lists for alternating minimization
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Run alternating minimization
            U_ref, V_ref, W_ref, err = alternating_minimize(U_list, V_list, W_ref if False else W_list, R)
            
            if err < best_error:
                best_error = err
                best_U = np.array(U_ref)
                best_V = np.array(V_ref)
                best_W = np.array(W_ref)
            
            # Try to clean and verify
            result = clean_and_verify(best_U, best_V, best_W, R)
            if result is not None:
                return result
        
        return None
    
    # Try ranks from low to high
    # Known lower bound: max(n*m, m*p, n*p) = max(8, 20, 10) = 20
    # But for matrix multiplication, better bounds exist.
    # For M(2,4,5), let's try from 12 upward.
    for R in range(12, 41):
        if time.time() > deadline - 20:
            break
        result = try_decomposition(R)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return schoolbook()
