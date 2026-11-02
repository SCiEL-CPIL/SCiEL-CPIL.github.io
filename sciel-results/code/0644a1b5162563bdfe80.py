"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication."""
    n, m, p = DIMENSIONS  # 3, 4, 4
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # The tensor T has entries T[i, j, k] for i in 0..n-1, j in 0..m-1, k in 0..p-1
    # T[i,j,k] = 1 if we multiply A[i,j] * B[j,k] and add to C[i,k]
    # Actually the standard matmul tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i, j, k] which is 1 for all valid (i,j,k)
    # We want rank-r decomposition: T = sum_{r=0}^{R-1} u_r ⊗ v_r ⊗ w_r
    # where u_r has shape (n*m,), v_r has shape (m*p,), w_r has shape (p*n,)
    # Using trace convention: W[k*n + i, r]
    
    # Schoolbook gives rank = n*m*p = 48
    # Let's try to find a lower rank decomposition
    
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
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank using iterative approach
    # The condition for rank-r decomposition:
    # For each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a system of trilinear equations.
    
    # Strategy: Fix U and V randomly, then solve for W.
    # For a given rank r, we have r unknowns per W entry.
    # The constraint is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is linear in W! So for fixed U,V, we can solve for W.
    
    # But we also need the decomposition to be "exact" - the tensor must match exactly.
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # Let's try random initialization and iterative refinement.
    
    def try_rank(r, max_iter=100, seed=42):
        """Try to find rank-r decomposition using iterative method."""
        rng = np.random.RandomState(seed)
        
        # Initialize U and V randomly
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Solve for W: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is a linear system in W.
            # Number of equations: n*m*p = 48
            # Number of unknowns: p*n*r = 12*r
            # For r < 4, 12*r < 48, so system is underdetermined - may have solutions
            # For r >= 4, 12*r >= 48, may have solutions
            
            # Build the linear system A * W_flat = b
            # A has shape (n*m*p, p*n*r)
            # b has shape (n*m*p,) with all 1s
            
            A = np.zeros((n * m * p, p * n * r))
            b = np.ones(n * m * p)
            
            eq = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r_idx in range(r):
                            # W[k*n+i, r_idx] is at position (k*n+i)*r + r_idx in W_flat
                            w_idx = (k * n + i) * r + r_idx
                            A[eq, w_idx] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
                        eq += 1
            
            # Solve A * W_flat = b using least squares or check consistency
            try:
                # Use least squares
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if solution is exact
                if len(residuals) > 0 and residuals[0] < 1e-10:
                    W = W_flat.reshape(p * n, r)
                    
                    # Verify the decomposition
                    # Reconstruct tensor and check
                    tensor_reconstructed = np.zeros((n * m, m * p, p * n))
                    for r_idx in range(r):
                        # outer product of U[:,r_idx], V[:,r_idx], W[:,r_idx]
                        for a in range(n * m):
                            for b_idx in range(m * p):
                                tensor_reconstructed[a, b_idx, :] += U[a, r_idx] * V[b_idx, r_idx] * W[:, r_idx]
                        # Actually this is wrong. Let me redo.
                    
                    # Better verification:
                    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = sum(U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx] for r_idx in range(r))
                                max_err = max(max_err, abs(val - 1.0))
                    
                    if max_err < 1e-10:
                        # Found valid decomposition!
                        U_list = [[float(x) for x in U[i]] for i in range(n*m)]
                        V_list = [[float(x) for x in V[i]] for i in range(m*p)]
                        W_list = [[float(x) for x in W[i]] for i in range(p*n)]
                        return {"U": U_list, "V": V_list, "W": W_list, "rank": r}
                
                # If not exact, try to update U and V
                # Use gradient-like update or just random perturbation
                U += rng.randn(n * m, r) * 0.1
                V += rng.randn(m * p, r) * 0.1
                
            except Exception:
                U = rng.randn(n * m, r)
                V = rng.randn(m * p, r)
        
        return None
    
    # Try ranks from 48 down to some lower bound
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # But realistically, for matmul, the rank is at least n*m*p / something
    
    # Try a few random seeds for each rank
    for r in range(48, 15, -1):
        if time.time() > deadline:
            break
        for seed in range(10):
            if time.time() > deadline:
                break
            result = try_rank(r, max_iter=50, seed=seed)
            if result is not None:
                best_rank = r
                best = result
                print(f"Found rank {r} decomposition with seed {seed}")
                break
        if best_rank < 48:
            break
    
    # Return the best found
    result = {"U": best["U"], "V": best["V"], "W": best["W"]}
    return result
# EVOLVE-BLOCK-END
