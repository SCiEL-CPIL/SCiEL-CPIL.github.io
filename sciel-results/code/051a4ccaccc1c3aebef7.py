import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def get_rank50():
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # Tile 2x4 x 4x7:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) separately with rank-1 terms
        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    def try_lower_rank_local_search(U_init, V_init, W_init, max_rank_to_try=48):
        """Attempt to find a lower rank decomposition via local search from a high-rank init."""
        best_rank = None
        best_decomp = None
        
        # Get the rank 50 solution for initialization
        U50, V50, W50 = U_init, V_init, W_init
        
        for target_rank in range(49, max_rank_to_try - 1, -1):
            if time.time() > deadline:
                break
            
            # We try a few different initializations for the lower rank
            # 1. Random linear combinations of the rank 50 vectors
            # This ensures the initial subspace is related to the known solution
            for attempt in range(5):
                if time.time() > deadline:
                    break
                
                np.random.seed(42 + target_rank * 100 + attempt)
                
                # Generate random mixing matrices to create new basis vectors
                # The new vectors are linear combinations of the old ones
                # This keeps the initial guess "close" to the valid manifold
                M_U = np.random.randn(50, target_rank)
                M_V = np.random.randn(50, target_rank)
                M_W = np.random.randn(50, target_rank)
                
                # However, simply mixing U, V, W independently might break the rank structure 
                # if we don't do it consistently. 
                # A better approach: Start with the first 'target_rank' columns? 
                # Or better: Use the fact that if a rank R decomposition exists, 
                # any full rank linear transformation of the factor matrices (with consistent inverse transforms) 
                # is also a decomposition. 
                # But we are reducing rank. 
                
                # Let's try a different initialization: 
                # Take a random subset of the 50 columns? No, that leaves error.
                # Let's try to optimize a rank 'target_rank' decomposition 
                # initialized by projecting the rank 50 solution onto a random subspace?
                
                # Simplest robust init: 
                # Take the first 'target_rank' columns of U50, V50, W50.
                # This is a rank 'target_rank' approximation (with error).
                # Then polish with gradient descent.
                
                U = U50[:, :target_rank].copy()
                V = V50[:, :target_rank].copy()
                W = W50[:, :target_rank].copy()
                
                # Flatten for scipy minimize
                def loss_and_grad(params):
                    # Reshape
                    U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
                    V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_flat = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    recon = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
                    error = recon - T
                    loss = np.sum(error**2)
                    
                    # Gradients
                    dU = np.einsum('abc,br,cr->ar', error, V_flat, W_flat)
                    dV = np.einsum('abc,ar,cr->br', error, U_flat, W_flat)
                    dW = np.einsum('abc,ar,br->cr', error, U_flat, V_flat)
                    
                    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                    return loss, grad

                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                try:
                    res = minimize(loss_and_grad, x0, jac=True, method='L-BFGS-B', 
                                   options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                    
                    if res.success or res.fun < 1e-8:
                        # Extract solution
                        U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                        V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                        W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                        
                        if verify_decomp(U_opt, V_opt, W_opt):
                            best_rank = target_rank
                            best_decomp = (U_opt, V_opt, W_opt)
                            break # Found a solution for this rank, try lower
                            
                except Exception:
                    pass

            if best_rank is not None and best_rank == target_rank:
                # Found a solution for this rank, we can stop trying this rank and go lower
                # But wait, if we found rank 49, we should try 48.
                # The loop structure handles this by continuing to next target_rank if not broken.
                # If we break inner loop, we should check if we want to continue to lower rank.
                # Yes, we want the MINIMAL rank.
                pass

        return best_decomp

    # Get the rank-50 decomposition
    U50, V50, W50 = get_rank50()

    # Verify it's correct
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a lower rank decomposition if we have time
    if time.time() < deadline - 10:
        lower_decomp = try_lower_rank_local_search(U50, V50, W50, max_rank_to_try=45)
        if lower_decomp is not None:
            U, V, W = lower_decomp
            if verify_decomp(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
