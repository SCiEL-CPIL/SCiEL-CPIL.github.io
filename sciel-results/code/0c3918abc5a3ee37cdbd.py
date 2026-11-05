"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from itertools import product
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W, rank
    
    # Check if a decomposition is valid
    def check_valid(U, V, W, tol=1e-10):
        r = U.shape[1]
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        T_target = np.ones((n, m, p))
        return np.allclose(T_rec, T_target, atol=tol)
        
    def snap_to_half_int(A):
        return np.round(A * 2) / 2.0

    # Try to find a lower rank decomposition using optimization
    def try_optimize(rank, time_budget=20, seed=None):
        """Try to find a valid decomposition of given rank using optimization."""
        if time.time() > deadline:
            return None
            
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(int(time.time() * 1000) % 100000)

        # Initialize with random values
        scale = 1.0
        U = np.random.randn(n * m, rank) * scale
        V = np.random.randn(m * p, rank) * scale
        W = np.random.randn(p * n, rank) * scale
        
        # Objective: minimize ||T_rec - T_target||^2
        def objective(params):
            U_flat, V_flat, W_flat = params
            U_3d = U_flat.reshape(n, m, rank)
            V_3d = V_flat.reshape(m, p, rank)
            W_3d = W_flat.reshape(p, n, rank)
            T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            T_target = np.ones((n, m, p))
            err = T_rec - T_target
            return np.sum(err ** 2)
        
        # Gradient
        def gradient(params):
            U_flat, V_flat, W_flat = params
            U_3d = U_flat.reshape(n, m, rank)
            V_3d = V_flat.reshape(m, p, rank)
            W_3d = W_flat.reshape(p, n, rank)
            T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            T_target = np.ones((n, m, p))
            err = T_rec - T_target
            
            # dT/dU[i,j,c] = V[j,k,c] * W[k,i,c] summed over k
            dU = 2 * np.einsum('ijk,jkc->ijc', err, W_3d)
            dV = 2 * np.einsum('ijk,kic->jkc', err, U_3d)
            dW = 2 * np.einsum('ijk,jkc->kic', err, V_3d)
            
            return [dU.flatten(), dV.flatten(), dW.flatten()]
        
        # Flatten parameters
        def pack(U, V, W):
            return [U.flatten(), V.flatten(), W.flatten()]
        
        def unpack(params):
            U_flat, V_flat, W_flat = params
            return U_flat.reshape(n, m, rank), V_flat.reshape(m, p, rank), W_flat.reshape(p, n, rank)
        
        # Use L-BFGS-B optimization
        try:
            result = opt.minimize(
                objective,
                pack(U, V, W),
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            # Check if optimization converged to a low error
            if result.fun < 1e-4:
                U_opt, V_opt, W_opt = unpack(result.x)
                
                # Snap to half-integers
                U_snap = snap_to_half_int(U_opt)
                V_snap = snap_to_half_int(V_opt)
                W_snap = snap_to_half_int(W_opt)
                
                # Verify snapped version
                if check_valid(U_snap, V_snap, W_snap, tol=1e-6):
                    return U_snap.reshape(n*m, rank), V_snap.reshape(m*p, rank), W_snap.reshape(p*n, rank), rank
        except Exception:
            pass
        
        return None
    
    # Try various ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    min_possible = max(n * m, m * p, p * n)
    
    best_result = None
    best_rank = n * m * p
    
    # We will try a range of ranks. 
    # It's unlikely that the rank is close to the lower bound for non-square matrices.
    # Let's try from min_possible up to 60.
    # We use multiple seeds for each rank to increase the chance of finding a solution.
    
    ranks_to_try = list(range(min_possible, 61))
    
    # Sort ranks: try lower ranks first, but maybe skip some if time is tight?
    # Actually, let's just iterate.
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5: # 5s buffer
            break
            
        # Try a few random seeds for this rank
        seeds = [42, 123, 999, 777, 2023]
        for seed in seeds:
            if time.time() > deadline - 2:
                break
                
            result = try_optimize(rank, time_budget=5, seed=seed)
            if result is not None:
                U, V, W, r = result
                best_result = (U, V, W, r)
                best_rank = r
                break # Found a valid decomposition for this rank, try next lower rank? No, we iterate low to high, so this is the best for now.
                # Actually, we want the MINIMAL rank. So if we find one at rank R, we should check if we can find one at R-1?
                # But we are iterating low to high. So the first one we find is the minimal (among those we tried).
                # However, we might have skipped some ranks? No, we iterate sequentially.
                # So if we find a solution at rank 30, we return it.
                # But wait, the loop breaks. We should return immediately?
                # Yes, because we are going from low to high.
                break
        
        if best_result is not None:
            break
    
    # If no better found, use schoolbook
    if best_result is None:
        U, V, W, rank = schoolbook()
    else:
        U, V, W, rank = best_result
    
    # Final verification
    if not check_valid(U, V, W, tol=1e-8):
        # Fallback to schoolbook
        U, V, W, rank = schoolbook()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
