"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # Leave 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    # Build target tensor T[i,j,k] = 1 if (i,j,k) is valid else 0
    T = np.ones((n, m, p))
    T_flat = T.flatten()
    NM, MP, PN = n*m, m*p, p*n
    
    def check_exact(U, V, W, rank):
        """Check if U, V, W with given rank exactly reconstruct T"""
        U_r = np.array(U).reshape(n, m, rank)
        V_r = np.array(V).reshape(m, p, rank)
        W_r = np.array(W).reshape(p, n, rank)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec.flatten(), T_flat, atol=1e-8)
    
    def schoolbook_init():
        """Create the schoolbook (trivial) decomposition"""
        U = np.zeros((n, m, rank_schoolbook))
        V = np.zeros((m, p, rank_schoolbook))
        W = np.zeros((p, n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, j, col] = 1
                    V[j, k, col] = 1
                    W[k, i, col] = 1
                    col += 1
        return U, V, W
    
    def snap_to_halfint(X):
        """Snap values to nearest half-integer"""
        return np.round(X * 2.0) / 2.0
    
    def alternating_projection(rank, max_iters=200, seed=0, init=None):
        """
        Alternating projection: 
        1. Solve linear least-squares for each factor (affine projection)
        2. Snap to nearest half-integers
        Iterate until convergence or exact solution found.
        """
        np.random.seed(seed)
        
        if init is not None:
            U, V, W = [x.copy() for x in init]
        else:
            # Initialize with small random values snapped to half-integers
            U = snap_to_halfint(np.random.randn(n, m, rank) * 0.5)
            V = snap_to_halfint(np.random.randn(m, p, rank) * 0.5)
            W = snap_to_halfint(np.random.randn(p, n, rank) * 0.5)
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Check current error
            T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = T_rec - T
            current_err = np.sum(err**2)
            
            if current_err < 1e-10:
                # Check if already half-integer
                if np.allclose(U, snap_to_halfint(U), atol=1e-10) and \
                   np.allclose(V, snap_to_halfint(V), atol=1e-10) and \
                   np.allclose(W, snap_to_halfint(W), atol=1e-10):
                    return U, V, W
                # Snap and recheck
                U = snap_to_halfint(U)
                V = snap_to_halfint(V)
                W = snap_to_halfint(W)
                T_rec2 = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                if np.allclose(T_rec2.flatten(), T_flat, atol=1e-8):
                    return U, V, W
                return None
            
            if abs(prev_err - current_err) < 1e-14 and it > 10:
                # Converged but not exact
                U = snap_to_halfint(U)
                V = snap_to_halfint(V)
                W = snap_to_halfint(W)
                T_rec2 = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                if np.allclose(T_rec2.flatten(), T_flat, atol=1e-8):
                    return U, V, W
                return None
            prev_err = current_err
            
            # Step 1: Solve for U (affine projection)
            for i in range(n):
                for j in range(m):
                    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                    # For fixed i,j: b[k] = T[i,j,k], A[c,k] = V[j,k,c]*W[k,i,c]
                    A = np.einsum('kc,kc->ck', V[j], W[:, i])  # (rank, p)
                    b = T[i, j, :]  # (p,)
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i, j, :] = u_sol
                    except:
                        pass
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    A = np.einsum('ic,ic->ci', U[:, j, :], W[k, :, :])  # (rank, n)
                    b = T[:, j, k]  # (n,)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        V[j, k, :] = v_sol
                    except:
                        pass
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    A = np.einsum('jc,jc->cj', U[i, :, :], V[:, k, :])  # (rank, m)
                    b = T[i, :, k]  # (m,)
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        W[k, i, :] = w_sol
                    except:
                        pass
            
            # Step 2: Snap to half-integers
            U = snap_to_halfint(U)
            V = snap_to_halfint(V)
            W = snap_to_halfint(W)
        
        # Final check
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        if np.allclose(T_rec.flatten(), T_flat, atol=1e-8):
            return U, V, W
        return None
    
    def try_rank_ap(rank, num_restarts=20, init_from_schoolbook=False):
        """Try alternating projection at given rank"""
        for seed in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            
            if init_from_schoolbook and rank <= rank_schoolbook:
                U_sb, V_sb, W_sb = schoolbook_init()
                # Randomly perturb schoolbook columns
                np.random.seed(seed)
                selected = np.random.choice(rank_schoolbook, size=rank, replace=False)
                U_init = U_sb[:, :, selected].copy()
                V_init = V_sb[:, :, selected].copy()
                W_init = W_sb[:, :, selected].copy()
                # Small random perturbation
                U_init += np.random.randn(n, m, rank) * 0.1
                V_init += np.random.randn(m, p, rank) * 0.1
                W_init += np.random.randn(p, n, rank) * 0.1
                result = alternating_projection(rank, max_iters=100, seed=seed, init=(U_init, V_init, W_init))
            else:
                result = alternating_projection(rank, max_iters=100, seed=seed)
            
            if result is not None:
                U, V, W = result
                return (U.reshape(NM, rank).tolist(), 
                       V.reshape(MP, rank).tolist(), 
                       W.reshape(PN, rank).tolist())
        
        return None
    
    # Strategy: Try ranks from 72 down to 60
    # Start with schoolbook (rank 72) as guaranteed solution
    U_sb, V_sb, W_sb = schoolbook_init()
    best_result = {"U": U_sb.reshape(NM, rank_schoolbook).tolist(),
                   "V": V_sb.reshape(MP, rank_schoolbook).tolist(),
                   "W": W_sb.reshape(PN, rank_schoolbook).tolist()}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    for rank in range(rank_schoolbook - 1, 59, -1):
        if time.time() > deadline - 20:
            break
        
        # Try with schoolbook-inspired initialization
        result = try_rank_ap(rank, num_restarts=10, init_from_schoolbook=True)
        if result is not None:
            best_result = {"U": result[0], "V": result[1], "W": result[2]}
            best_rank = rank
            break
        
        # Also try random initialization
        result = try_rank_ap(rank, num_restarts=5, init_from_schoolbook=False)
        if result is not None:
            best_result = {"U": result[0], "V": result[1], "W": result[2]}
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
