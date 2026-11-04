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
    # T has shape (n, m, p)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
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
    
    def try_merge(U, V, W, rank, num_trials=20):
        """
        Try to reduce rank by merging pairs of terms.
        For each pair of columns (a,b), try to find a linear combination
        that preserves the tensor.
        """
        if rank <= 1:
            return None
        
        # Try random linear combinations of pairs
        np.random.seed(int(time.time() * 1000) % 10000)
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Pick two random columns to potentially merge
            a, b = np.random.choice(rank, size=2, replace=False)
            
            # Try several random coefficients
            found = False
            for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                for beta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    # Create new column as alpha*col_a + beta*col_b
                    U_new = U[:, :, a].copy() * alpha + U[:, :, b].copy() * beta
                    V_new = V[:, :, a].copy() * alpha + V[:, :, b].copy() * beta
                    W_new = W[:, :, a].copy() * alpha + W[:, :, b].copy() * beta
                    
                    # Check if this single new column can replace both old columns
                    # Remove columns a and b, add new column
                    new_rank = rank - 1
                    if new_rank < 1:
                        continue
                    
                    # Build new tensors
                    U_test = np.delete(U, [a, b], axis=2)
                    V_test = np.delete(V, [a, b], axis=2)
                    W_test = np.delete(W, [a, b], axis=2)
                    
                    # Insert new column at position a
                    U_test = np.insert(U_test, a, U_new, axis=2)
                    V_test = np.insert(V_test, a, V_new, axis=2)
                    W_test = np.insert(W_test, a, W_new, axis=2)
                    
                    # Check
                    T_rec = np.einsum('ijc,jkc,kic->ijk', U_test, V_test, W_test)
                    if np.allclose(T_rec.flatten(), T_flat, atol=1e-6):
                        return U_test, V_test, W_test, new_rank
            
            # Also try: replace one column with a combination
            for a in range(rank):
                for b in range(rank):
                    if a == b:
                        continue
                    for alpha in [0.5, 1.0, -0.5, -1.0]:
                        U_new = U[:, :, a].copy() * alpha + U[:, :, b].copy() * 1.0
                        V_new = V[:, :, a].copy() * alpha + V[:, :, b].copy() * 1.0
                        W_new = W[:, :, a].copy() * alpha + W[:, :, b].copy() * 1.0
                        
                        U_test = U.copy()
                        V_test = V.copy()
                        W_test = W.copy()
                        U_test[:, :, a] = U_new
                        V_test[:, :, a] = V_new
                        W_test[:, :, a] = W_new
                        
                        T_rec = np.einsum('ijc,jkc,kic->ijk', U_test, V_test, W_test)
                        if np.allclose(T_rec.flatten(), T_flat, atol=1e-6):
                            # We can remove column b
                            new_rank = rank - 1
                            if new_rank < 1:
                                continue
                            U_test = np.delete(U_test, b, axis=2)
                            V_test = np.delete(V_test, b, axis=2)
                            W_test = np.delete(W_test, b, axis=2)
                            return U_test, V_test, W_test, new_rank
    
        return None
    
    def als_decompose(rank, max_iters=50, tol=1e-12, seed=0, init=None):
        """Alternating Least Squares for 3rd order tensor decomposition"""
        np.random.seed(seed)
        
        if init is not None:
            U, V, W = init
        else:
            # Initialize with small random values
            U = np.random.randn(n, m, rank) * 0.3
            V = np.random.randn(m, p, rank) * 0.3
            W = np.random.randn(p, n, rank) * 0.3
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Compute current error
            T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = T_rec - T
            current_err = np.sum(err**2)
            
            if abs(prev_err - current_err) < tol:
                break
            prev_err = current_err
            
            # Solve for U: fix V, W
            for i in range(n):
                for j in range(m):
                    # A[c, k] = V[j,k,c] * W[k,i,c]
                    A = np.einsum('kc,kc->ck', V[j], W[:, i])  # shape (rank, p)
                    b = T[i, j, :]  # shape (p,)
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i, j, :] = u_sol
                    except:
                        pass
            
            # Solve for V: fix U, W
            for j in range(m):
                for k in range(p):
                    A = np.einsum('ic,ic->ci', U[:, j, :], W[k, :, :])  # shape (rank, n)
                    b = T[:, j, k]  # shape (n,)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        V[j, k, :] = v_sol
                    except:
                        pass
            
            # Solve for W: fix U, V
            for k in range(p):
                for i in range(n):
                    A = np.einsum('jc,jc->cj', U[i, :, :], V[:, k, :])  # shape (rank, m)
                    b = T[i, :, k]  # shape (m,)
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        W[k, i, :] = w_sol
                    except:
                        pass
        
        # Final check
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        final_err = np.sum((T_rec - T)**2)
        
        if final_err < 1e-4:
            # Try rounding to various grids
            for scale in [2, 4, 8]:
                U_rounded = np.round(U * scale) / scale
                V_rounded = np.round(V * scale) / scale
                W_rounded = np.round(W * scale) / scale
                if check_exact(U_rounded, V_rounded, W_rounded, rank):
                    return (U_rounded.reshape(NM, rank).tolist(), 
                           V_rounded.reshape(MP, rank).tolist(), 
                           W_rounded.reshape(PN, rank).tolist())
            
            # Try integer rounding
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if check_exact(U_int, V_int, W_int, rank):
                return (U_int.reshape(NM, rank).tolist(), 
                       V_int.reshape(MP, rank).tolist(), 
                       W_int.reshape(PN, rank).tolist())
        
        return None
    
    def try_rank_als(rank, num_restarts=15):
        """Try to find decomposition of given rank using ALS"""
        for seed in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            result = als_decompose(rank, max_iters=40, tol=1e-12, seed=seed)
            if result is not None:
                return result
        
        # Try with schoolbook-inspired initialization
        U_sb, V_sb, W_sb = schoolbook_init()
        for seed in range(5):
            if time.time() > deadline - 10:
                return None
            np.random.seed(seed + 1000)
            # Randomly select a subset of schoolbook columns
            selected = np.random.choice(rank_schoolbook, size=min(rank, rank_schoolbook), replace=False)
            U_init = U_sb[:, :, selected]
            V_init = V_sb[:, :, selected]
            W_init = W_sb[:, :, selected]
            
            result = als_decompose(rank, max_iters=40, tol=1e-12, seed=seed, init=(U_init, V_init, W_init))
            if result is not None:
                return result
        
        return None
    
    def try_rank_merge(start_rank, target_rank, max_merges=200):
        """
        Start from schoolbook and try to reduce rank via merging.
        """
        U, V, W = schoolbook_init()
        current_rank = start_rank
        
        # First, try to reduce to target_rank by removing redundant columns
        # Then try merging
        
        merges_done = 0
        while current_rank > target_rank and merges_done < max_merges:
            if time.time() > deadline - 10:
                break
            
            result = try_merge(U, V, W, current_rank, num_trials=10)
            if result is not None:
                U, V, W, current_rank = result
                merges_done += 1
            else:
                # Try ALS to clean up
                result = als_decompose(current_rank, max_iters=20, tol=1e-12, seed=merges_done, init=(U, V, W))
                if result is not None:
                    U = np.array(result[0]).reshape(n, m, current_rank)
                    V = np.array(result[1]).reshape(m, p, current_rank)
                    W = np.array(result[2]).reshape(p, n, current_rank)
                else:
                    break
        
        if current_rank <= target_rank:
            # Round to clean values
            for scale in [2, 4, 8]:
                U_rounded = np.round(U * scale) / scale
                V_rounded = np.round(V * scale) / scale
                W_rounded = np.round(W * scale) / scale
                if check_exact(U_rounded, V_rounded, W_rounded, current_rank):
                    return (U_rounded.reshape(NM, current_rank).tolist(), 
                           V_rounded.reshape(MP, current_rank).tolist(), 
                           W_rounded.reshape(PN, current_rank).tolist())
            
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if check_exact(U_int, V_int, W_int, current_rank):
                return (U_int.reshape(NM, current_rank).tolist(), 
                       V_int.reshape(MP, current_rank).tolist(), 
                       W_int.reshape(PN, current_rank).tolist())
        
        return None
    
    # Strategy:
    # 1. First, try to find a good decomposition via ALS from rank 60 down to 48
    # 2. If that fails, use the merge approach starting from schoolbook
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Phase 1: ALS search from reasonable ranks
    for rank in range(62, 47, -1):
        if time.time() > deadline - 20:
            break
        
        result = try_rank_als(rank, num_restarts=8)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break
    
    # Phase 2: If Phase 1 failed, try merge approach
    if best_result is None:
        # Start from schoolbook and try to reduce
        for target in [55, 50, 48]:
            if time.time() > deadline - 15:
                break
            
            result = try_rank_merge(rank_schoolbook, target, max_merges=100)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = len(U[0]) if U else 0
                break
    
    # Phase 3: If still no result, use schoolbook
    if best_result is None:
        U = [[0] * rank_schoolbook for _ in range(NM)]
        V = [[0] * rank_schoolbook for _ in range(MP)]
        W = [[0] * rank_schoolbook for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank_schoolbook
    
    return best_result
# EVOLVE-BLOCK-END
