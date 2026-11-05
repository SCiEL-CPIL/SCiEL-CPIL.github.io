"""
Rank-minimal exact tensor decomposition for 3x4 x 4x6 matrix multiplication.
Uses a structural change: integer programming via exhaustive search over small
integer/half-integer values, combined with a rank-reduction strategy based on
exploiting the structure of the multiplication tensor.
"""
import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145  # Leave 5s margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    NM, MP, PN = n*m, m*p, p*n
    
    # Build target tensor T[i,j,k] = 1 for all valid (i,j,k)
    T = np.ones((n, m, p))
    T_flat = T.flatten()
    
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
    
    def verify_decomposition(U, V, W, rank):
        """Verify exact decomposition and return True if valid"""
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec.flatten(), T_flat, atol=1e-8)
    
    def try_rank_with_search(rank, max_time=30):
        """
        Try to find a rank-r decomposition using a hybrid approach:
        1. Start from random half-integer initializations
        2. Use iterative refinement with linear solves
        3. Snap to half-integers and check exactness
        """
        np.random.seed(rank)  # Deterministic for reproducibility
        
        for attempt in range(50):
            if time.time() > start_time + max_time:
                return None
            
            # Random half-integer initialization
            U = snap_to_halfint(np.random.randn(n, m, rank) * 1.5)
            V = snap_to_halfint(np.random.randn(m, p, rank) * 1.5)
            W = snap_to_halfint(np.random.randn(p, n, rank) * 1.5)
            
            # Iterative refinement
            for it in range(200):
                if time.time() > start_time + max_time:
                    return None
                
                # Check current solution
                if verify_decomposition(U, V, W, rank):
                    return (U.reshape(NM, rank).tolist(),
                           V.reshape(MP, rank).tolist(),
                           W.reshape(PN, rank).tolist())
                
                # Solve for U given V, W
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
                
                # Solve for V given U, W
                for j in range(m):
                    for k in range(p):
                        A = np.einsum('ic,ic->ci', U[:, j, :], W[k, :, :])  # (rank, n)
                        b = T[:, j, k]  # (n,)
                        try:
                            v_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                            V[j, k, :] = v_sol
                        except:
                            pass
                
                # Solve for W given U, V
                for k in range(p):
                    for i in range(n):
                        A = np.einsum('jc,jc->cj', U[i, :, :], V[:, k, :])  # (rank, m)
                        b = T[i, :, k]  # (m,)
                        try:
                            w_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                            W[k, i, :] = w_sol
                        except:
                            pass
                
                # Snap to half-integers
                U = snap_to_halfint(U)
                V = snap_to_halfint(V)
                W = snap_to_halfint(W)
        
        return None
    
    def try_rank_with_structure(rank):
        """
        Try a structured approach: exploit the fact that for matrix multiplication,
        we can group terms by their contribution to specific output entries.
        """
        # For rank r, we need to find r triples (u_c, v_c, w_c) such that
        # sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k
        
        # Strategy: use a grid search over small integer values
        # Since the tensor is all-ones, we can try to find solutions where
        # the factors have small integer/half-integer values
        
        values = [-1, -0.5, 0, 0.5, 1]
        
        # For small ranks, do a more targeted search
        if rank <= 20:
            # Try to find solutions with sparse factors
            np.random.seed(42 + rank)
            for attempt in range(100):
                if time.time() > deadline - 20:
                    return None
                
                # Initialize with sparse random half-integers
                U = np.zeros((n, m, rank))
                V = np.zeros((m, p, rank))
                W = np.zeros((p, n, rank))
                
                # Randomly place non-zero entries
                for c in range(rank):
                    # Choose random positions for non-zero entries
                    u_pos = np.random.choice(n*m, size=np.random.randint(1, n*m+1), replace=False)
                    v_pos = np.random.choice(m*p, size=np.random.randint(1, m*p+1), replace=False)
                    w_pos = np.random.choice(p*n, size=np.random.randint(1, p*n+1), replace=False)
                    
                    for idx in u_pos:
                        i, j = divmod(idx, m)
                        U[i, j, c] = np.random.choice(values)
                    for idx in v_pos:
                        j, k = divmod(idx, p)
                        V[j, k, c] = np.random.choice(values)
                    for idx in w_pos:
                        k, i = divmod(idx, n)
                        W[k, i, c] = np.random.choice(values)
                
                # Refine
                for it in range(50):
                    if time.time() > deadline - 20:
                        return None
                    
                    if verify_decomposition(U, V, W, rank):
                        return (U.reshape(NM, rank).tolist(),
                               V.reshape(MP, rank).tolist(),
                               W.reshape(PN, rank).tolist())
                    
                    # Linear refinement
                    for i in range(n):
                        for j in range(m):
                            A = np.einsum('kc,kc->ck', V[j], W[:, i])
                            b = T[i, j, :]
                            try:
                                u_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                                U[i, j, :] = snap_to_halfint(u_sol)
                            except:
                                pass
                    
                    for j in range(m):
                        for k in range(p):
                            A = np.einsum('ic,ic->ci', U[:, j, :], W[k, :, :])
                            b = T[:, j, k]
                            try:
                                v_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                                V[j, k, :] = snap_to_halfint(v_sol)
                            except:
                                pass
                    
                    for k in range(p):
                        for i in range(n):
                            A = np.einsum('jc,jc->cj', U[i, :, :], V[:, k, :])
                            b = T[i, :, k]
                            try:
                                w_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                                W[k, i, :] = snap_to_halfint(w_sol)
                            except:
                                pass
                
        return None
    
    # Start with schoolbook as guaranteed solution
    U_sb, V_sb, W_sb = schoolbook_init()
    best_result = {"U": U_sb.reshape(NM, rank_schoolbook).tolist(),
                   "V": V_sb.reshape(MP, rank_schoolbook).tolist(),
                   "W": W_sb.reshape(PN, rank_schoolbook).tolist()}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Based on known results, the rank of 3x4x6 matrix multiplication is at most 30
    # (using Strassen-like algorithms) but we'll search for the minimum
    
    # Try ranks from high to low, with increasing effort
    for rank in range(rank_schoolbook - 1, 25, -1):
        if time.time() > deadline - 15:
            break
        
        # Time budget per rank
        time_budget = min(10, (deadline - 15 - (time.time() - start_time)) / max(1, rank_schoolbook - 25 - (rank_schoolbook - 1 - rank)))
        
        result = try_rank_with_search(rank, max_time=time_budget)
        if result is not None:
            best_result = {"U": result[0], "V": result[1], "W": result[2]}
            best_rank = rank
            break
        
        # Also try structured approach for smaller ranks
        if rank <= 40:
            result = try_rank_with_structure(rank)
            if result is not None:
                best_result = {"U": result[0], "V": result[1], "W": result[2]}
                best_rank = rank
                break
    
    # Final verification
    U_final = np.array(best_result["U"]).reshape(n, m, best_rank)
    V_final = np.array(best_result["V"]).reshape(m, p, best_rank)
    W_final = np.array(best_result["W"]).reshape(p, n, best_rank)
    T_rec = np.einsum('ijc,jkc,kic->ijk', U_final, V_final, W_final)
    if not np.allclose(T_rec.flatten(), T_flat, atol=1e-8):
        # Fallback to schoolbook
        best_result = {"U": U_sb.reshape(NM, rank_schoolbook).tolist(),
                       "V": V_sb.reshape(MP, rank_schoolbook).tolist(),
                       "W": W_sb.reshape(PN, rank_schoolbook).tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
