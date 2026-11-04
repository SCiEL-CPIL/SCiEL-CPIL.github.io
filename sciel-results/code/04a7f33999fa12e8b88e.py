# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = (2, 5, 6)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s for safety

    rank_school = n * m * p

    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank_school

    def verify_exact_fast(U, V, W, rank):
        """
        Fast verification using floating point first, then exact if close.
        U, V, W are lists of lists of floats (snapped to half-integers).
        """
        # First pass: float check
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Residual calculation: sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1
        # We can do this with einsum if we reshape, but loops are small.
        # Let's just do a quick float check.
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        if max_err > 1e-6:
            return False
            
        # Second pass: Exact check with Fractions
        # Only convert to Fraction if float check passes
        U_frac = [[Fraction(round(x * 2) / 2) for x in row] for row in U]
        V_frac = [[Fraction(round(x * 2) / 2) for x in row] for row in V]
        W_frac = [[Fraction(round(x * 2) / 2) for x in row] for row in W]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def compute_residual_fast(U, V, W):
        """
        U, V, W are numpy arrays.
        Returns max absolute error.
        """
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    e = abs(val - 1.0)
                    if e > err:
                        err = e
        return err

    def als_decompose(rank, seed, max_iters=200):
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize to prevent overflow/underflow
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None, False
            
            # Update U
            # For each (i,j), solve: sum_k V[j,k,c] W[k,i,c] U[i,j,c] = 1
            # This is a linear system A u = b, where A is (p, rank) and b is ones(p)
            new_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        # Use lstsq
                        u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        new_U[i*m+j, :] = u
                    except:
                        return None, False
            U = new_U
            
            # Update V
            # For each (j,k), solve: sum_i U[i,j,c] W[k,i,c] V[j,k,c] = 1
            new_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        new_V[j*p+k, :] = v
                    except:
                        return None, False
            V = new_V
            
            # Update W
            # For each (k,i), solve: sum_j U[i,j,c] V[j,k,c] W[k,i,c] = 1
            new_W = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        new_W[k*n+i, :] = w
                    except:
                        return None, False
            W = new_W
            
            err = compute_residual_fast(U, V, W)
            if err < 1e-9:
                break
            if abs(prev_err - err) < 1e-12:
                break
            prev_err = err
                
        err = compute_residual_fast(U, V, W)
        if err > 1e-4:
            return None, False
            
        # Snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if verify_exact_fast(U_list, V_list, W_list, rank):
            return (U_list, V_list, W_list), True
        else:
            return None, False

    def polish_local_search(U, V, W, rank, max_steps=1000):
        """
        Local search on the half-integer lattice to fix small errors.
        U, V, W are lists of lists of floats (snapped).
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        if verify_exact_fast(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), rank):
            return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), True
        
        best_err = compute_residual_fast(U_arr, V_arr, W_arr)
        
        # Pre-compute indices for speed
        # We will try to flip signs or add/subtract 0.5 to entries
        # To be efficient, we only try to modify entries that have large contribution to error
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            if best_err < 1e-9:
                break
            
            # Pick a random matrix, row, col
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                mat = U_arr
                rows = n * m
                # Compute gradient-like contribution? Too complex. Just random.
            elif mat_idx == 1:
                mat = V_arr
                rows = m * p
            else:
                mat = W_arr
                rows = p * n
                
            row = np.random.randint(rows)
            col = np.random.randint(rank)
            
            old_val = mat[row, col]
            
            # Try neighbors: old_val, old_val +/- 0.5
            # Also try 0 if old_val is large?
            candidates = [old_val - 0.5, old_val, old_val + 0.5]
            
            best_cand = old_val
            best_cand_err = best_err
            
            for cand in candidates:
                mat[row, col] = cand
                err = compute_residual_fast(U_arr, V_arr, W_arr)
                if err < best_cand_err:
                    best_cand_err = err
                    best_cand = cand
            mat[row, col] = best_cand
            
            if best_cand_err < best_err - 1e-10:
                best_err = best_cand_err
            else:
                # No improvement, maybe restart or break?
                # For now, just continue to try more
                pass
            
            if best_err < 1e-9:
                break
                
        U_list = U_arr.tolist()
        V_list = V_arr.tolist()
        W_list = W_arr.tolist()
        
        if verify_exact_fast(U_list, V_list, W_list, rank):
            return U_list, V_list, W_list, True
        
        return None, None, None, False

    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    # Try ranks from 55 down to 50
    # 60 is schoolbook. 55 is a good target.
    candidate_ranks = [55, 54, 53, 52, 51, 50]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        found = False
        
        # Try multiple seeds
        for seed in range(50):
            if time.time() > deadline - 10:
                break
            
            result, success = als_decompose(target_rank, seed)
            if success:
                best_U, best_V, best_W, best_rank = result[0], result[1], result[2], target_rank
                found = True
                break
        
        if found:
            continue
        
        # If ALS didn't find exact, try polish on near-converged solutions
        if time.time() > deadline - 10:
            break
        
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(seed)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            U /= np.sqrt(target_rank)
            V /= np.sqrt(target_rank)
            W /= np.sqrt(target_rank)
            
            # Run ALS for fewer iterations
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                # Update U
                new_U = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            new_U[i*m+j, :] = u
                        except:
                            break
                U = new_U
                # Update V
                new_V = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            new_V[j*p+k, :] = v
                        except:
                            break
                V = new_V
                # Update W
                new_W = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            new_W[k*n+i, :] = w
                        except:
                            break
                W = new_W
            
            err = compute_residual_fast(U, V, W)
            if err < 1.0:  # Close enough to try polish
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                polished_U, polished_V, polished_W, success = polish_local_search(
                    U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), target_rank
                )
                if success:
                    best_U, best_V, best_W, best_rank = polished_U, polished_V, polished_W, target_rank
                    found = True
                    break
                    
    if best_rank == rank_school:
        U, V, W, _ = make_schoolbook()
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
