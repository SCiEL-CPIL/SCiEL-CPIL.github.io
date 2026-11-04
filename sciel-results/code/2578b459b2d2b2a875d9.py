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
    deadline = start_time + 140
    
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
    
    def verify_exact(U, V, W, rank):
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

    def compute_residual(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err += (val - 1.0)**2
        return err

    def als_decompose(rank, seed, max_iters=150):
        np.random.seed(seed)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None, False
            
            # Update U
            new_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        new_U[i*m+j, :] = u
                    except:
                        return None, False
            U = new_U
            
            # Update V
            new_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    N = np.zeros((n, rank))
                    for i in range(n):
                        N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(N, np.ones(n), rcond=None)
                        new_V[j*p+k, :] = v
                    except:
                        return None, False
            V = new_V
            
            # Update W
            new_W = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    P = np.zeros((m, rank))
                    for j in range(m):
                        P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(P, np.ones(m), rcond=None)
                        new_W[k*n+i, :] = w
                    except:
                        return None, False
            W = new_W
            
            err = compute_residual(U, V, W)
            if err < 1e-10:
                break
            if abs(prev_err - err) < 1e-15:
                break
            prev_err = err
                
        err = compute_residual(U, V, W)
        if err > 1e-6:
            return None, False
            
        # Snap to half-integers and verify
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if verify_exact(U_list, V_list, W_list, rank):
            return (U_list, V_list, W_list), True
        else:
            return None, False

    def polish_half_integer(U, V, W, rank, max_steps=500):
        """Try to adjust entries to nearest half-integers while maintaining exactness."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # First snap
        U_arr = np.round(U_arr * 2) / 2
        V_arr = np.round(V_arr * 2) / 2
        W_arr = np.round(W_arr * 2) / 2
        
        # Check if already exact
        if verify_exact(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), rank):
            return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), True
        
        # Try local search: perturb individual entries by +/- 0.5
        best_U = U_arr.copy()
        best_V = V_arr.copy()
        best_W = W_arr.copy()
        best_err = compute_residual(best_U, best_V, best_W)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try perturbing a random entry
            matrices = [('U', best_U, n*m), ('V', best_V, m*p), ('W', best_W, p*n)]
            mat_name, mat, rows = matrices[np.random.randint(3)]
            row = np.random.randint(rows)
            col = np.random.randint(rank)
            
            old_val = mat[row, col]
            # Try nearest half-integers around old_val
            candidates = [old_val - 0.5, old_val, old_val + 0.5]
            # Also try the two nearest half-integers
            nearest = np.round(old_val * 2) / 2
            candidates = list(set([nearest - 0.5, nearest, nearest + 0.5]))
            
            for cand in candidates:
                if mat_name == 'U':
                    best_U[row, col] = cand
                    err = compute_residual(best_U, best_V, best_W)
                    best_U[row, col] = old_val
                elif mat_name == 'V':
                    best_V[row, col] = cand
                    err = compute_residual(best_U, best_V, best_W)
                    best_V[row, col] = old_val
                else:
                    best_W[row, col] = cand
                    err = compute_residual(best_U, best_V, best_W)
                    best_W[row, col] = old_val
                
                if err < best_err - 1e-10:
                    best_err = err
                    improved = True
                    if mat_name == 'U':
                        best_U[row, col] = cand
                    elif mat_name == 'V':
                        best_V[row, col] = cand
                    else:
                        best_W[row, col] = cand
                    break
            
            if not improved:
                # Try a larger perturbation: reset one matrix row
                break
        
        if best_err < 1e-10:
            U_list = best_U.tolist()
            V_list = best_V.tolist()
            W_list = best_W.tolist()
            if verify_exact(U_list, V_list, W_list, rank):
                return U_list, V_list, W_list, True
        
        return None, None, None, False

    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    # Try structured initializations and random ones
    candidate_ranks = [55, 54, 53, 52, 51, 50, 48, 45, 42, 40]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Try multiple seeds
        for seed in range(30):
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
        
        for seed in range(15):
            if time.time() > deadline - 8:
                break
            
            np.random.seed(seed)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            U /= np.sqrt(target_rank)
            V /= np.sqrt(target_rank)
            W /= np.sqrt(target_rank)
            
            # Run ALS for fewer iterations to get close
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                new_U = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            new_U[i*m+j, :] = u
                        except:
                            break
                U = new_U
                
                new_V = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, target_rank))
                        for i in range(n):
                            N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            v, _, _, _ = np.linalg.lstsq(N, np.ones(n), rcond=None)
                            new_V[j*p+k, :] = v
                        except:
                            break
                V = new_V
                
                new_W = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        P = np.zeros((m, target_rank))
                        for j in range(m):
                            P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            w, _, _, _ = np.linalg.lstsq(P, np.ones(m), rcond=None)
                            new_W[k*n+i, :] = w
                        except:
                            break
                W = new_W
            
            err = compute_residual(U, V, W)
            if err < 0.5:  # Close enough to try polish
                U_list = np.round(U * 2) / 2
                V_list = np.round(V * 2) / 2
                W_list = np.round(W * 2) / 2
                
                polished_U, polished_V, polished_W, success = polish_half_integer(
                    U_list.tolist(), V_list.tolist(), W_list.tolist(), target_rank
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
