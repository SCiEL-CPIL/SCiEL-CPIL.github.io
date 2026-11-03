import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def build_rank55():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Part A: j=2 (trivial, 2*10=20 terms)
        for i in range(n):
            for k in range(p):
                U[i * m + 2, col] = 1
                V[2 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        
        # Part B: j=0,1 with Strassen on 5 blocks of 2 columns
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_00 = 0 * m + 0
            u_01 = 0 * m + 1
            u_10 = 1 * m + 0
            u_11 = 1 * m + 1
            
            v_00 = 0 * p + k0
            v_01 = 0 * p + k1
            v_10 = 1 * p + k0
            v_11 = 1 * p + k1
            
            # M1 = (a00 + a11)(b00 + b11)
            U[u_00, col] += 1; U[u_11, col] += 1
            V[v_00, col] += 1; V[v_11, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M2 = (a10 + a11)b00
            U[u_10, col] += 1; U[u_11, col] += 1
            V[v_00, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            
            # M3 = a00(b01 - b11)
            U[u_00, col] += 1
            V[v_01, col] += 1; V[v_11, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M4 = a11(b10 - b00)
            U[u_11, col] += 1
            V[v_10, col] += 1; V[v_00, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            
            # M5 = (a00 + a01)b11
            U[u_00, col] += 1; U[u_01, col] += 1
            V[v_11, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            
            # M6 = (a10 - a00)(b00 + b01)
            U[u_10, col] += 1; U[u_00, col] += -1
            V[v_00, col] += 1; V[v_01, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7 = (a01 - a11)(b10 + b11)
            U[u_01, col] += 1; U[u_11, col] += -1
            V[v_10, col] += 1; V[v_11, col] += 1
            W[w_00, col] += 1
            col += 1
        
        return U, V, W, rank
    
    def compute_error(U, V, W):
        """Compute max absolute error for all (i,j,k) triples."""
        err_max = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m + j, :]
                    v_row = V[j*p + k, :]
                    w_row = W[k*n + i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err_max = max(err_max, abs(val - 1.0))
        return err_max
    
    def solve_W_given_UV(U, V, target_rank):
        """
        Given U and V, solve for W such that the tensor decomposition is exact.
        For each (k,i): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
        
        This is a linear system in W[k*n+i, :] for each (k,i).
        """
        W = np.zeros((pn, target_rank))
        
        for k in range(p):
            for i in range(n):
                # Build system: for each j, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Let A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # Then sum_r A[j, r] * W[k*n+i, r] = 1 for j=0,1,2
                A = np.zeros((m, target_rank))
                for j in range(m):
                    A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                b = np.ones(m)
                
                # Solve least squares
                w_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                W[k*n + i, :] = w_sol
        
        return W
    
    def solve_U_given_VW(V, W, target_rank):
        """Given V and W, solve for U."""
        U = np.zeros((nm, target_rank))
        
        for i in range(n):
            for j in range(m):
                # For each k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                A = np.zeros((p, target_rank))
                for k in range(p):
                    A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                b = np.ones(p)
                u_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i*m + j, :] = u_sol
        
        return U
    
    def solve_V_given_UW(U, W, target_rank):
        """Given U and W, solve for V."""
        V = np.zeros((mp, target_rank))
        
        for j in range(m):
            for k in range(p):
                # For each i: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                A = np.zeros((n, target_rank))
                for i in range(n):
                    A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                b = np.ones(n)
                v_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j*p + k, :] = v_sol
        
        return V
    
    def iterative_solve(U_init, V_init, target_rank, max_iters=50, tol=1e-10):
        """
        Alternating solve: fix two matrices, solve for the third.
        """
        U_cur, V_cur = U_init.copy(), V_init.copy()
        W_cur = solve_W_given_UV(U_cur, V_cur, target_rank)
        
        best_err = float('inf')
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            U_cur = solve_U_given_VW(V_cur, W_cur, target_rank)
            # Update V
            V_cur = solve_V_given_UW(U_cur, W_cur, target_rank)
            # Update W
            W_cur = solve_W_given_UV(U_cur, V_cur, target_rank)
            
            err = compute_error(U_cur, V_cur, W_cur)
            if err < best_err:
                best_err = err
                if err < tol:
                    break
        
        return U_cur, V_cur, W_cur, best_err
    
    # Build initial rank-55 solution
    U_best, V_best, W_best, rank_best = build_rank55()
    
    # Try to reduce rank
    for target_rank in range(rank_best - 1, 45, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        max_restarts = 15 if target_rank >= 52 else 10
        
        for trial in range(max_restarts):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(trial * 7919 + target_rank * 13)
            
            # Strategy: take first target_rank columns of rank-55, add noise
            U_init = U_best[:, :target_rank].copy()
            V_init = V_best[:, :target_rank].copy()
            
            # Add small noise to escape local minima
            noise_scale = 0.02
            U_init += noise_scale * np.random.randn(nm, target_rank)
            V_init += noise_scale * np.random.randn(mp, target_rank)
            
            # Run iterative solver
            U_cur, V_cur, W_cur, best_err = iterative_solve(
                U_init, V_init, target_rank, max_iters=80
            )
            
            if best_err < 1e-9:
                U_best, V_best, W_best, rank_best = U_cur, V_cur, W_cur, target_rank
                found = True
                break
            elif best_err < 1e-6:
                # Try more iterations with tighter tolerance
                U_cur2, V_cur2, W_cur2, best_err2 = iterative_solve(
                    U_cur, V_cur, target_rank, max_iters=100, tol=1e-12
                )
                if best_err2 < 1e-9:
                    U_best, V_best, W_best, rank_best = U_cur2, V_cur2, W_cur2, target_rank
                    found = True
                    break
        
        if not found:
            break  # Can't reduce further
    
    # Final verification
    final_err = compute_error(U_best, V_best, W_best)
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
