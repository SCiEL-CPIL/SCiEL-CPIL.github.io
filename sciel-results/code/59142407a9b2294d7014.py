"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    nm = n * m
    mp = m * p
    pn = p * n

    def check_solution(U, V, W, n, m, p, tol=1e-6):
        """Check if U, V, W satisfy the tensor equations for matrix multiplication."""
        r = U.shape[1]
        # Reshape W to (p, n, r) for easier access W[k, i, r]
        # W input shape is (p*n, r) with index k*n+i
        W_reshaped = W.T.reshape(r, p, n).T  # (p, n, r)
        
        # Precompute W_flat for dot products?
        # We need C[i', k'] = sum_r W[k', i', r] * U[i, j, r] * V[j, k, r]
        
        for i in range(n):
            for j in range(m):
                u = U[i*m + j]  # (r,)
                for k in range(p):
                    v = V[j*p + k]  # (r,)
                    prod = u * v    # (r,)
                    
                    # Compute C_calc for this (i, j, k)
                    # C_calc[i', k'] = sum_r prod[r] * W_reshaped[k', i', r]
                    # This is a linear combination of columns of W_reshaped
                    # Let's use matrix multiplication
                    # W_reshaped is (p, n, r). We want sum_r prod[r] * W_reshaped[:, :, r]
                    # This is np.tensordot(prod, W_reshaped, axes=([0], [2])) -> (p, n)
                    C_calc = np.tensordot(prod, W_reshaped, axes=([0], [2]))
                    
                    # Expected C is E[i, k] (1 at i, k, 0 elsewhere)
                    # Note: C_calc is (p, n) i.e., indexed by (k', i')
                    # We need C_calc[k, i] == 1 and C_calc[k', i'] == 0 for others
                    
                    # Check diagonal
                    if abs(C_calc[k, i] - 1.0) > tol:
                        return False
                    
                    # Check off-diagonals
                    # Create a mask or just check sum of squares
                    C_sq = C_calc.copy()
                    C_sq[k, i] = 0
                    if np.linalg.norm(C_sq) > tol:
                        return False
        return True

    def try_find_solution(n, m, p, rank, time_limit, num_restarts=10):
        """Try to find a solution of given rank using random init + ALS."""
        nm = n * m
        mp = m * p
        pn = p * n
        best_solution = None
        start_time = time.time()
        
        for restart in range(num_restarts):
            if time.time() > start_time + time_limit:
                break
                
            # Random initialization
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # ALS Iteration
            converged = False
            for it in range(50):
                if time.time() > start_time + time_limit - 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Build M and b for U[i*m+j, :]
                        # M has size (n*p) x rank
                        # b has size (n*p)
                        M = np.zeros((n*p, rank))
                        b = np.zeros(n*p)
                        
                        for k in range(p):
                            v = V[j*p + k]
                            for ip in range(n*p):
                                # ip corresponds to k'*n + i'
                                k_prime = ip // n
                                i_prime = ip % n
                                w_val = W[ip] # W[k'*n+i', r] is W[ip, r]
                                M[ip, :] = w_val * v
                                if i_prime == i and k_prime == k:
                                    b[ip] = 1.0
                        try:
                            x, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                            U[i*m + j, :] = x
                        except:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n*m, rank)) # Wait, V depends on j, k.
                        # Equation: sum_r W[k', i', r] U[i, j', r] V[j, k, r] = delta
                        # For fixed j, k, we solve for V[j, k, r].
                        # Variables: V[j, k, r]
                        # Equations for all i, j', i', k'?
                        # No, the input is E[i, j'] and E[j, k].
                        # So i, j' vary.
                        # Output C[i', k']
                        # Term: U[i, j', r] V[j, k, r] W[k', i', r]
                        # M size: (n * m * n * p) x rank? No.
                        # The constraint is for ALL i, j', i', k'.
                        # But V[j, k, r] only appears when the second index of A is j' (which is the row of B)
                        # Wait, A is n x m. B is m x p.
                        # A_{i, j'}, B_{j', k}.
                        # So j' is the column of A and row of B.
                        # In our V index, it is V[j, k]. So j is the row of B.
                        # So for fixed V[j, k], j is fixed.
                        # The varying indices are i (row of A) and k (col of B is fixed in V index? No, k is fixed in V index).
                        # So for V[j, k], the varying input indices are i (from A) and... wait.
                        # The term is U[i, j, r] V[j, k, r] W[k, i, r].
                        # Here j is fixed (from V index).
                        # i varies (from A).
                        # k is fixed (from V index).
                        # So the equations are for all i, i', k'.
                        # M size: (n * n * p) x rank?
                        # Indices: i (input row), i' (output row), k' (output col).
                        # M[i, i', k', r] = U[i, j, r] * W[k', i', r]
                        # b[i, i', k'] = delta[i', i] * delta[k', k]
                        
                        M = np.zeros((n * n * p, rank))
                        b = np.zeros(n * n * p)
                        
                        idx = 0
                        for i in range(n):
                            u = U[i*m + j]
                            for i_prime in range(n):
                                for k_prime in range(p):
                                    w_val = W[k_prime*n + i_prime]
                                    M[idx, :] = u * w_val
                                    if i_prime == i and k_prime == k:
                                        b[idx] = 1.0
                                    idx += 1
                        try:
                            x, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                            V[j*p + k, :] = x
                        except:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        # Solve for W[k*n+i, r]
                        # Term: U[i, j, r] V[j, k, r] W[k, i, r]
                        # Fixed k, i.
                        # Varying: j (col of A / row of B).
                        # Output: i', k'.
                        # M size: (m * n * p) x rank
                        # Indices: j, i', k'
                        # M[j, i', k', r] = U[i, j, r] * V[j, k, r]
                        # b[j, i', k'] = delta[i', i] * delta[k', k]
                        
                        M = np.zeros((m * n * p, rank))
                        b = np.zeros(m * n * p)
                        
                        idx = 0
                        for j in range(m):
                            u = U[i*m + j]
                            v = V[j*p + k]
                            for i_prime in range(n):
                                for k_prime in range(p):
                                    M[idx, :] = u * v
                                    if i_prime == i and k_prime == k:
                                        b[idx] = 1.0
                                    idx += 1
                        try:
                            x, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                            W[k*n + i, :] = x
                        except:
                            pass

                # Check convergence
                if check_solution(U, V, W, n, m, p, tol=1e-4):
                    converged = True
                    break
            
            if converged:
                return U, V, W

        return None

    def construct_block_solution(n, m, p, time_limit):
        # Try to find M[n,m,3]
        R1 = None
        U1 = None
        V1 = None
        W1 = None
        
        # Search for 3,3,3
        min_rank_333 = max(n*m, m*3, 3*n)
        for r in range(min_rank_333, 28):
            if time.time() > time_limit:
                break
            res = try_find_solution(n, m, 3, r, time_limit/2, num_restarts=5)
            if res:
                U1, V1, W1 = res
                R1 = r
                break
        
        if R1 is None:
            # Use schoolbook for 3,3,3
            R1 = n * m * 3
            U1 = np.zeros((n*m, R1))
            V1 = np.zeros((m*3, R1))
            W1 = np.zeros((3*n, R1))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(3):
                        U1[i*m+j, col] = 1
                        V1[j*3+k, col] = 1
                        W1[k*n+i, col] = 1
                        col += 1
        
        # Construct 3,3,7 solution
        # Part 1: 3,3,3 (cols 0-2 of B)
        # Part 2: 3,3,4 (cols 3-6 of B) - Schoolbook
        R2 = n * m * 4
        R_total = R1 + R2
        
        U = np.zeros((n*m, R_total))
        V = np.zeros((m*p, R_total))
        W = np.zeros((p*n, R_total))
        
        # Copy Part 1
        U[:, :R1] = U1
        V[:m*3, :R1] = V1
        W[:3*n, :R1] = W1
        
        # Part 2: Schoolbook for k=3..6
        col = R1
        for i in range(n):
            for j in range(m):
                for k in range(3, p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        return U, V, W, R_total

    # Try direct search for 3,3,7
    min_rank = max(nm, mp, pn)
    for rank in range(min_rank, min_rank + 10):
        if time.time() > deadline - 30:
            break
        res = try_find_solution(n, m, p, rank, 10, num_restarts=5)
        if res:
            U, V, W = res
            # Round and check
            U_out = np.round(U * 2) / 2
            V_out = np.round(V * 2) / 2
            W_out = np.round(W * 2) / 2
            if check_solution(U_out, V_out, W_out, n, m, p):
                return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
            # If rounded fails, check float
            if check_solution(U, V, W, n, m, p, tol=1e-5):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to block construction
    U, V, W, R = construct_block_solution(n, m, p, deadline - 10)
    
    # Verify block solution
    if check_solution(U, V, W, n, m, p):
        # Round to 0.5
        U_out = np.round(U * 2) / 2
        V_out = np.round(V * 2) / 2
        W_out = np.round(W * 2) / 2
        if check_solution(U_out, V_out, W_out, n, m, p):
            return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Final Fallback: Schoolbook 3,3,7
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
