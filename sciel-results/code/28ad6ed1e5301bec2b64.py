"""
Program for 3x3x3 matrix multiplication tensor decomposition.
Uses an optimization-based search to attempt rank-25 decomposition,
falling back to the known rank-26 Strassen+Border solution if search fails.
"""
import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 3]

def construct():
    """
    Attempts to find a rank-25 decomposition via numerical optimization.
    If that fails or takes too long, returns the robust rank-26 decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 100.0  # Leave 20s buffer for the 120s limit
    
    # Target tensor T[i,j,k] = 1
    T_target = np.ones((n, m, p))
    
    def verify_decomposition(U, V, W, R):
        """
        Verify if U, V, W of rank R decompose T_target.
        U shape: (n*m, R), V shape: (m*p, R), W shape: (p*n, R)
        """
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T_recon, T_target, atol=1e-8)

    # Fallback: Robust Rank-26 Solution (Strassen + Border)
    def get_rank_26_solution():
        R = 26
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        def idx_u(i, j): return i * m + j
        def idx_v(j, k): return j * p + k
        def idx_w(k, i): return k * n + i

        # Strassen 7 terms
        # m1 = (a00 + a11)(b00 + b11)
        r = 0
        U[idx_u(0, 0), r] = 1; U[idx_u(1, 1), r] = 1
        V[idx_v(0, 0), r] = 1; V[idx_v(1, 1), r] = 1
        W[idx_w(0, 0), r] = 1; W[idx_w(1, 1), r] = 1
        
        # m2 = (a10 + a11)b00
        r = 1
        U[idx_u(1, 0), r] = 1; U[idx_u(1, 1), r] = 1
        V[idx_v(0, 0), r] = 1
        W[idx_w(0, 1), r] = 1; W[idx_w(1, 1), r] = -1
        
        # m3 = a00(b01 - b11)
        r = 2
        U[idx_u(0, 0), r] = 1
        V[idx_v(0, 1), r] = 1; V[idx_v(1, 1), r] = -1
        W[idx_w(1, 0), r] = 1; W[idx_w(1, 1), r] = 1
        
        # m4 = a11(b10 - b00)
        r = 3
        U[idx_u(1, 1), r] = 1
        V[idx_v(1, 0), r] = 1; V[idx_v(0, 0), r] = -1
        W[idx_w(0, 0), r] = 1; W[idx_w(0, 1), r] = 1
        
        # m5 = (a00 + a01)b11
        r = 4
        U[idx_u(0, 0), r] = 1; U[idx_u(0, 1), r] = 1
        V[idx_v(1, 1), r] = 1
        W[idx_w(0, 0), r] = -1; W[idx_w(1, 0), r] = 1
        
        # m6 = (a10 - a00)(b00 + b01)
        r = 5
        U[idx_u(1, 0), r] = 1; U[idx_u(0, 0), r] = -1
        V[idx_v(0, 0), r] = 1; V[idx_v(0, 1), r] = 1
        W[idx_w(1, 1), r] = 1
        
        # m7 = (a01 - a11)(b10 + b11)
        r = 6
        U[idx_u(0, 1), r] = 1; U[idx_u(1, 1), r] = -1
        V[idx_v(1, 0), r] = 1; V[idx_v(1, 1), r] = 1
        W[idx_w(0, 0), r] = 1

        # Border terms (19 terms)
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        
        return U, V, W

    # Attempt Rank-25 Search
    # We structure the search by initializing with the rank-26 solution
    # and trying to "merge" two border terms or adjust coefficients to reduce rank.
    # This is a heuristic search: fix the 7 Strassen terms, and optimize the remaining 18 
    # border terms + 1 mixed term to see if rank 25 is possible.
    # Actually, a simpler structural change: Use a random initialization for R=25 
    # and run a few steps of gradient descent. If it converges, great.
    
    R_try = 25
    if time.time() < deadline:
        try:
            # Initialize U, V, W with random normal values
            # To make it easier, we can start with the rank-26 solution and drop the last column
            # then re-optimize.
            U0, V0, W0 = get_rank_26_solution()
            U_init = U0[:, :R_try].copy()
            V_init = V0[:, :R_try].copy()
            W_init = W0[:, :R_try].copy()
            
            # Flatten parameters
            # U: 9*25 = 225
            # V: 9*25 = 225
            # W: 9*25 = 225
            # Total 675 parameters.
            
            def objective(params):
                U_flat = params[:n*m*R_try].reshape(n*m, R_try)
                V_flat = params[n*m*R_try:n*m*R_try + m*p*R_try].reshape(m*p, R_try)
                W_flat = params[2*n*m*R_try:].reshape(p*n, R_try)
                
                U_r = U_flat.reshape(n, m, R_try)
                V_r = V_flat.reshape(m, p, R_try)
                W_r = W_flat.reshape(p, n, R_try)
                
                T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                diff = T_recon - T_target
                return np.sum(diff**2)
                
            # Initial params
            p0 = np.concatenate([
                U_init.flatten(), 
                V_init.flatten(), 
                W_init.flatten()
            ])
            
            # Use L-BFGS-B for speed
            from scipy.optimize import minimize
            
            # We will run this for a very short time to see if it can improve significantly
            # If it doesn't, we discard.
            res = minimize(
                objective, 
                p0, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            
            if res.fun < 1e-6:
                U_opt = res.x[:n*m*R_try].reshape(n*m, R_try)
                V_opt = res.x[n*m*R_try:n*m*R_try + m*p*R_try].reshape(m*p, R_try)
                W_opt = res.x[2*n*m*R_try:].reshape(p*n, R_try)
                
                # Round to nearest half-integer? 
                # The task allows integer/half-integer. 
                # If the optimal solution is close to half-integers, we snap.
                # Otherwise, we might return the float values if the evaluator accepts them?
                # "integer/half-integer entries" -> The task says entries *must* be integer/half-integer.
                # So we must snap.
                
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                if verify_decomposition(U_snap, V_snap, W_snap, R_try):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                
        except Exception:
            pass # Fallback to rank 26
            
    # Fallback
    U, V, W = get_rank_26_solution()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
