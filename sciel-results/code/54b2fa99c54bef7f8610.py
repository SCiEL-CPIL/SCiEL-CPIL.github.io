import numpy as np
from scipy.optimize import least_squares
import time

def construct():
    n, m, p = 2, 3, 3
    deadline = time.time() + 140
    
    # Precompute the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1
    
    nz_idx = np.argwhere(T != 0)
    a_idx = nz_idx[:, 0]
    b_idx = nz_idx[:, 1]
    c_idx = nz_idx[:, 2]
    
    def als_decompose(rank, max_iters=200, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        prev_cost = np.inf
        for it in range(max_iters):
            # Update U: for each a, solve min sum_b T[a,b,:]*V[b,s]*W[c,s] ... 
            # Actually, the tensor equation is: T[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s]
            # For fixed V,W, U[a,s] can be found by least squares:
            # For each a, we want: sum_s U[a,s] * (V[b,s]*W[c,s]) = T[a,b,c] for all b,c
            # This is a linear system in U[a,:]
            
            # Build the system for U
            # For each a, the equation is: U[a,:] @ (V*W)_b,c = T[a,b,c]
            # where (V*W)_b,c = sum_s V[b,s]*W[c,s]
            # This is a rank-rank matrix: M_a = V @ W.T (shape m*p x p*n)
            # But we need T[a,b,c] for all b,c
            
            # Actually, let me think more carefully.
            # T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
            # Let M[s,b,c] = V[b,s]*W[c,s] (shape rank x m*p x p*n)
            # Then T[a,:,:] = sum_s U[a,s] * M[s,:,:]
            # This is: T_flat[a] = U[a,:] @ M_flat
            # where M_flat has shape (m*p*p*n, rank) with M_flat[(b,c),s] = V[b,s]*W[c,s]
            
            # For efficiency, only use nonzero entries
            # For nonzero (a,b,c): T[a,b,c] = 1 = sum_s U[a,s]*V[b,s]*W[c,s]
            # Group by a: for each a, we have equations for all (b,c) where T[a,b,c]!=0
            
            # Build design matrix for U update
            # For each a, collect all (b,c) pairs where T[a,b,c]!=0
            # The equation is: sum_s U[a,s] * (V[b,s]*W[c,s]) = 1
            
            # This is a linear system: D_a @ U[a,:] = 1
            # where D_a has rows indexed by (b,c) pairs, columns by s
            # D_a[(b,c), s] = V[b,s]*W[c,s]
            
            # For efficiency, precompute V*W for all nonzero (b,c)
            # But it depends on V and W which change each iteration
            
            # Let me just do it directly
            U_new = np.zeros_like(U)
            for a in range(n*m):
                # Find all (b,c) where T[a,b,c] != 0
                mask = (nz_idx[:, 0] == a)
                if not np.any(mask):
                    continue
                b_sub = b_idx[mask]
                c_sub = c_idx[mask]
                # Build design matrix: shape (num_eq, rank)
                D = V[b_sub, :] * W[c_sub, :]  # broadcasting: (num_eq, rank)
                # Solve D @ U[a,:] = 1
                try:
                    U_new[a, :] = np.linalg.lstsq(D, np.ones(len(b_sub)), rcond=None)[0]
                except:
                    U_new[a, :] = U[a, :]
            U = U_new
            
            # Similarly for V
            V_new = np.zeros_like(V)
            for b in range(m*p):
                mask = (nz_idx[:, 1] == b)
                if not np.any(mask):
                    continue
                a_sub = a_idx[mask]
                c_sub = c_idx[mask]
                D = U[a_sub, :] * W[c_sub, :]
                try:
                    V_new[b, :] = np.linalg.lstsq(D, np.ones(len(a_sub)), rcond=None)[0]
                except:
                    V_new[b, :] = V[b, :]
            V = V_new
            
            # Similarly for W
            W_new = np.zeros_like(W)
            for c in range(p*n):
                mask = (nz_idx[:, 2] == c)
                if not np.any(mask):
                    continue
                a_sub = a_idx[mask]
                b_sub = b_idx[mask]
                D = U[a_sub, :] * V[b_sub, :]
                try:
                    W_new[c, :] = np.linalg.lstsq(D, np.ones(len(a_sub)), rcond=None)[0]
                except:
                    W_new[c, :] = W[c, :]
            W = W_new
            
            # Compute cost
            R_nz = np.zeros(len(nz_idx))
            for s in range(rank):
                R_nz += U[a_idx, s] * V[b_idx, s] * W[c_idx, s]
            cost = np.sum((R_nz - 1.0)**2)
            
            if abs(prev_cost - cost) < tol:
                break
            prev_cost = cost
        
        return U, V, W, cost
    
    def try_rank_als(rank, num_restarts=5, time_budget=20):
        start_time = time.time()
        best = None
        best_cost = np.inf
        
        for r in range(num_restarts):
            if time.time() - start_time > time_budget:
                break
            np.random.seed(5000 + rank * 100 + r * 31)
            U, V, W, cost = als_decompose(rank, max_iters=300, tol=1e-12)
            if cost < best_cost:
                best_cost = cost
                best = (U, V, W)
            if cost < 1e-8:
                break
        
        if best is None or best_cost > 1e-4:
            return None, best_cost
        
        U, V, W = best
        
        # Snap to half-integers
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        
        R_check = np.einsum('as,bs,cs->abc', U_half, V_half, W_half)
        if np.allclose(R_check, T, atol=1e-9):
            return (U_half, V_half, W_half), best_cost
        
        # Try integer rounding
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        R_check2 = np.einsum('as,bs,cs->abc', U_int, V_int, W_int)
        if np.allclose(R_check2, T, atol=1e-9):
            return (U_int, V_int, W_int), best_cost
        
        return None, best_cost
    
    # Try ranks from 13 to 17
    for rank in range(13, 18):
        if time.time() > deadline - 25:
            break
        
        sol, cost = try_rank_als(rank, num_restarts=5, time_budget=22)
        if sol is not None:
            U, V, W = sol
            U_out = [[float(x) for x in row] for row in U]
            V_out = [[float(x) for x in row] for row in V]
            W_out = [[float(x) for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
