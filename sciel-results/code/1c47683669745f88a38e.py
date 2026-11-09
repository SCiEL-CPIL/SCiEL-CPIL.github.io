import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]
TIME_LIMIT = 145  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    total_time_budget = TIME_LIMIT
    start_time = time.time()
    
    # Best known: schoolbook rank = 42
    # We'll try to find a lower rank decomposition using Winograd-style factorization
    # exploiting 7 = 3 + 4
    
    # Schoolbook decomposition as fallback
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook_decomposition()
    
    # Winograd-style approach: split B's columns into two groups: first 3, last 4
    # B = [B1 (3x3), B2 (3x4)]
    # AB = A * B = [A*B1, A*B2]
    # 
    # For A*B1: 2x3 times 3x3, schoolbook rank = 18
    # For A*B2: 2x3 times 3x4, schoolbook rank = 24
    # Total: 42 (same as schoolbook)
    # 
    # But we can use Winograd for the 2x2 subblocks!
    # 
    # Let's try a different approach: use the fact that 7 = 3+4 and apply
    # Winograd's method to reduce the 3x3 and 3x4 multiplications.
    #
    # Actually, let me try a more direct constructive approach:
    # 
    # The multiplication tensor M_{2,3,7} has a known lower bound. 
    # Let's try to construct a decomposition with rank < 42.
    #
    # One approach: use block structure. Write A as a 2x3 matrix, B as 3x7.
    # We can think of this as: for each column k of B, compute A * B[:,k].
    # This is 7 separate 2x3 times 3x1 multiplications, each of rank 3.
    # Total: 21, but we need to combine them.
    #
    # Better: use the structure that A is 2x3. We can precompute certain
    # linear combinations of rows of A.
    #
    # Let me try a randomized optimization approach with a better initialization.
    
    import random
    random.seed(42)
    np.random.seed(42)
    
    # Build the nonzero indices of the multiplication tensor
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_indices.append((i*m+j, j*p+k, i*n+k))
    
    # Convert to arrays for vectorized operations
    nz_a = np.array([x[0] for x in nz_indices])
    nz_b = np.array([x[1] for x in nz_indices])
    nz_c = np.array([x[2] for x in nz_indices])
    
    def evaluate_error(U, V, W):
        """Compute the error of the decomposition on the nonzero entries."""
        # For each nonzero entry (a,b,c), we need U[a,:]*V[b,:]*W[c,:] summed over r = 1
        # Vectorized: for each r, compute U[nz_a,r] * V[nz_b,r] * W[nz_c,r]
        rank = U.shape[1]
        vals = np.zeros(len(nz_a))
        for r in range(rank):
            vals += U[nz_a, r] * V[nz_b, r] * W[nz_c, r]
        # Error on nonzero entries: should be 1
        err_nz = np.sum((vals - 1.0) ** 2)
        
        # Check zero entries: sample some random ones
        num_zero_checks = min(2000, 20 * len(nz_indices))
        rng = np.random.RandomState(42)
        a_rand = rng.randint(0, n*m, num_zero_checks)
        b_rand = rng.randint(0, m*p, num_zero_checks)
        c_rand = rng.randint(0, n*p, num_zero_checks)
        
        # Check which are actually zero entries
        nz_set = set(nz_indices)
        valid_mask = np.array([((a_rand[i], b_rand[i], c_rand[i]) not in nz_set) for i in range(num_zero_checks)])
        
        if valid_mask.sum() > 0:
            a_valid = a_rand[valid_mask]
            b_valid = b_rand[valid_mask]
            c_valid = c_rand[valid_mask]
            
            vals_zero = np.zeros(len(a_valid))
            for r in range(rank):
                vals_zero += U[a_valid, r] * V[b_valid, r] * W[c_valid, r]
            err_zero = np.sum(vals_zero ** 2)
        else:
            err_zero = 0.0
        
        return err_nz + err_zero
    
    def try_rank(r, max_restarts=10, max_iters=300):
        """Try to find a rank-r decomposition using random restarts and ALS."""
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(max_restarts):
            if time.time() - start_time > total_time_budget - 10:
                break
            
            # Random initialization
            U_mat = np.random.randn(n*m, r) * 0.5
            V_mat = np.random.randn(m*p, r) * 0.5
            W_mat = np.random.randn(n*p, r) * 0.5
            
            # Alternating least squares: fix two, solve for third
            for it in range(max_iters):
                if time.time() - start_time > total_time_budget - 5:
                    break
                
                # Solve for U, fixing V and W
                # For each row a of U, we have equations:
                # sum_r U[a,r] * V[b,r] * W[c,r] = 1 for (a,b,c) in nz
                # Group by a: for each a, find all (b,c) such that (a,b,c) in nz
                # Then solve: U[a,:] @ (V[b,:] * W[c,:]) = 1
                # This is: U[a,:] @ M_a = 1 where M_a is a matrix with rows V[b,:] * W[c,:]
                
                # Build the linear system for U
                # For efficiency, precompute the grouping
                U_new = np.zeros_like(U_mat)
                for a in range(n*m):
                    # Find all (b,c) such that (a,b,c) in nz_indices
                    mask = (nz_a == a)
                    if not np.any(mask):
                        continue
                    b_vals = nz_b[mask]
                    c_vals = nz_c[mask]
                    
                    # Build M: each row is V[b,:] * W[c,:]
                    M = np.zeros((len(b_vals), r))
                    for idx in range(len(b_vals)):
                        M[idx, :] = V_mat[b_vals[idx], :] * W_mat[c_vals[idx], :]
                    
                    # Solve U[a,:] @ M.T = 1 (vector of ones)
                    # This is underdetermined if len(b_vals) < r, use least squares
                    try:
                        U_new[a, :], residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(len(b_vals)), rcond=None)
                    except:
                        U_new[a, :] = U_mat[a, :]
                
                U_mat = U_new
                
                # Solve for V, fixing U and W
                V_new = np.zeros_like(V_mat)
                for b in range(m*p):
                    mask = (nz_b == b)
                    if not np.any(mask):
                        continue
                    a_vals = nz_a[mask]
                    c_vals = nz_c[mask]
                    
                    M = np.zeros((len(a_vals), r))
                    for idx in range(len(a_vals)):
                        M[idx, :] = U_mat[a_vals[idx], :] * W_mat[c_vals[idx], :]
                    
                    try:
                        V_new[b, :], residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(len(a_vals)), rcond=None)
                    except:
                        V_new[b, :] = V_mat[b, :]
                
                V_mat = V_new
                
                # Solve for W, fixing U and V
                W_new = np.zeros_like(W_mat)
                for c in range(n*p):
                    mask = (nz_c == c)
                    if not np.any(mask):
                        continue
                    a_vals = nz_a[mask]
                    b_vals = nz_b[mask]
                    
                    M = np.zeros((len(a_vals), r))
                    for idx in range(len(a_vals)):
                        M[idx, :] = U_mat[a_vals[idx], :] * V_mat[b_vals[idx], :]
                    
                    try:
                        W_new[c, :], residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(len(a_vals)), rcond=None)
                    except:
                        W_new[c, :] = W_mat[c, :]
                
                W_mat = W_new
                
                # Check convergence
                err = evaluate_error(U_mat, V_mat, W_mat)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
                
                if err < 1e-12:
                    break
        
        if best_err < 1e-8:
            return best_U, best_V, best_W
        return None, None, None
    
    # Try ranks from 41 down to some lower bound
    # Lower bound for 2x3x7 is at least max(2*3, 3*7, 2*7) = 21, but likely higher
    # Let's try a few promising ranks
    
    for r in range(41, 25, -1):
        if time.time() - start_time > total_time_budget - 15:
            break
        
        U_r, V_r, W_r = try_rank(r, max_restarts=5, max_iters=200)
        
        if U_r is not None:
            # Verify the solution
            err = evaluate_error(U_r, V_r, W_r)
            if err < 1e-6:
                # Round to nearest half-integers if possible
                U_rounded = np.round(U_r * 2) / 2
                V_rounded = np.round(V_r * 2) / 2
                W_rounded = np.round(W_r * 2) / 2
                
                err_rounded = evaluate_error(U_rounded, V_rounded, W_rounded)
                if err_rounded < 1e-6:
                    best_rank = r
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                else:
                    best_rank = r
                    best_U, best_V, best_W = U_r, V_r, W_r
                break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < n * m * p:
        U, V, W = best_U, best_V, best_W
    else:
        U, V, W = schoolbook_decomposition()
    
    # Convert to lists for the return format
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
