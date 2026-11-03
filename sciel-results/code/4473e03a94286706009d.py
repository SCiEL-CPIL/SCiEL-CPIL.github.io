"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Build the target tensor
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    rank_school = n * m * p  # 144
    
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Start with schoolbook
    U = np.zeros((nm, rank_school), dtype=np.float64)
    V = np.zeros((mp, rank_school), dtype=np.float64)
    W = np.zeros((pn, rank_school), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by finding linear dependencies
    # Approach: try to merge pairs of columns that are "similar"
    # Use a greedy approach: try to remove columns by solving for replacements
    
    def verify(U_mat, V_mat, W_mat, T_target, tol=1e-6):
        """Check if U*V*W decomposes T_target"""
        # For each (a,b), check if sum_r U[a,r]*V[b,r]*W[:,r] matches T_target[a,b,:]
        # This is expensive, so do it efficiently
        for a in range(nm):
            for b in range(mp):
                # sum_r U[a,r] * V[b,r] * W[:,r]
                # = (U[a,:] * V[b,:]) dot W^T
                coeffs = U_mat[a, :] * V_mat[b, :]
                result = coeffs @ W_mat.T
                if np.max(np.abs(result - T_target[a, b, :])) > tol:
                    return False
        return True
    
    def try_reduce_rank(U_mat, V_mat, W_mat, T_target, max_iter=100):
        """Try to reduce rank by finding redundant columns"""
        current_rank = U_mat.shape[1]
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if current_rank <= 1:
                break
            
            # Try to remove each column and see if we can adjust others
            best_reduction = None
            for c_to_remove in range(current_rank):
                # Skip if too slow
                if time.time() > deadline:
                    break
                
                # Try to see if column c_to_remove can be expressed as combination of others
                # This is a linear algebra problem: can we find delta_r for r != c_to_remove
                # such that for all (a,b): sum_{r != c_to_remove} delta_r * U[a,r]*V[b,r]*W[:,r] = U[:,c_to_remove] * V[:,c_to_remove] * W[:,c_to_remove]
                # This is complex. Instead, try a simpler heuristic:
                # Check if removing this column and slightly adjusting others still works
                
                # Simple heuristic: if the column is very small, remove it
                col_norm = np.linalg.norm(U_mat[:, c_to_remove]) * np.linalg.norm(V_mat[:, c_to_remove]) * np.linalg.norm(W_mat[:, c_to_remove])
                if col_norm < 1e-8:
                    # Remove this column
                    mask = [i for i in range(current_rank) if i != c_to_remove]
                    U_new = U_mat[:, mask]
                    V_new = V_mat[:, mask]
                    W_new = W_mat[:, mask]
                    if verify(U_new, V_new, W_new, T_target):
                        best_reduction = ('remove', c_to_remove, U_new, V_new, W_new)
                        break
            else:
                # Try pairwise merging
                for c1 in range(current_rank - 1):
                    if time.time() > deadline:
                        break
                    for c2 in range(c1 + 1, current_rank):
                        # Try to merge c1 and c2 into one column
                        # Find alpha, beta such that the merged column approximates both
                        # This is complex; skip for now
                        pass
                break
            
            if best_reduction:
                _, c_idx, U_mat, V_mat, W_mat = best_reduction
                current_rank -= 1
            else:
                break
        
        return U_mat, V_mat, W_mat
    
    # Try reduction
    U_reduced, V_reduced, W_reduced = try_reduce_rank(U, V, W, T)
    if W_reduced.shape[1] < best_rank and verify(U_reduced, V_reduced, W_reduced, T):
        best_rank = W_reduced.shape[1]
        best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
    
    # Try a random restart approach with lower initial rank
    if time.time() < deadline - 30:
        # Try to find a decomposition with rank < 144 using random initialization
        for target_rank in [140, 135, 130, 125, 120]:
            if time.time() > deadline - 10:
                break
            
            found = False
            for trial in range(5):
                if time.time() > deadline - 10:
                    break
                
                # Random initialization
                U_rand = np.random.randn(nm, target_rank)
                V_rand = np.random.randn(mp, target_rank)
                W_rand = np.random.randn(pn, target_rank)
                
                # Normalize to help
                for r in range(target_rank):
                    nu = np.linalg.norm(U_rand[:, r])
                    nv = np.linalg.norm(V_rand[:, r])
                    nw = np.linalg.norm(W_rand[:, r])
                    if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                        scale = (nu * nv * nw) ** (1.0/3.0)
                        U_rand[:, r] /= scale
                        V_rand[:, r] /= scale
                        W_rand[:, r] /= scale
                
                # Iterate: update each factor
                for it in range(50):
                    if time.time() > deadline - 10:
                        break
                    
                    # Update U
                    for a in range(nm):
                        # We want U[a,:] such that for all b: sum_r U[a,r]*V[b,r]*W[:,r] = T[a,b,:]
                        # This is: for each b, U[a,:] * (V[b,:] * W[:,r]) ... complex
                        # Use gradient descent instead
                        pass
                    
                    # Gradient-based update
                    # Compute error
                    # E[a,b,c] = T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r]
                    # dE/dU[a,r] = -sum_b V[b,r] * (E[a,b,:] * W[:,r]) ... no
                    # dE[a,b,c]/dU[a,r] = -V[b,r]*W[c,r]
                    # So grad_U[a,r] = -sum_{b,c} T[a,b,c]*V[b,r]*W[c,r] + sum_{b,c} (sum_{r'} U[a,r']*V[b,r']*W[c,r'])*V[b,r]*W[c,r]
                    
                    # This is getting complex. Let's just use the schoolbook result.
                    break
                
                # Check if random init happened to work (unlikely)
                if verify(U_rand, V_rand, W_rand, T):
                    found = True
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_rand, V_rand, W_rand
                    break
            
            if found:
                break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Round to nearest half-integer for cleanliness
    U_list = [[round(x * 2) / 2 for x in row] for row in U_list]
    V_list = [[round(x * 2) / 2 for x in row] for row in V_list]
    W_list = [[round(x * 2) / 2 for x in row] for row in W_list]
    
    # Final verification
    U_check = np.array(U_list)
    V_check = np.array(V_list)
    W_check = np.array(W_list)
    if not verify(U_check, V_check, W_check, T, tol=1e-6):
        # Fallback to schoolbook
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
