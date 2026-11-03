import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (12, rank)
    
    # Construct the target tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    # But we only need to check the non-zero entries.
    # Non-zero entries are at (4*i+j, 4*j+k, 3*k+i) with value 1.
    # Let's store the list of (a, b, c) indices where T[a,b,c] = 1.
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    def check_solution(U, V, W):
        """
        U, V, W are lists of lists or numpy arrays.
        Returns True if they exactly decompose the tensor.
        """
        rank = len(U[0])
        # Convert to numpy for speed if needed, but for small sizes lists might be ok.
        # Let's use numpy for the dot product check.
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Check all target indices
        for a, b, c in target_indices:
            # Sum over r of U[a,r] * V[b,r] * W[c,r]
            val = np.dot(U_np[a, :], V_np[b, :])
            # Wait, the decomposition is sum_r U[a,r] V[b,r] W[c,r]
            # This is not a simple dot product of U and V.
            # It is element-wise product of U[a,:], V[b,:], W[c,:] summed.
            prod = U_np[a, :] * V_np[b, :] * W_np[c, :]
            s = np.sum(prod)
            if abs(s - 1.0) > 1e-9:
                return False
        
        # Check that all other entries are 0
        # This is expensive if we iterate all 12*16*12 = 2304 entries.
        # But we can just check a sample or rely on the fact that if the
        # rank is minimal and we found a solution, it's likely correct?
        # No, we must be exact.
        # However, if we start from schoolbook and do ALS, the "other" entries
        # are initially 0. If we change coefficients, they might become non-zero.
        # We must verify all entries.
        
        # Efficient check:
        # Compute the full tensor T_approx = sum_r u_r v_r^T w_r^T?
        # T_approx[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        
        # We can compute this by iterating over r.
        # T_approx = zeros((12, 16, 12))
        # for r in range(rank):
        #     T_approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        # Then check T_approx[target_indices] == 1 and T_approx[other] == 0.
        
        T_approx = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_approx += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
            
        # Check target indices
        for a, b, c in target_indices:
            if abs(T_approx[a, b, c] - 1.0) > 1e-9:
                return False
        
        # Check other indices
        # Create a mask of target indices
        mask = np.zeros((n*m, m*p, p*n), dtype=bool)
        for a, b, c in target_indices:
            mask[a, b, c] = True
            
        # Sum of absolute values of non-target entries
        non_target_sum = np.sum(np.abs(T_approx[~mask]))
        if non_target_sum > 1e-9:
            return False
            
        return True

    def als_decompose(rank, max_iters=50, tol=1e-6):
        """
        Alternating Least Squares to find U, V, W of given rank.
        Initializes randomly.
        """
        # Random initialization with small integers
        U = np.random.randint(-2, 3, (n*m, rank)) * 0.5
        V = np.random.randint(-2, 3, (m*p, rank)) * 0.5
        W = np.random.randint(-2, 3, (p*n, rank)) * 0.5
        
        # Ensure not all zeros
        if np.all(U == 0): U[0,0] = 1
        if np.all(V == 0): V[0,0] = 1
        if np.all(W == 0): W[0,0] = 1
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            # Update U
            # For each a, U[a, :] minimizes sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
            # This is a linear system for U[a, :].
            # Let M_r = V[:, r] * W[:, r] (outer product flattened? No, V is m*p, W is p*n)
            # The term is sum_r U[a,r] * (V[:,r] * W[:,r])_{b,c}
            # We need to solve for U[a, :] given V, W.
            # The objective is || T_a - U[a,:] M ||_F^2 where M is a matrix of size (m*p*p*n, rank)?
            # Actually, for fixed a, T_a is a vector of size (m*p * p*n) = 64.
            # The contribution of U[a,r] is U[a,r] * (V[:,r] * W[:,r]) flattened.
            # Let's construct the matrix A_u of size (64, rank).
            # A_u[:, r] = flatten(V[:, r] * W[:, r])? 
            # No, V[:, r] is size 16, W[:, r] is size 12.
            # The product in the tensor is element-wise multiplication of the vectors?
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
            # For fixed a, T[a, :, :] is a 16x12 matrix.
            # We want to approximate T_a by sum_r U[a,r] (V[:,r] * W[:,r]^T).
            # So we flatten T_a to a vector of size 192.
            # And we form a matrix B of size (192, rank) where B[:, r] = flatten(V[:,r] * W[:,r]^T).
            # Then U[a, :] is the least squares solution to B U[a,:]^T = T_a_flat.
            
            # Precompute B for all r?
            # B is (192, rank).
            B_u = np.zeros((m*p * p*n, rank))
            for r in range(rank):
                # V[:, r] is (16,), W[:, r] is (12,)
                # Outer product is (16, 12)
                outer = np.outer(V[:, r], W[:, r])
                B_u[:, r] = outer.flatten()
                
            # Solve for each a
            for a in range(n*m):
                # T_a is the slice of the target tensor at index a.
                # It is mostly zeros, except where (a,b,c) is in target_indices.
                T_a_flat = np.zeros(m*p * p*n)
                for b, c in [(b, c) for (aa, b, cc) in target_indices if aa == a]:
                    # Index in flattened (16, 12) matrix: b * 12 + c
                    T_a_flat[b * p*n + c] = 1.0
                
                # Solve B_u X = T_a_flat
                # Use least squares
                U[a, :], residuals, rank_l, sv = np.linalg.lstsq(B_u, T_a_flat, rcond=None)
                
            # Update V
            B_v = np.zeros((n*m * p*n, rank))
            for r in range(rank):
                outer = np.outer(U[:, r], W[:, r])
                B_v[:, r] = outer.flatten()
                
            for b in range(m*p):
                T_b_flat = np.zeros(n*m * p*n)
                for a, c in [(a, c) for (aa, bb, cc) in target_indices if bb == b]:
                    T_b_flat[a * p*n + c] = 1.0
                V[b, :], _, _, _ = np.linalg.lstsq(B_v, T_b_flat, rcond=None)
                
            # Update W
            B_w = np.zeros((n*m * m*p, rank))
            for r in range(rank):
                outer = np.outer(U[:, r], V[:, r])
                B_w[:, r] = outer.flatten()
                
            for c in range(p*n):
                T_c_flat = np.zeros(n*m * m*p)
                for a, b in [(a, b) for (aa, bb, cc) in target_indices if cc == c]:
                    T_c_flat[a * m*p + b] = 1.0
                W[c, :], _, _, _ = np.linalg.lstsq(B_w, T_c_flat, rcond=None)
            
            # Calculate error
            # Error = || T - sum U V W ||^2
            # We can just check the target indices and non-target indices roughly
            # For speed, we'll just check target indices sum of squares error
            err = 0
            for a, b, c in target_indices:
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                err += (val - 1.0)**2
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        if (a,b,c) not in target_indices: # This is slow
                            pass # Skip full check in loop for speed, rely on final check
                
            # Better error check:
            # T_approx = sum_r outer(U[:,r], outer(V[:,r], W[:,r]))
            # This is expensive.
            # Let's just use the target error for convergence check in ALS.
            # The non-target error might grow, but we check at the end.
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W

    # Fallback: Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a][col] = 1
                V_school[b][col] = 1
                W_school[c][col] = 1
                col += 1
                
    best_solution = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank
    # We try ranks from 47 down to 40
    for r in range(rank_school - 1, 39, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        for restart in range(10):
            if time.time() > deadline:
                break
                
            U, V, W = als_decompose(r, max_iters=20)
            
            # Round to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Convert to lists for check_solution
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            if check_solution(U_list, V_list, W_list):
                best_solution = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                # If we found a lower rank, we can continue searching for even lower
                # But we break out of restarts for this rank?
                # No, we can try to find an even better one? No, rank is fixed.
                # We break to try next lower rank.
                break
                
    return best_solution
