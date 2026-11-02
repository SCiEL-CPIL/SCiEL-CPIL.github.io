import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 2, 6]
EVAL_TIMEOUT = 120
SAFETY_MARGIN = 5  # seconds

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i, j, k] = 1 if the (i,j) element of A times (j,k) element of B contributes to (i,k) element of C
    # A is (n x m), B is (m x p), C is (n x p)
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # U index: i * m + j  (i in 0..n-1, j in 0..m-1)
    # V index: j * p + k  (j in 0..m-1, k in 0..p-1)
    # W index: k * n + i  (k in 0..p-1, i in 0..n-1)
    
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1
    
    # Verify the rank-21 Strassen solution
    def get_rank_21():
        rank = 21
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
            ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
            ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
            ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
            ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
        ]
        
        r = 0
        for b in range(3):
            for t in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen[t]
                for idx, val in u_coeffs.items():
                    U[idx][r] = val
                for l, val in v_coeffs.items():
                    if l == 0: v_idx = 0 * p + 2 * b
                    elif l == 1: v_idx = 0 * p + 2 * b + 1
                    elif l == 2: v_idx = 1 * p + 2 * b
                    elif l == 3: v_idx = 1 * p + 2 * b + 1
                    V[v_idx][r] = val
                for (i, k_local), val in w_coeffs.items():
                    k_global = 2 * b + k_local
                    w_idx = k_global * n + i
                    W[w_idx][r] = val
                r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W):
        """Check if the decomposition exactly reconstructs T."""
        rank = len(U[0]) if U else 0
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # Reconstruct: for each rank term r, add outer product U[:,r] * V[:,r] * W[:,r]
        T_recon = np.zeros_like(T, dtype=float)
        for r in range(rank):
            for u_idx in range(n * m):
                for v_idx in range(m * p):
                    for w_idx in range(p * n):
                        T_recon[u_idx, v_idx, w_idx] += U_mat[u_idx, r] * V_mat[v_idx, r] * W_mat[w_idx, r]
        
        # Check exact match
        return np.allclose(T_recon, T, atol=1e-9)
    
    # First, verify the rank-21 solution
    rank21_sol = get_rank_21()
    if not verify_decomposition(rank21_sol['U'], rank21_sol['V'], rank21_sol['W']):
        raise ValueError("Rank-21 solution is invalid")
    
    # Try to find a rank-20 decomposition using structured search
    # The key insight: for 2x2 by 2x6, we can try to find a rank-20 solution
    # by exploiting the structure. The schoolbook rank is 24.
    
    # Strategy: Try to construct a rank-20 decomposition using the following approach:
    # 1. The tensor has a block structure with 3 blocks of 2x2x2 (for the 6 columns)
    # 2. Each 2x2x2 block can be done in rank 7 (Strassen), giving 21 total
    # 3. To get rank 20, we need to share one rank term across blocks
    
    # Alternative approach: Use integer/half-integer search with constraints
    # This is a complex search, so we'll use a heuristic approach
    
    # Try a direct search for rank-20 with structured patterns
    # Pattern: For each of the 3 blocks, use 7 terms but share one term
    
    # Actually, let's try a different approach: use the fact that the problem
    # has some symmetry and try to find a rank-20 solution directly
    
    # The rank of matrix multiplication [n,m,p] is the tensor rank of the matmul tensor
    # For [2,2,6], the known lower bound is 19 and upper bound is 24 (schoolbook)
    # The best known is 21 (from Strassen-like schemes)
    
    # Let's try a more systematic search using the structure
    # We'll parameterize the decomposition and try to find integer/half-integer solutions
    
    # For rank 20, we need 20 rank-1 tensors that sum to T
    # Each rank-1 tensor is (u, v, w) where u in R^(4), v in R^(12), w in R^(12)
    
    # This is a very large search space, so we need to use structure
    # The key structure: the tensor decomposes as a sum over the 3 blocks
    
    # Let's try: for block b (b=0,1,2), the contribution is from A[:, b*2:(b+1)*2] * B[b*2:(b+1)*2, :]
    # Each block is a 2x2x2 matmul
    
    # Approach: Try to find a common rank term that can be shared
    # This is essentially looking for a rank-1 update that works for all 3 blocks
    
    # Given the time constraints, let's try a different approach:
    # Use the known rank-21 solution and try to merge two terms
    
    # Check if we can merge any two rank terms from the rank-21 solution
    # Two terms (u1, v1, w1) and (u2, v2, w2) can be merged if
    # there exists a rank-1 tensor (u, v, w) such that
    # u1*v1*w1 + u2*v2*w2 = u*v*w
    # This requires u1*v1*w1 + u2*v2*w2 to be rank-1
    
    U21 = np.array(rank21_sol['U'], dtype=float)
    V21 = np.array(rank21_sol['V'], dtype=float)
    W21 = np.array(rank21_sol['W'], dtype=float)
    
    best_sol = rank21_sol
    
    # Try to find a rank-20 solution by merging two terms
    # For each pair of terms, check if their sum is rank-1
    # This is O(21^2) = 441 checks, each requiring a rank check
    
    for r1 in range(21):
        if time.time() - start_time > EVAL_TIMEOUT - SAFETY_MARGIN:
            break
        for r2 in range(r1 + 1, 21):
            if time.time() - start_time > EVAL_TIMEOUT - SAFETY_MARGIN:
                break
            
            # Sum of two rank-1 tensors
            sum_tensor = np.zeros((n * m, m * p, p * n), dtype=float)
            for i in range(n * m):
                for j in range(m * p):
                    for k in range(p * n):
                        sum_tensor[i, j, k] = U21[i, r1] * V21[j, r1] * W21[k, r1] + U21[i, r2] * V21[j, r2] * W21[k, r2]
            
            # Check if sum_tensor is rank-1
            # A tensor is rank-1 if it can be written as u * v * w
            # This is equivalent to: for any two slices, they are proportional
            # Or: the matrix unfolding has rank 1
            
            # Check via SVD of one unfolding
            # Unfold along the first mode: matrix of shape (4, 12*12) = (4, 144)
            mat1 = sum_tensor.reshape(n * m, -1)
            if np.linalg.matrix_rank(mat1) > 1:
                continue
            
            # Unfold along the second mode: matrix of shape (12, 4*12) = (12, 48)
            mat2 = sum_tensor.transpose(1, 0, 2).reshape(m * p, -1)
            if np.linalg.matrix_rank(mat2) > 1:
                continue
            
            # If both unfoldings have rank <= 1, the tensor is rank-1
            # Extract the rank-1 components
            # u = mat1[:, 0] / norm, etc.
            # But we need to check that the decomposition is consistent
            
            # Get u from the first unfolding
            u_vec = mat1[:, 0]
            if np.allclose(u_vec, 0):
                # Try another column
                for c in range(1, mat1.shape[1]):
                    if not np.allclose(mat1[:, c], 0):
                        u_vec = mat1[:, c]
                        break
            if np.allclose(u_vec, 0):
                continue
            
            # Get v*w from the remaining
            # sum_tensor[i, j, k] = u[i] * (v*w)[j, k]
            # So (v*w)[j, k] = sum_tensor[0, j, k] / u[0] if u[0] != 0
            if abs(u_vec[0]) < 1e-10:
                # Find a non-zero entry in u
                nz = np.where(np.abs(u_vec) > 1e-10)[0]
                if len(nz) == 0:
                    continue
                idx = nz[0]
                vw = sum_tensor[idx, :, :] / u_vec[idx]
            else:
                vw = sum_tensor[0, :, :] / u_vec[0]
            
            # Check if vw has rank 1
            vw_rank = np.linalg.matrix_rank(vw)
            if vw_rank > 1:
                continue
            
            # Get v and w from vw
            v_vec = vw[:, 0]
            if np.allclose(v_vec, 0):
                for c in range(1, vw.shape[1]):
                    if not np.allclose(vw[:, c], 0):
                        v_vec = vw[:, c]
                        break
            if np.allclose(v_vec, 0):
                continue
            
            # w[k] = vw[0, k] / v[0] if v[0] != 0
            if abs(v_vec[0]) < 1e-10:
                nz = np.where(np.abs(v_vec) > 1e-10)[0]
                if len(nz) == 0:
                    continue
                idx = nz[0]
                w_vec = vw[idx, :] / v_vec[idx]
            else:
                w_vec = vw[0, :] / v_vec[0]
            
            # Verify the decomposition
            recon = np.einsum('i,jk->ijk', u_vec, vw)
            if not np.allclose(recon, sum_tensor, atol=1e-8):
                continue
            
            # Build the new rank-20 solution
            new_U = [[0] * 20 for _ in range(n * m)]
            new_V = [[0] * 20 for _ in range(m * p)]
            new_W = [[0] * 20 for _ in range(p * n)]
            
            # Copy all terms except r1 and r2
            new_r = 0
            for r in range(21):
                if r == r1 or r == r2:
                    continue
                for i in range(n * m):
                    new_U[i][new_r] = U21[i, r]
                for j in range(m * p):
                    new_V[j][new_r] = V21[j, r]
                for k in range(p * n):
                    new_W[k][new_r] = W21[k, r]
                new_r += 1
            
            # Add the merged term
            for i in range(n * m):
                new_U[i][new_r] = u_vec[i]
            for j in range(m * p):
                new_V[j][new_r] = v_vec[j]
            for k in range(p * n):
                new_W[k][new_r] = w_vec[k]
            new_r += 1
            
            # Verify the full decomposition
            if verify_decomposition(new_U, new_V, new_W):
                best_sol = {"U": new_U, "V": new_V, "W": new_W}
                return best_sol
    
    # If no rank-20 solution found via merging, try a different approach
    # Use a structured search with integer/half-integer coefficients
    
    # Given the time, return the best solution found
    return best_sol
