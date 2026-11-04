"""
Find rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication.

Strategy:
1. Define the target tensor T of shape (n*m, m*p, p*n) based on the trace convention.
2. Use a greedy algorithm to select rank-1 terms (vectors with entries in {-1, 0, 1})
   that maximize the reduction of the remaining tensor error.
3. This approach naturally produces integer/half-integer solutions, satisfying the constraint.
4. If the greedy search fails to find a low-rank solution within the time budget,
   fall back to the schoolbook decomposition (rank n*m*p).
"""

import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Dimensions for the factor matrices
    dim_u = n * m  # 8
    dim_v = m * p  # 28
    dim_w = p * n  # 14
    
    # Build the target tensor T[a, b, c]
    # a = i*m + j (input A index)
    # b = j*p + k (input B index)
    # c = k*n + i (output C index, per W[k*n+i] convention)
    # T[a, b, c] = 1 if the triple (a,b,c) corresponds to a valid multiplication step
    # Specifically, for each (i, j, k), we have:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # The tensor is a sum of rank-1 tensors? No, the tensor T is the "structure" tensor.
    # We want T = sum_r U_r outer V_r outer W_r.
    # The entries of T are 1 for the valid triples and 0 elsewhere.
    
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
                
    # The total number of non-zero entries is n*m*p = 56.
    
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Greedy search for rank-1 terms with entries in {-1, 0, 1}
    # We want to approximate T with sum of outer products.
    # Since T has only 1s and 0s, and we want integer coefficients, 
    # a natural candidate for rank-1 terms is to pick subsets of indices.
    
    # A specific strategy:
    # The tensor T is sparse. Each non-zero entry T[a,b,c]=1 involves specific a,b,c.
    # A rank-1 term U outer V outer W will have U[a]V[b]W[c] at that position.
    # If we restrict U, V, W to have entries in {0, 1}, then a term covers a set of entries
    # where U[a]=1, V[b]=1, W[c]=1. To exactly match T, we need the sum of these indicators
    # to be 1 for all (a,b,c) in the support of T, and 0 elsewhere.
    
    # However, {0,1} vectors might not be sufficient if the support has overlapping structures.
    # Using {-1, 0, 1} allows for cancellation.
    
    # Let's try a greedy approach:
    # 1. Start with Residual R = T.
    # 2. Find a rank-1 tensor X (with small integer entries) that best correlates with R.
    # 3. Subtract X from R.
    # 4. Repeat until R is zero or time runs out.
    
    # To make this efficient, we can precompute potential "atoms".
    # But the space of atoms is huge. 
    # Instead, let's use a randomized greedy approach with local search.
    
    # Alternative: Since n=2 is small, we can exploit the structure.
    # C = A B.
    # A is 2x4, B is 4x7.
    # C[0, :] = A[0, :] B
    # C[1, :] = A[1, :] B
    
    # Let's try to find a decomposition with rank < 56.
    # The Strassen rank for 2x2 is 7. For 2x4x7, we can potentially do better than 56.
    # Upper bound: n * rank(m x p) ? No.
    # Simple upper bound: n * m * p / 2 ?
    
    # Let's implement a random restart greedy search.
    
    def try_greedy_rank(max_rank, seed):
        np.random.seed(seed)
        R = T.copy().astype(np.float64)
        U_cols = []
        V_cols = []
        W_cols = []
        
        for r in range(max_rank):
            if time.time() > deadline:
                return None
            
            # Find best rank-1 approximation to R
            # We will sample random sign vectors and 0/1 masks.
            # Actually, let's use a more structured approach:
            # Pick random sparse vectors U, V, W with entries in {-1, 1} where non-zero.
            # To maximize correlation, we can align signs with R.
            
            # Simple heuristic:
            # 1. Generate random supports for U, V, W.
            # 2. Set values to match sign of R on the support.
            
            best_corr = 0
            best_U = None
            best_V = None
            best_W = None
            
            # Try a few random candidates
            for _ in range(20):
                # Random sparsity
                p_u = np.random.uniform(0.1, 0.5)
                p_v = np.random.uniform(0.1, 0.5)
                p_w = np.random.uniform(0.1, 0.5)
                
                mask_u = np.random.rand(dim_u) < p_u
                mask_v = np.random.rand(dim_v) < p_v
                mask_w = np.random.rand(dim_w) < p_w
                
                if not np.any(mask_u) or not np.any(mask_v) or not np.any(mask_w):
                    continue
                
                # Construct candidate term
                # We want U outer V outer W to approximate R.
                # Let's try to set U, V, W such that the term is "aligned" with R.
                # This is a non-linear problem. 
                # Approximation: 
                # If we fix V and W, U is the projection of R onto V outer W.
                # Let's iterate locally for a few steps for each candidate.
                
                U_cand = np.zeros(dim_u)
                V_cand = np.zeros(dim_v)
                W_cand = np.zeros(dim_w)
                
                # Initialize with random signs on supports
                U_cand[mask_u] = np.random.choice([-1, 1], size=np.sum(mask_u))
                V_cand[mask_v] = np.random.choice([-1, 1], size=np.sum(mask_v))
                W_cand[mask_w] = np.random.choice([-1, 1], size=np.sum(mask_w))
                
                # Local refinement: 
                # Maximize sum_{a,b,c} R[a,b,c] U[a] V[b] W[c]
                # This is the tensor nuclear norm / alignment.
                # We can do a few rounds of alternating updates.
                for _ in range(10):
                    # Update U
                    # U[a] = sign( sum_{b,c} R[a,b,c] V[b] W[c] )
                    temp_u = np.zeros(dim_u)
                    # Efficiently: 
                    # V outer W is a matrix of shape dim_v x dim_w
                    # R is dim_u x dim_v x dim_w
                    # We can contract R with V and W.
                    # R_contracted[a] = sum_{b,c} R[a,b,c] V[b] W[c]
                    # = sum_b (sum_c R[a,b,c] W[c]) V[b]
                    
                    # Let's compute this. 
                    # To save time, we only do this if we have budget.
                    if time.time() > deadline:
                        return None
                    
                    # Flatten R for speed? 
                    # R[a, b, c]
                    # Let's just do it directly for small dims.
                    for a in range(dim_u):
                        if not mask_u[a]:
                            temp_u[a] = 0
                            continue
                        s = 0.0
                        for b in range(dim_v):
                            if V_cand[b] == 0: continue
                            s += V_cand[b] * np.dot(R[a, b, :], W_cand)
                        temp_u[a] = 1.0 if s > 0 else (-1.0 if s < 0 else 0.0)
                    U_cand = temp_u
                    
                    # Update V
                    temp_v = np.zeros(dim_v)
                    for b in range(dim_v):
                        if not mask_v[b]:
                            temp_v[b] = 0
                            continue
                        s = 0.0
                        for a in range(dim_u):
                            if U_cand[a] == 0: continue
                            s += U_cand[a] * np.dot(R[a, b, :], W_cand)
                        temp_v[b] = 1.0 if s > 0 else (-1.0 if s < 0 else 0.0)
                    V_cand = temp_v
                    
                    # Update W
                    temp_w = np.zeros(dim_w)
                    for c in range(dim_w):
                        if not mask_w[c]:
                            temp_w[c] = 0
                            continue
                        s = 0.0
                        for a in range(dim_u):
                            if U_cand[a] == 0: continue
                            for b in range(dim_v):
                                if V_cand[b] == 0: continue
                                s += U_cand[a] * V_cand[b] * R[a, b, c]
                        temp_w[c] = 1.0 if s > 0 else (-1.0 if s < 0 else 0.0)
                    W_cand = temp_w
                    
                # Calculate correlation (inner product with R)
                # Inner product = sum R * (U outer V outer W)
                # = sum_a U[a] * (sum_b V[b] * (sum_c W[c] * R[a,b,c]))
                corr = 0.0
                for a in range(dim_u):
                    if U_cand[a] == 0: continue
                    for b in range(dim_v):
                        if V_cand[b] == 0: continue
                        corr += U_cand[a] * V_cand[b] * np.dot(R[a, b, :], W_cand)
                
                if corr > best_corr:
                    best_corr = corr
                    best_U = U_cand.copy()
                    best_V = V_cand.copy()
                    best_W = W_cand.copy()
            
            if best_U is None or best_corr < 1e-6:
                break
            
            # Subtract the term from Residual
            # R -= best_U outer best_V outer best_W
            for a in range(dim_u):
                if best_U[a] == 0: continue
                for b in range(dim_v):
                    if best_V[b] == 0: continue
                    R[a, b, :] -= best_U[a] * best_V[b] * best_W
            
            U_cols.append(best_U)
            V_cols.append(best_V)
            W_cols.append(best_W)
            
            # Check if R is zero
            if np.allclose(R, 0, atol=1e-9):
                break
        
        if np.allclose(R, 0, atol=1e-9):
            # Success
            U_mat = np.column_stack(U_cols)
            V_mat = np.column_stack(V_cols)
            W_mat = np.column_stack(W_cols)
            
            # Verify
            T_approx = np.zeros_like(T)
            for r_idx in range(len(U_cols)):
                T_approx += np.outer(np.outer(U_mat[:, r_idx], V_mat[:, r_idx]), W_mat[:, r_idx])
            
            if np.allclose(T_approx, T, atol=1e-9):
                return U_mat, V_mat, W_mat, len(U_cols)
        
        return None

    # Try to find a solution with rank < 56
    # We know schoolbook is 56.
    # Let's try ranks from 30 to 55.
    # If we find one, great. If not, return schoolbook.
    
    best_result = None
    # Use multiple seeds
    for rank in range(30, 56):
        if time.time() > deadline:
            break
        for seed in range(5):
            if time.time() > deadline:
                break
            result = try_greedy_rank(rank, seed)
            if result:
                U, V, W, r_found = result
                if best_result is None or r_found < best_result[3]:
                    best_result = (U, V, W, r_found)
                break
    
    if best_result:
        U, V, W, r_found = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: Schoolbook decomposition (Rank 56)
    rank = n * m * p
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    construct()
