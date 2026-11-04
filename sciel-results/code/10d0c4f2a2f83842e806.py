"""
Strategy: Combinatorial Greedy Search for Tensor Rank.
Instead of relying on fixed Strassen tilings, we generate a massive pool of random
rank-1 tensor candidates (with integer/half-integer entries) and use a greedy
algorithm to find a sparse linear combination that exactly reconstructs the
matrix multiplication tensor. This allows discovery of non-standard decompositions
that might have lower rank than known tiling methods.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # 1. Baseline: Schoolbook (Rank 96)
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook

    # 2. Generate Candidate Pool
    # We need rank-1 tensors T[i,j,k] = U[i,j] * V[j,k] * W[k,i]
    # with entries in {-1, -0.5, 0.5, 1}.
    # To ensure the final sum is integer, we can restrict coefficients to integers 
    # or ensure the tensor itself allows integer summation.
    # The task allows half-integers in U,V,W.
    
    # Let's generate candidates where U, V, W have small support.
    # Support size 1: Unit vectors. (Already covered by schoolbook)
    # Support size 2: Pairs of indices.
    
    candidates = []
    candidate_factors = [] # Store (U_flat, V_flat, W_flat) to reconstruct later
    
    # Precompute indices
    u_indices = list(range(n * m))
    v_indices = list(range(m * p))
    w_indices = list(range(p * n))
    
    values = [-1.0, -0.5, 0.5, 1.0]
    
    # Add all "unit" candidates (support size 1)
    for u_idx in u_indices:
        for v_idx in v_indices:
            # Check compatibility: u_idx (i,j), v_idx (j,k) -> k must match w_idx (k,i)
            i = u_idx // m
            j = u_idx % m
            j_v = v_idx // p
            k = v_idx % p
            if j == j_v:
                w_idx = k * n + i
                # T[i,j,k] = 1
                # This is the schoolbook basis.
                # We might not need to add all of these to the pool if we handle them separately,
                # but for a general search, they are valid candidates.
                pass # Skip adding all 96 schoolbook terms to the random pool to save memory, 
                     # we will use them as a fallback or part of a hybrid.
    
    # Generate random sparse candidates
    # Focus on support sizes that allow cancellation.
    # A rank-1 term with support 2 in U, 2 in V, 2 in W creates a 4x4x4 sub-tensor pattern.
    
    num_random_candidates = 50000
    rng = np.random.RandomState(42)
    
    # We will represent each candidate as a flat vector of length n*m*p
    # and store the factors.
    
    # Optimization: Pre-calculate the "shape" of the tensor for quick checking
    # T[i, j, k] is non-zero only if U[i,j]!=0, V[j,k]!=0, W[k,i]!=0.
    
    # To make the search effective, we should generate candidates that are "orthogonal" 
    # or at least diverse.
    
    # Let's try a specific structured search:
    # Use "Strassen-like" patterns but allow random signs and shifts.
    
    # Actually, a better strategy for 4x4x6 is to try to find a decomposition 
    # that reuses computations.
    
    # Let's generate candidates with support size 2 for U, V, W.
    # There are C(16,2)*C(24,2)*C(24,2) ~ 120 * 276 * 276 ~ 9M combinations.
    # Too many to iterate all. We will sample.
    
    # But we must ensure the resulting tensor has integer entries? 
    # No, the *sum* must be the identity tensor (all 1s).
    # If we use half-integers in factors, the product might be 0.25, 0.5, etc.
    # The target is 1.0.
    # So we need combinations of these to sum to 1.0.
    
    # Let's restrict the generated candidates to have entries in {-1, 1} for simplicity 
    # first, then allow half-integers if needed.
    
    # Revised Plan:
    # 1. Generate a pool of candidates with entries in {-1, 1} (support size 1 or 2).
    # 2. Use Greedy Set Cover to select a subset that sums to the target tensor.
    # 3. If exact match isn't found with integers, try half-integers.
    
    pool_size = 100000
    tensor_dim = n * m * p
    target = np.ones(tensor_dim)
    
    # Candidate storage
    # For memory efficiency, store as sparse representations or flat arrays.
    # Flat arrays of 100k * 96 floats = 9.6M floats = ~76MB. Acceptable.
    
    C = np.zeros((pool_size, tensor_dim))
    factors_list = []
    
    # Helper to build a candidate
    def build_candidate(u_sup, u_vals, v_sup, v_vals, w_sup, w_vals):
        vec = np.zeros(tensor_dim)
        u_dict = dict(zip(u_sup, u_vals))
        v_dict = dict(zip(v_sup, v_vals))
        w_dict = dict(zip(w_sup, w_vals))
        
        for i in range(n):
            for j in range(m):
                u_val = u_dict.get((i,j), 0)
                if u_val == 0: continue
                for k in range(p):
                    v_val = v_dict.get((j,k), 0)
                    w_val = w_dict.get((k,i), 0)
                    if v_val != 0 and w_val != 0:
                        vec[i * m * p + j * p + k] = u_val * v_val * w_val
        return vec
    
    # Generate Schoolbook candidates (Rank 1, support 1)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_sup = [(i,j)]
                v_sup = [(j,k)]
                w_sup = [(k,i)]
                vec = build_candidate(u_sup, [1], v_sup, [1], w_sup, [1])
                C[idx] = vec
                factors_list.append((u_sup, [1], v_sup, [1], w_sup, [1]))
                idx += 1
                if idx >= pool_size:
                    break
    
    # Generate Random Sparse Candidates (Support size 2)
    # We need to be careful with indices.
    # U: (i,j), V: (j,k), W: (k,i).
    
    while idx < pool_size:
        if time.time() > deadline - 20:
            break
            
        # Random support sizes
        su = rng.randint(1, 3)
        sv = rng.randint(1, 3)
        sw = rng.randint(1, 3)
        
        u_sup = list(set(rng.choice(n*m, size=su, replace=False)))
        v_sup = list(set(rng.choice(m*p, size=sv, replace=False)))
        w_sup = list(set(rng.choice(p*n, size=sw, replace=False)))
        
        # Convert to (i,j) etc for building
        u_pairs = [(int(x)//m, int(x)%m) for x in u_sup]
        v_pairs = [(int(x)//p, int(x)%p) for x in v_sup]
        w_pairs = [(int(x)//n, int(x)%n) for x in w_sup]
        
        # Random values from {-1, 1}
        u_vals = [rng.choice([-1, 1]) for _ in range(len(u_pairs))]
        v_vals = [rng.choice([-1, 1]) for _ in range(len(v_pairs))]
        w_vals = [rng.choice([-1, 1]) for _ in range(len(w_pairs))]
        
        vec = build_candidate(u_pairs, u_vals, v_pairs, v_vals, w_pairs, w_vals)
        
        if np.any(vec != 0):
            C[idx] = vec
            factors_list.append((u_pairs, u_vals, v_pairs, v_vals, w_pairs, w_vals))
            idx += 1
    
    C = C[:idx]
    factors_list = factors_list[:idx]
    
    # 3. Greedy Search
    # We want sum_{r in S} coeff_r * C[r] = target
    # Coeffs should be integers (or half-integers) to keep U,V,W entries clean?
    # Actually, if C[r] has integer entries, and we use integer coeffs, the sum is integer.
    # The target is 1.
    
    residual = target.copy()
    chosen_indices = []
    chosen_coeffs = []
    
    # Precompute norms
    norms_sq = np.sum(C * C, axis=1)
    norms_sq[norms_sq < 1e-12] = 1.0
    
    max_iter = 100
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        if np.linalg.norm(residual) < 1e-9:
            break
            
        # Correlation with residual
        corrs = C @ residual
        scores = np.abs(corrs) / norms_sq
        
        # Mask out already chosen? 
        # Greedy usually allows re-selection but we want a sparse set.
        # We can set scores of chosen indices to 0 to prevent immediate re-selection 
        # unless it helps reduce residual further (unlikely in this context).
        
        best_idx = np.argmax(scores)
        
        if scores[best_idx] < 1e-8:
            break
            
        # Determine coefficient
        # Project residual onto C[best_idx]
        proj = corrs[best_idx] / norms_sq[best_idx]
        
        # Round to nearest integer
        coeff = np.round(proj)
        if coeff == 0:
            coeff = 1 if proj > 0 else -1
            
        new_residual = residual - coeff * C[best_idx]
        
        if np.linalg.norm(new_residual) < np.linalg.norm(residual) - 1e-9:
            residual = new_residual
            chosen_indices.append(best_idx)
            chosen_coeffs.append(coeff)
        else:
            # Try opposite sign
            coeff = -coeff
            new_residual = residual - coeff * C[best_idx]
            if np.linalg.norm(new_residual) < np.linalg.norm(residual) - 1e-9:
                residual = new_residual
                chosen_indices.append(best_idx)
                chosen_coeffs.append(coeff)
            else:
                break
                
    if len(chosen_indices) > 0 and np.linalg.norm(residual) < 1e-8:
        rank_found = len(chosen_indices)
        if rank_found < best_rank:
            # Reconstruct U, V, W
            U = [[0] * rank_found for _ in range(n * m)]
            V = [[0] * rank_found for _ in range(m * p)]
            W = [[0] * rank_found for _ in range(p * n)]
            
            for col, (c_idx, coeff) in enumerate(zip(chosen_indices, chosen_coeffs)):
                u_pairs, u_vals, v_pairs, v_vals, w_pairs, w_vals = factors_list[c_idx]
                
                for (i, j), val in zip(u_pairs, u_vals):
                    U[i * m + j][col] += coeff * val
                for (j, k), val in zip(v_pairs, v_vals):
                    V[j * p + k][col] += coeff * val
                for (k, i), val in zip(w_pairs, w_vals):
                    W[k * n + i][col] += coeff * val
                    
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank_found
                print(f"Found rank {rank_found} via greedy search")
            else:
                print("Greedy search result failed verification")
    else:
        print(f"Greedy search did not converge or rank not improved (best so far: {best_rank})")
        
    return best_result

def _verify_decomposition(U, V, W, n, m, p, num_trials=3, seed=12345):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    rng = np.random.RandomState(seed)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Optimization: Use matrix multiplication to verify
    # Result[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is not a simple matrix product due to the W index depending on i,k.
    # But we can vectorize.
    
    # For each r, let U_r be vector of length nm, V_r of length mp, W_r of length pn.
    # T_r[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # Let's just do the loop for small n,m,p to be safe and fast enough for 3 trials.
    for _ in range(num_trials):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        expected = A @ B
        result = np.zeros((n, p))
        
        for i in range(n):
            for k in range(p):
                s = 0.0
                W_row = W_np[k * n + i]
                for j in range(m):
                    U_row = U_np[i * m + j]
                    V_row = V_np[j * p + k]
                    s += np.dot(np.multiply(U_row, V_row), W_row)
                result[i, k] = s
                
        if not np.allclose(result, expected, atol=1e-9):
            return False
    return True

# EVOLVE-BLOCK-END
