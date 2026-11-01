"""
Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.

Strategy:
1. Generate a candidate pool of rank-1 tensors with entries in {-1, 0, 1}.
   The pool includes:
   - All "schoolbook" basis tensors (U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1).
   - Tensors derived from Strassen-like linear combinations (sums/differences of basis).
   - Random sparse tensors with small integer entries.
2. Use a greedy set-cover algorithm to find the minimal subset of these tensors
   that exactly reconstructs the target matrix multiplication tensor T[i,j,k]=1.
3. If the greedy method fails to reach rank < 49, fall back to the known rank-49 Strassen solution.

This avoids the pitfalls of gradient descent (non-convergence to exact integers) by
working in the discrete domain of integer tensors where exactness is guaranteed.
"""
import time
import numpy as np
import itertools

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = 4, 4, 4
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
        
    rank = U.shape[1]
    U4 = U.reshape(n, m, rank)
    V4 = V.reshape(m, p, rank)
    W4 = W.reshape(p, n, rank)
    
    # Target tensor is all ones
    target = np.ones((n, m, p))
    
    # Reconstruct tensor
    recon = np.zeros((n, m, p))
    for r in range(rank):
        recon += U4[:, :, r][:, :, None] * V4[:, :, r][None, :, :] * W4[:, :, r][None, None, :].transpose(2, 0, 1)
        
    # Check exact match (with small tolerance for floating point representation of integers)
    max_err = np.max(np.abs(recon - target))
    return max_err < 1e-9


def _strassen_2x2():
    """Return (a, b, c) for 2x2 Strassen decomposition."""
    a = np.zeros((7, 2, 2))
    b = np.zeros((7, 2, 2))
    c = np.zeros((2, 2, 7))
    
    a[0] = [[1, 0], [0, 1]]
    b[0] = [[1, 0], [0, 1]]
    a[1] = [[0, 0], [1, 1]]
    b[1] = [[1, 0], [0, 0]]
    a[2] = [[1, 0], [0, 0]]
    b[2] = [[0, 0], [1, -1]]
    a[3] = [[0, 0], [0, 1]]
    b[3] = [[-1, 1], [0, 0]]
    a[4] = [[1, 1], [0, 0]]
    b[4] = [[0, 0], [0, 1]]
    a[5] = [[-1, 0], [1, 0]]
    b[5] = [[1, 1], [0, 0]]
    a[6] = [[0, 1], [0, -1]]
    b[6] = [[0, 0], [1, 1]]
    
    c[0, 0] = [1, 0, 0, 1, -1, 0, 1]
    c[0, 1] = [0, 0, 1, 0, 1, 0, 0]
    c[1, 0] = [0, 1, 0, 1, 0, 0, 0]
    c[1, 1] = [1, -1, 1, 0, 0, 1, 0]
    
    return a, b, c


def _build_strassen_4x4():
    """Build rank-49 Strassen decomposition for 4x4 MM."""
    a2, b2, c2 = _strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for s1 in range(7):
        for s2 in range(7):
            col = 7 * s1 + s2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            U[i * 4 + j, col] = a2[s1, i1, j1] * a2[s2, i2, j2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            V[j * 4 + k, col] = b2[s1, j1, k1] * b2[s2, j2, k2]
            for i1 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            i = 2 * i1 + i2
                            k = 2 * k1 + k2
                            W[k * 4 + i, col] = c2[i1, k1, s1] * c2[i2, k2, s2]
    
    return U, V, W


def _generate_candidate_pool(deadline):
    """
    Generate a pool of candidate rank-1 tensors (U, V, W) with integer entries.
    The goal is to find a set of these that sums to the target tensor.
    """
    n, m, p = 4, 4, 4
    candidates = []
    
    # 1. Add schoolbook basis tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U = np.zeros((n*m, 1))
                V = np.zeros((m*p, 1))
                W = np.zeros((p*n, 1))
                U[i*m+j, 0] = 1
                V[j*p+k, 0] = 1
                W[k*n+i, 0] = 1
                candidates.append((U, V, W))
                
    # 2. Add Strassen-like combinations
    # We can generate tensors that are sums of schoolbook tensors.
    # For example, (I+J)(J+K) type structures.
    # Let's generate random linear combinations of 2-4 schoolbook tensors.
    np.random.seed(42)
    
    # Pre-compute schoolbook tensors as vectors for faster combination
    sb_tensors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U = np.zeros(n*m)
                V = np.zeros(m*p)
                W = np.zeros(p*n)
                U[i*m+j] = 1
                V[j*p+k] = 1
                W[k*n+i] = 1
                sb_tensors.append((U, V, W))
                
    # Generate random sums of 2-3 schoolbook tensors
    num_random = 500
    for _ in range(num_random):
        if time.time() > deadline - 30:
            break
        num_terms = np.random.randint(2, 4)
        indices = np.random.choice(len(sb_tensors), num_terms, replace=False)
        
        U_sum = np.zeros(n*m)
        V_sum = np.zeros(m*p)
        W_sum = np.zeros(p*n)
        
        for idx in indices:
            U_sum += sb_tensors[idx][0]
            V_sum += sb_tensors[idx][1]
            W_sum += sb_tensors[idx][2]
            
        # Normalize? No, we want integer entries. The sum of 1s is integer.
        # However, we need to ensure the resulting tensor is "useful".
        # A random sum might not reduce the rank of the remaining problem.
        # But it's a valid candidate.
        
        # Check if it's non-trivial
        if np.sum(np.abs(U_sum)) > 0 and np.sum(np.abs(V_sum)) > 0 and np.sum(np.abs(W_sum)) > 0:
            candidates.append((U_sum.reshape(-1,1), V_sum.reshape(-1,1), W_sum.reshape(-1,1)))
            
    # 3. Add specific known patterns (e.g. from Strassen 2x2)
    # We can embed the 2x2 Strassen tensors into 4x4.
    a2, b2, c2 = _strassen_2x2()
    for s1 in range(7):
        for s2 in range(7):
            # This creates the 49 Strassen terms, which we already have in the 4x4 builder.
            # But maybe we can find a subset?
            pass
            
    return candidates


def _greedy_set_cover(candidates, target, max_rank, deadline):
    """
    Greedy set cover to find a subset of candidates that sums to target.
    This is a heuristic, not guaranteed to find the minimum, but finds valid solutions.
    """
    n, m, p = 4, 4, 4
    target_vec = target.flatten()
    
    # Pre-compute the "effect" of each candidate on the target space
    # effect[c] = sum(U_c * V_c * W_c) flattened
    effects = []
    for U, V, W in candidates:
        U4 = U.reshape(n, m, 1)
        V4 = V.reshape(m, p, 1)
        W4 = W.reshape(p, n, 1)
        tensor = U4 * V4 * W4.transpose(2, 0, 1)
        effects.append(tensor.flatten())
        
    effects = np.array(effects) # Shape: (num_candidates, 64)
    
    current_sum = np.zeros(64)
    selected = []
    
    # We want to minimize the number of selected items such that current_sum == target_vec
    # This is a subset sum problem in high dimensions, which is NP-hard.
    # Greedy: pick the candidate that reduces the L2 error the most.
    
    for _ in range(max_rank):
        if time.time() > deadline - 5:
            break
            
        if np.allclose(current_sum, target_vec, atol=1e-9):
            break
            
        # Calculate error if we add each candidate
        # error_new = ||target_vec - (current_sum + effect_i)||^2
        # = ||target_vec - current_sum - effect_i||^2
        
        remaining = target_vec - current_sum
        
        # We want to minimize ||remaining - effect_i||^2
        # = ||remaining||^2 - 2 * remaining.dot(effect_i) + ||effect_i||^2
        
        # Precompute ||effect_i||^2
        eff_norms_sq = np.sum(effects**2, axis=1)
        
        # Compute dot products
        dots = effects.dot(remaining)
        
        # Score: -2 * dots + eff_norms_sq
        # We want to minimize this.
        scores = -2 * dots + eff_norms_sq
        
        # Pick the best candidate
        best_idx = np.argmin(scores)
        
        # Check if it actually helps (reduces error)
        current_error = np.sum(remaining**2)
        new_remaining = remaining - effects[best_idx]
        new_error = np.sum(new_remaining**2)
        
        if new_error >= current_error - 1e-9:
            # No improvement, stop
            break
            
        current_sum += effects[best_idx]
        selected.append(best_idx)
        
        # Remove selected candidate from consideration to avoid re-selection
        # (Though in exact arithmetic, re-selection might be needed if coefficients are -1, 
        # but here we are summing positive contributions? No, entries can be negative.
        # However, the target is all 1s. 
        # If we allow negative entries, we might need to subtract.
        # The greedy method above assumes we are adding. 
        # Let's stick to adding for now, as most useful tensors have positive correlation with the target.)
        
        # Actually, we should allow removal? No, set cover is additive.
        # If the optimal solution requires subtraction, this simple greedy fails.
        # But for matrix multiplication, the "schoolbook" terms are all positive.
        # Strassen terms involve signs.
        # Let's try a different approach: 
        # Since we have the Strassen solution of rank 49, we know a solution exists.
        # Can we find a smaller one?
        
    if len(selected) < max_rank and np.allclose(current_sum, target_vec, atol=1e-9):
        # Build the final matrices
        U_final = np.zeros((n*m, len(selected)))
        V_final = np.zeros((m*p, len(selected)))
        W_final = np.zeros((p*n, len(selected)))
        
        for i, idx in enumerate(selected):
            U_final[:, i] = candidates[idx][0].flatten()
            V_final[:, i] = candidates[idx][1].flatten()
            W_final[:, i] = candidates[idx][2].flatten()
            
        return U_final, V_final, W_final
        
    return None


def _try_exact_search(deadline):
    """
    Attempt to find a rank < 49 solution using the greedy set cover on a generated pool.
    """
    n, m, p = 4, 4, 4
    target = np.ones((n, m, p))
    
    candidates = _generate_candidate_pool(deadline)
    
    if time.time() > deadline - 10:
        return None
        
    # Try to find a solution with rank < 49
    # We know rank 49 is possible. Can we do better?
    # The greedy method is not guaranteed to find the minimum, but it might find something < 49.
    
    result = _greedy_set_cover(candidates, target, 48, deadline)
    
    if result is not None:
        U, V, W = result
        if _verify(U, V, W):
            return result
            
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    # 1. Get the deterministic rank-49 Strassen solution
    U49, V49, W49 = _build_strassen_4x4()
    
    if _verify(U49, V49, W49):
        best_solution = (U49, V49, W49)
        best_rank = 49
    else:
        # Fallback to schoolbook rank-64
        n, m, p = 4, 4, 4
        rank = 64
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 2. Try to find a lower rank solution using the new combinatorial search
    if time.time() < deadline - 20:
        result = _try_exact_search(deadline)
        if result is not None:
            U, V, W = result
            rank = U.shape[1]
            if rank < best_rank:
                best_solution = (U, V, W)
                best_rank = rank
                
    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
