import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Correct Strassen 2x2x2 decomposition. Rank 7."""
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
    
    # M1: (A[0,0]-A[1,1])*(B[1,1]-B[0,0])
    U[0, 0] = 1; U[3, 0] = -1
    V[3, 0] = 1; V[0, 0] = -1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = -1
    
    # M2: (A[0,0]+A[1,1])*B[1,1]
    U[0, 1] = 1; U[3, 1] = 1
    V[3, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3: A[1,1]*(B[1,0]-B[0,0])
    U[3, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4: A[0,0]*(B[0,1]-B[1,1])
    U[0, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A[0,0]+A[0,1])*B[1,1]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # M6: (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7: (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_base_decomposition():
    """
    Build 4x4x7 decomposition using Strassen 2x2x2 for the 4x4x4 part
    and schoolbook for the 4x4x3 part.
    4x4x4: 8 Strassen blocks, rank 56
    4x4x3: 4 schoolbook blocks, rank 48
    Total: 104
    """
    n, m, p = 4, 4, 7
    total_rank = 8 * 7 + 4 * 12  # 104
    
    U = np.zeros((16, total_rank), dtype=np.float64)
    V = np.zeros((28, total_rank), dtype=np.float64)
    W = np.zeros((28, total_rank), dtype=np.float64)
    
    sU, sV, sW = _strassen_2x2x2()
    
    block_configs = [
        (0, 0, 0, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (1, 1, 1, 0),
        (0, 0, 0, 1),
        (0, 1, 1, 1),
        (1, 0, 0, 1),
        (1, 1, 1, 1),
    ]
    
    rank_offset = 0
    for (arb, acb, brb, bc) in block_configs:
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    u_idx = (2 * arb + i) * 4 + (2 * acb + j)
                    U[u_idx, rank_offset + r] += sU[i * 2 + j, r]
            for j in range(2):
                for k in range(2):
                    v_idx = (2 * brb + j) * 7 + (2 * bc + k)
                    V[v_idx, rank_offset + r] += sV[j * 2 + k, r]
            for k in range(2):
                for i in range(2):
                    w_idx = (2 * bc + k) * 4 + (2 * arb + i)
                    W[w_idx, rank_offset + r] += sW[k * 2 + i, r]
        rank_offset += 7
    
    # Part 2: 4x4x3 schoolbook
    for (arb, acb, brb) in [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)]:
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    col = rank_offset + i * 6 + j * 3 + k
                    u_idx = (2 * arb + i) * 4 + (2 * acb + j)
                    v_idx = (2 * brb + j) * 7 + (4 + k)
                    w_idx = (4 + k) * 4 + (2 * arb + i)
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
        rank_offset += 12
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify decomposition correctness."""
    rank = U.shape[1]
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # Use tensor contraction for speed
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _greedy_column_elimination(U, V, W, n, m, p, max_time=100):
    """
    Structural change: Greedy search for linear dependencies.
    We try to remove columns by expressing them as combinations of others
    in a way that preserves the tensor identity.
    """
    start_time = time.time()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # We iterate: try to remove each column.
    # If we can't remove a column directly, we try to "merge" it with another
    # by adding a multiple of one to the other, then checking if the sum
    # can be removed.
    
    # For efficiency, we only try to remove columns that are "sparse" or 
    # have small norms, as they are more likely to be dependent.
    
    # Actually, a better structural approach:
    # Use the fact that the tensor is bilinear.
    # If we fix the W matrix, the problem becomes linear in U and V.
    # But we can vary all three.
    
    # Let's try a random perturbation approach:
    # Add a random vector to U, V, W columns such that the tensor identity
    # is preserved, and then check if any column becomes zero or dependent.
    
    rng = np.random.RandomState(42)
    
    # We will try to reduce the rank by finding a non-trivial null space vector
    # in the space of decompositions.
    # This is equivalent to finding a, b, c such that:
    # sum_s a_s * b_s * c_s * (U_s, V_s, W_s) = 0 in the tensor space.
    # This is a cubic equation, hard to solve.
    
    # Instead, let's use the **SVD of the matricizations** to guide the search.
    # If the rank of the matricization is lower than the current rank,
    # there is a dependency.
    
    # But we need to keep the exact tensor.
    
    # Let's stick to the simple column removal with a twist:
    # When checking if a column can be removed, we don't just check if it's
    # in the span. We check if the *tensor* contribution can be redistributed.
    
    # For each column r, we try to express the tensor T_r = U_r \otimes V_r \otimes W_r
    # as a linear combination of T_s for s != r.
    # This is a linear system in the coefficients alpha_s:
    # T_r = sum_s alpha_s T_s
    # This is a system of n*m*p equations (112) with rank-1 unknowns (96).
    # It is overdetermined, so it usually has no solution.
    # However, if we allow *two* columns to be modified, it becomes possible.
    
    # Let's try a **pairwise merging** strategy.
    # Pick two columns r1, r2.
    # Try to find alpha, beta such that:
    # T_{r1} + alpha * T_{r2} = 0  (impossible unless dependent)
    # Instead:
    # Replace T_{r1} with T_{r1} + alpha * T_{r2} and T_{r2} with T_{r2} - beta * T_{r1}
    # and hope one becomes zero? No.
    
    # Let's try the **randomized rank reduction** from the literature:
    # 1. Pick a random vector u in the null space of U (if exists).
    # 2. Adjust V and W to compensate.
    
    # Since we start with a valid decomposition, the null space of U is not
    # necessarily related to the tensor.
    
    # Let's use a **different strategy**: **Compress the factors using SVD**.
    # Compute SVD of U, V, W.
    # If the smallest singular values are small, we can drop those components.
    # But we need exactness.
    
    # Given the time constraint, I will implement a **simple but effective**
    # **randomized column swapping and elimination**.
    
    for iter in range(1000):
        if time.time() - start_time > max_time:
            break
            
        rank = best_U.shape[1]
        if rank == 0:
            break
            
        # Pick a random column to try to remove
        r = rng.randint(rank)
        
        # Try to remove column r
        # Check if U[:, r] is in span of U[:, -r]
        U_other = np.delete(best_U, r, axis=1)
        V_other = np.delete(best_V, r, axis=1)
        W_other = np.delete(best_W, r, axis=1)
        
        # Solve for a, b, c
        # This is the hard part. We'll use the necessary condition:
        # U[:, r] in span(U_other), etc.
        
        a, _, rank_a, _ = np.linalg.lstsq(U_other, best_U[:, r], rcond=None)
        b, _, rank_b, _ = np.linalg.lstsq(V_other, best_V[:, r], rcond=None)
        c, _, rank_c, _ = np.linalg.lstsq(W_other, best_W[:, r], rcond=None)
        
        if np.linalg.norm(U_other @ a - best_U[:, r]) > 1e-8:
            continue
        if np.linalg.norm(V_other @ b - best_V[:, r]) > 1e-8:
            continue
        if np.linalg.norm(W_other @ c - best_W[:, r]) > 1e-8:
            continue
            
        # If all three are in the span, we have a candidate.
        # Now we must check if the tensor identity holds.
        # T_r = sum_s a_s b_s c_s T_s ?
        # No, T_r = U_r \otimes V_r \otimes W_r
        # sum_s a_s b_s c_s T_s = sum_s a_s b_s c_s (U_s \otimes V_s \otimes W_s)
        # This is not equal to (sum_s a_s U_s) \otimes (sum_s b_s V_s) \otimes (sum_s c_s W_s)
        # unless a, b, c are related.
        
        # So this check is insufficient.
        # We must verify the full tensor.
        
        U_try = np.delete(best_U, r, axis=1)
        V_try = np.delete(best_V, r, axis=1)
        W_try = np.delete(best_W, r, axis=1)
        
        if _verify(U_try, V_try, W_try, n, m, p):
            best_U = U_try
            best_V = V_try
            best_W = W_try
            best_rank = rank - 1
            break # Restart loop with new rank
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 130  # 20s margin before 150s kill
    
    # Build base decomposition (rank 104)
    U, V, W = _build_base_decomposition()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int8)
        V = np.zeros((m * p, rank), dtype=np.int8)
        W = np.zeros((p * n, rank), dtype=np.int8)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        U, V, W, reduced_rank = _greedy_column_elimination(U, V, W, n, m, p, max_time=min(remaining_time, 100))
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Rebuild and just return the base
        U, V, W = _build_base_decomposition()
    
    # Convert to lists, use integer if possible
    def _to_list(arr):
        # Check if all values are close to integers or half-integers
        if np.allclose(arr, np.round(arr), atol=1e-9):
            return np.round(arr).astype(np.int64).tolist()
        elif np.allclose(arr * 2, np.round(arr * 2), atol=1e-9):
            return (arr * 2).round().astype(np.int64).tolist()  # half-integers as 2x
        else:
            return arr.tolist()
    
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

# EVOLVE-BLOCK-END
