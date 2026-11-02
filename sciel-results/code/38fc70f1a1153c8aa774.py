"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = delta_{j,k} (shared index j matches with k in B)
    # Actually: T[i,j,k] where i is row of A, j is col of A/row of B, k is col of B
    # C[i,k] = sum_j A[i,j] B[j,k], so the tensor has T[i,j,k] = 1 if we identify the free indices
    
    # Check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1
    # when this corresponds to a valid term in matrix multiplication
    # The matrix multiplication tensor: T[i,j,k] = 1 for all valid (i,j,k) since
    # C[i,k] = sum_j A[i,j]*B[j,k] means each triple (i,j,k) contributes A[i,j]*B[j,k]
    
    # Actually the tensor for matrix mult is: T[i,j,k] = 1 for all i,j,k
    # because the multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is defined such that T[i,j,k] * A[i,j] * B[j,k] summed gives C[i,k]
    # So T[i,j,k] = 1 for all valid indices.
    
    # Wait, let me reconsider. The standard formulation:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T has indices (i, j, k) where T[i,j,k] = 1 for all valid i,j,k
    # The decomposition: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # But here W has index k*n+i, so W[k*n+i, c]
    
    # Let me just verify: for each (i,j,k), the sum should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _strassen_3x3_core(n, m, p):
    """
    Use Strassen's 7-term algorithm for the 3x3 core, then extend to 3x3x5.
    
    For 3x3 x 3x3, Strassen gives rank 7.
    For 3x3 x 3x5, we can use rank 7 for the first 3 columns of B,
    plus 3 more terms for the remaining 2 columns... but that's not quite right.
    
    Actually, let me think differently. The 3x3x5 tensor can be seen as:
    C[:, :3] = A * B[:, :3]  (rank 7 via Strassen)
    C[:, 3:] = A * B[:, 3:]  (rank 6 via 3x2 schoolbook, or optimized)
    
    Total: 7 + 6 = 13? No, that's too optimistic because Strassen's 7 terms
    are for the full 3x3 product.
    
    Let me use a different approach: start with schoolbook (rank 45) and try to
    reduce rank by finding linear dependencies among the 45 terms.
    """
    pass

def _try_rank_reduction(U, V, W, n, m, p, max_rank, seed=42):
    """Try to reduce the rank by finding linear combinations of columns."""
    random.seed(seed)
    np.random.seed(seed)
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = U_arr.shape[1]
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = current_rank
    
    # Try to find pairs of columns that can be merged
    # Two columns c1, c2 can be merged if there exist scalars a, b such that
    # the new column (a*U[:,c1] + b*U[:,c2], etc.) still satisfies the tensor equation
    # This is complex. Instead, try random projections.
    
    for target_rank in range(max_rank, current_rank - 1):
        if target_rank >= best_rank:
            continue
        
        # Try random linear combinations to reduce rank
        for attempt in range(100):
            # Randomly select target_rank columns worth of information
            # Create a random transformation
            T = np.random.randn(current_rank, target_rank)
            
            new_U = U_arr @ T
            new_V = V_arr @ T
            new_W = W_arr @ T
            
            # Check if this still decomposes the tensor (approximately)
            # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
            # This is a 3-tensor contraction, hard to check directly
            # Instead, check on a random sample of (i,j,k) tuples
            
            valid = True
            for _ in range(200):
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                s = 0
                for c in range(target_rank):
                    s += new_U[i*m+j, c] * new_V[j*p+k, c] * new_W[k*n+i, c]
                if abs(s - 1) > 1e-6:
                    valid = False
                    break
            
            if valid:
                # Try to round to integers/half-integers
                new_U_r = np.round(new_U * 2) / 2
                new_V_r = np.round(new_V * 2) / 2
                new_W_r = np.round(new_W * 2) / 2
                
                # Verify exactness with rounded values
                exact_valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(target_rank):
                                s += new_U_r[i*m+j, c] * new_V_r[j*p+k, c] * new_W_r[k*n+i, c]
                            if abs(s - 1) > 1e-10:
                                exact_valid = False
                                break
                    if not exact_valid:
                        break
                if not exact_valid:
                    continue
                
                best_U, best_V, best_W = new_U_r, new_V_r, new_W_r
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    if best_rank < current_rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    # Try rank reduction
    # Known lower bound for 3x3x5 is around 43
    # Try to find rank 43 or better
    for target_rank in range(43, best_rank):
        if time.time() > deadline - 30:  # Leave 30s safety margin
            break
        
        result = _try_rank_reduction(
            [[x] for x in best_U] if isinstance(best_U, np.ndarray) else best_U,
            [[x] for x in best_V] if isinstance(best_V, np.ndarray) else best_V,
            [[x] for x in best_W] if isinstance(best_W, np.ndarray) else best_W,
            n, m, p, target_rank,
            seed=random.randint(0, 2**31)
        )
        
        if result is not None:
            new_U, new_V, new_W = result
            if len(new_U[0]) < best_rank:
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = len(new_U[0])
    
    # Final validation
    if time.time() < deadline - 5:
        if not _validate(best_U, best_V, best_W, n, m, p):
            # Fall back to schoolbook
            U, V, W = _schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
