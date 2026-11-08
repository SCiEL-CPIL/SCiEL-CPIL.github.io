import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]
TIME_LIMIT = 140  # safety margin from 150s

def _verify(U, V, W):
    """Check if U,V,W exactly decompose the 3x4x5 matmul tensor."""
    n, m, p = 3, 4, 5
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # T = einsum('ur,vr,wr->uvw', U, V, W)
    T = np.einsum('ur,vr,wr->uvw', U, V, W)
    
    # Expected: T[i*m+l, l*p+k, k*n+i] = 1, else 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_idx = i * m + l
                v_idx = l * p + k
                w_idx = k * n + i
                if abs(T[u_idx, v_idx, w_idx] - 1.0) > 1e-9:
                    return False
    
    # Check that T is zero elsewhere
    mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_idx = i * m + l
                v_idx = l * p + k
                w_idx = k * n + i
                mask[u_idx, v_idx, w_idx] = True
    
    invalid = ~mask
    if np.any(np.abs(T[invalid]) > 1e-9):
        return False
    
    return True

def _seed_solution():
    """Return the seed 56-rank solution as U,V,W numpy arrays."""
    n, m, p = 3, 4, 5
    U_parts = []
    V_parts = []
    W_parts = []
    
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W
    
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            U_parts.append(np.zeros(n * m))
            for i, idx in enumerate(u_idx):
                U_parts[-1][idx] = U_blk[i, r]
            V_parts.append(np.zeros(m * p))
            for i, idx in enumerate(v_idx):
                V_parts[-1][idx] = V_blk[i, r]
            W_parts.append(np.zeros(p * n))
            for i, idx in enumerate(w_idx):
                W_parts[-1][idx] = W_blk[i, r]
    
    # 12 blocks as in seed
    add_block([0,1,4,5], [0,1,5,6], [0,3,1,4], *strassen_2x2x2())
    add_block([0,1,4,5], [2,3,7,8], [6,9,7,10], *strassen_2x2x2())
    add_block([0,1,4,5], [4,9], [12,13], *schoolbook_2x2x1())
    add_block([2,3,6,7], [10,11,15,16], [0,3,1,4], *strassen_2x2x2())
    add_block([2,3,6,7], [12,13,17,18], [6,9,7,10], *strassen_2x2x2())
    add_block([2,3,6,7], [14,19], [12,13], *schoolbook_2x2x1())
    add_block([8,9], [0,1,5,6], [2,5], *schoolbook_1x2x2())
    add_block([8,9], [2,3,7,8], [8,11], *schoolbook_1x2x2())
    add_block([8,9], [4,9], [14], *schoolbook_1x2x1())
    add_block([10,11], [10,11,15,16], [2,5], *schoolbook_1x2x2())
    add_block([10,11], [12,13,17,18], [8,11], *schoolbook_1x2x2())
    add_block([10,11], [14,19], [14], *schoolbook_1x2x1())
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W

def _try_remove_columns(U, V, W, max_tries=100):
    """Try to remove one column (rank term) by checking if it's linearly dependent."""
    n, m, p = 3, 4, 5
    rank = U.shape[1]
    if rank <= 1:
        return None
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Try removing each column and see if the tensor still works
    for r in range(rank):
        U_test = np.delete(U, r, axis=1)
        V_test = np.delete(V, r, axis=1)
        W_test = np.delete(W, r, axis=1)
        if _verify(U_test, V_test, W_test):
            return U_test, V_test, W_test
    return None

def _try_linear_combination_removal(U, V, W):
    """Try to replace two columns with one linear combination."""
    n, m, p = 3, 4, 5
    rank = U.shape[1]
    if rank <= 2:
        return None
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Try all pairs of columns
    for r1 in range(rank):
        for r2 in range(r1 + 1, rank):
            # Try replacing columns r1 and r2 with a single column
            # that is a linear combination
            for alpha in [-2, -1, -0.5, 0.5, 1, 2]:
                for beta in [-2, -1, -0.5, 0.5, 1, 2]:
                    # Create new column as alpha * col_r1 + beta * col_r2
                    U_new = np.delete(U, [r1, r2], axis=1)
                    V_new = np.delete(V, [r1, r2], axis=1)
                    W_new = np.delete(W, [r1, r2], axis=1)
                    
                    # Add the combined column
                    U_combined = alpha * U[:, r1] + beta * U[:, r2]
                    V_combined = alpha * V[:, r1] + beta * V[:, r2]
                    W_combined = alpha * W[:, r1] + beta * W[:, r2]
                    
                    U_test = np.column_stack([U_new, U_combined])
                    V_test = np.column_stack([V_new, V_combined])
                    W_test = np.column_stack([W_new, W_combined])
                    
                    if _verify(U_test, V_test, W_test):
                        return U_test, V_test, W_test
    return None

def _random_perturbation_search(U, V, W, start_time, time_limit, max_iter=1000):
    """Try random perturbations to find equivalent decompositions with lower rank."""
    n, m, p = 3, 4, 5
    best = (U.copy(), V.copy(), W.copy())
    best_rank = U.shape[1]
    
    for iteration in range(max_iter):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        U_c, V_c, W_c = best[0].copy(), best[1].copy(), best[2].copy()
        rank = U_c.shape[1]
        
        # Try to reduce rank by 1 through random linear combinations
        if rank > 1:
            # Pick two random columns to combine
            r1, r2 = random.sample(range(rank), 2)
            alpha = random.uniform(-2, 2)
            beta = random.uniform(-2, 2)
            
            # Remove one of the columns and add the combination
            U_new = np.delete(U_c, r2, axis=1)
            V_new = np.delete(V_c, r2, axis=1)
            W_new = np.delete(W_c, r2, axis=1)
            
            U_combined = alpha * U_c[:, r1] + beta * U_c[:, r2]
            V_combined = alpha * V_c[:, r1] + beta * V_c[:, r2]
            W_combined = alpha * W_c[:, r1] + beta * W_c[:, r2]
            
            U_test = np.column_stack([U_new, U_combined])
            V_test = np.column_stack([V_new, V_combined])
            W_test = np.column_stack([W_new, W_combined])
            
            if _verify(U_test, V_test, W_test):
                if U_test.shape[1] < best_rank:
                    best = (U_test, V_test, W_test)
                    best_rank = U_test.shape[1]
    
    return best

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start from seed
    U, V, W = _seed_solution()
    best = (U.copy(), V.copy(), W.copy())
    best_rank = U.shape[1]
    
    # Verify seed
    if not _verify(best[0], best[1], best[2]):
        raise ValueError("Seed solution failed verification")
    
    # Try iterative column removal
    for iteration in range(100):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 20:
            break
        
        U_c, V_c, W_c = best[0].copy(), best[1].copy(), best[2].copy()
        result = _try_remove_columns(U_c, V_c, W_c)
        if result is not None:
            U_c, V_c, W_c = result
            if U_c.shape[1] < best_rank:
                best = (U_c, V_c, W_c)
                best_rank = U_c.shape[1]
        
        # If no column could be removed, break
        if result is None:
            break
    
    # Try linear combination removal
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 10:
        U_c, V_c, W_c = best[0].copy(), best[1].copy(), best[2].copy()
        result = _try_linear_combination_removal(U_c, V_c, W_c)
        if result is not None:
            if result[0].shape[1] < best_rank:
                best = result
                best_rank = result[0].shape[1]
    
    # Try random perturbation search with remaining time
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 5:
        time_budget = TIME_LIMIT - 5 - elapsed
        best = _random_perturbation_search(best[0], best[1], best[2], start_time, time_limit, max_iter=500)
        best_rank = best[0].shape[1]
    
    return {
        "U": best[0].tolist(),
        "V": best[1].tolist(),
        "W": best[2].tolist()
    }
