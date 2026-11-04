"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x9 matrix.
    
    The matrix multiplication tensor T for A(n×m) × B(m×p) = C(n×p) is defined as:
    T[a, b, c] where a indexes (i,j) in A, b indexes (j',k) in B, c indexes (k',i') in C
    T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) [i.e., j=j' and k=k' and i=i']
    
    The schoolbook decomposition uses rank n*m*p with one column per (i,j,k) triple.
    
    We use structured approaches to potentially find lower rank:
    1. Block decomposition exploiting rectangular structure
    2. Local search with exact verification
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    schoolbook_rank = n * m * p  # 144
    
    def verify(U, V, W):
        """Verify that the decomposition is exact for the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Check all (i,j,k) triples where the tensor entry is 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i*m+j, j*p+k, k*n+i] should equal 1
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_full(U, V, W):
        """Full verification: check that non-target entries are 0."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Build set of target (a,b,c) indices
        targets = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    targets.add((i*m+j, j*p+k, k*n+i))
        
        # Check all entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    expected = 1.0 if (a, b, c) in targets else 0.0
                    s = 0.0
                    for cc in range(rank):
                        s += U_arr[a, cc] * V_arr[b, cc] * W_arr[c, cc]
                    if abs(s - expected) > 1e-9:
                        return False
        return True
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition using structured search
    # For 4x4 * 4x9, we can try block-based approaches
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Approach: Try to use rank slightly less than schoolbook by finding
    # columns that can serve multiple (i,j,k) triples
    
    # For the matrix multiplication tensor, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    # and = 0 for all other (a,b,c)
    
    # Let's try a greedy approach: start with schoolbook and try to merge columns
    
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    col_map = {}  # (i,j,k) -> column index
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col_map[(i, j, k)] = col
                col += 1
    
    # Try local optimization: perturb columns and see if we can reduce effective rank
    # This is a heuristic approach
    
    def try_reduce_rank(U_init, V_init, W_init, target_rank, num_trials=100):
        """Try to reduce rank by finding a lower-rank approximation that is exact."""
        n, m, p = DIMENSIONS
        current_rank = len(U_init[0])
        
        if target_rank >= current_rank:
            return None
        
        U_arr = np.array(U_init, dtype=np.float64)
        V_arr = np.array(V_init, dtype=np.float64)
        W_arr = np.array(W_init, dtype=np.float64)
        
        for trial in range(num_trials):
            if time.time() > deadline - 15:
                break
            
            # Random subset of columns
            cols_to_keep = list(range(current_rank))
            random.shuffle(cols_to_keep)
            kept_cols = cols_to_keep[:target_rank]
            
            # Try to find new values for kept columns that satisfy constraints
            # This is a system of polynomial equations - use iterative refinement
            
            U_new = U_arr[:, kept_cols].copy()
            V_new = V_arr[:, kept_cols].copy()
            W_new = W_arr[:, kept_cols].copy()
            
            # Simple check: does this subset already work?
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U_new[i*m+j, c] * V_new[j*p+k, c] * W_new[k*n+i, c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Also check non-target entries are ~0
                full_valid = True
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            is_target = False
                            # Check if (a,b,c) corresponds to a valid (i,j,k)
                            i = a // m
                            j = a % m
                            j2 = b // p
                            k = b % p
                            k2 = c // n
                            i2 = c % n
                            if j == j2 and k == k2 and i == i2:
                                is_target = True
                            
                            expected = 1.0 if is_target else 0.0
                            s = 0.0
                            for cc in range(target_rank):
                                s += U_new[a, cc] * V_new[b, cc] * W_new[c, cc]
                            if abs(s - expected) > 1e-9:
                                full_valid = False
                                break
                        if not full_valid:
                            break
                    if not full_valid:
                        break
                
                if full_valid:
                    U_list = [[float(x) for x in row] for row in U_new]
                    V_list = [[float(x) for x in row] for row in V_new]
                    W_list = [[float(x) for x in row] for row in W_new]
                    return U_list, V_list, W_list
        
        return None
    
    # Try various target ranks
    for target_rank in range(100, schoolbook_rank):
        if time.time() > deadline - 30:
            break
        
        result = try_reduce_rank(U, V, W, target_rank, num_trials=20)
        if result is not None:
            U_res, V_res, W_res = result
            if verify(U_res, V_res, W_res):
                best_result = (U_res, V_res, W_res, target_rank)
                break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
