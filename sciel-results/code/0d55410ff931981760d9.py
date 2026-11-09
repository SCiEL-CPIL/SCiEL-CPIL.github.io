import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 3, 4
    
    # The MM tensor for multiplying (n x m) by (m x p) to get (n x p):
    # T_{(i,j),(j',k'),(i',k')} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # where U index = i*m+j, V index = j'*p+k', W index = k*n+i
    # 
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # and = 0 for all other (u,v,w) triples.
    #
    # Strategy: Use a structured decomposition.
    # For 3x3x3, Laderman gives rank 23.
    # For 3x3x4, we can use a block structure.
    #
    # Let's try to construct a rank-R decomposition directly.
    # 
    # Key idea: The tensor can be viewed as a sum of rank-1 tensors.
    # Each rank-1 term corresponds to a "path" through the multiplication.
    #
    # For the rectangular case, one approach:
    # C = A * B where A is 3x3, B is 3x4.
    # We can think of this as 4 separate 3x3 multiplications (one for each column of B)
    # but that gives rank 4*23 = 92 which is terrible.
    #
    # Better: Use the fact that the tensor is separable in certain ways.
    #
    # Let me try a direct construction approach:
    # We want to find U (9 x R), V (12 x R), W (12 x R) such that
    # the decomposition holds.
    #
    # Approach: Start with a known decomposition for 3x3x3 (rank 23) and extend.
    # Or, use a more systematic algebraic approach.
    #
    # Let's try a different approach: use the structure of the problem.
    # The MM tensor T satisfies:
    # T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}
    # where e_{i,j} is the basis vector for position (i,j) in the flattened matrix.
    #
    # Actually, let me just try to find a valid decomposition by solving
    # a system of equations using a structured initialization.
    #
    # For rank R, we have R columns in U, V, W.
    # The constraints are:
    # 1. For each (i,j,k) in valid set: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # 2. For each (u,v,w) not in valid set: sum_r U[u,r]*V[v,r]*W[w,r] = 0
    #
    # This is a system of polynomial equations. Let's use a greedy/structured approach.
    #
    # Let me try a construction based on the following observation:
    # If we can find vectors u_r, v_r, w_r such that the outer product
    # u_r \otimes v_r \otimes w_r approximates parts of the tensor, we can sum them.
    #
    # For now, let's try a more practical approach:
    # Use the schoolbook decomposition (rank 36) as a starting point,
    # then try to reduce the rank by finding linear dependencies or
    # using a more clever construction.
    #
    # Actually, let me try a completely different approach:
    # Use the fact that we can represent the multiplication using
    # a smaller number of "intermediate" products.
    #
    # For 3x3x4, let's try rank 27 (which is 3*3*3, the 3x3x3 case,
    # extended to handle the extra column).
    #
    # Let me try a direct construction with rank R for R from 23 to 36.
    
    # First, let's define the valid and invalid index sets
    valid_set = set()
    valid_list = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_set.add((u, v, w))
                valid_list.append((u, v, w))
    
    all_u = list(range(n * m))
    all_v = list(range(m * p))
    all_w = list(range(p * n))
    
    # For efficiency, precompute the invalid indices
    # But there are 9*12*12 = 1296 total, and 36 valid, so 1260 invalid.
    # We'll check a subset.
    
    import random
    random.seed(42)
    
    # Try to find a decomposition using a structured approach.
    # We'll use a "basis" approach: try to build the decomposition
    # column by column.
    
    # Actually, let me try a more promising approach:
    # Use the known Laderman decomposition for 3x3x3 and extend it.
    #
    # Laderman's rank-23 decomposition for 3x3x3:
    # This is complex to encode. Let me instead try a different strategy.
    #
    # Strategy: Use a "block" decomposition.
    # Write A and B in blocks and use a low-rank decomposition of the block matrix.
    #
    # For 3x3 * 3x4:
    # A = [A11 A12 A13]  (3x3)
    # B = [B11 B12 B13 B14]  (3x4)
    #
    # C = A * B = [A11*B11+A12*B21+A13*B31, ...]
    #
    # This doesn't immediately help with rank reduction.
    #
    # Let me try a more direct approach: use a randomized algebraic method.
    # Initialize U, V, W with random values from a small set,
    # then use a local search that specifically targets the constraint equations.
    #
    # Key improvement over the current approach: use a smarter initialization
    # and a more efficient error computation.
    
    def compute_error(U, V, W, valid_list, invalid_sample):
        """Compute the total error of the decomposition."""
        err = 0.0
        # Check valid constraints
        for (u, v, w) in valid_list:
            val = np.dot(U[u] * V[v], W[w])
            err += (val - 1.0) ** 2
        # Check invalid constraints (sample)
        for (u, v, w) in invalid_sample:
            val = np.dot(U[u] * V[v], W[w])
            err += val ** 2
        return err
    
    # Precompute a fixed sample of invalid indices for consistent evaluation
    all_invalid = []
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                if (u, v, w) not in valid_set:
                    all_invalid.append((u, v, w))
    
    # Use a fixed sample of 200 invalid indices
    invalid_sample = random.sample(all_invalid, min(200, len(all_invalid)))
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 23 to 36
    for R in range(23, best_rank + 1):
        if time.time() > deadline:
            break
        
        found = False
        
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Initialize with small random integers
            U = np.random.randint(-1, 2, (n * m, R))
            V = np.random.randint(-1, 2, (m * p, R))
            W = np.random.randint(-1, 2, (p * n, R))
            
            # Local search with restarts
            current_err = compute_error(U, V, W, valid_list, invalid_sample)
            
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                # Pick a random matrix and element to perturb
                mat_choice = random.random()
                if mat_choice < 0.33:
                    idx, r = random.randint(0, n * m - 1), random.randint(0, R - 1)
                    delta = random.choice([-1, 1])
                    U[idx, r] += delta
                    new_err = compute_error(U, V, W, valid_list, invalid_sample)
                    if new_err >= current_err:
                        U[idx, r] -= delta  # revert
                    else:
                        current_err = new_err
                elif mat_choice < 0.66:
                    idx, r = random.randint(0, m * p - 1), random.randint(0, R - 1)
                    delta = random.choice([-1, 1])
                    V[idx, r] += delta
                    new_err = compute_error(U, V, W, valid_list, invalid_sample)
                    if new_err >= current_err:
                        V[idx, r] -= delta  # revert
                    else:
                        current_err = new_err
                else:
                    idx, r = random.randint(0, p * n - 1), random.randint(0, R - 1)
                    delta = random.choice([-1, 1])
                    W[idx, r] += delta
                    new_err = compute_error(U, V, W, valid_list, invalid_sample)
                    if new_err >= current_err:
                        W[idx, r] -= delta  # revert
                    else:
                        current_err = new_err
                
                # Check for exact solution
                if current_err < 1e-10:
                    # Verify with full check
                    full_err = 0.0
                    for (u, v, w) in valid_list:
                        val = np.dot(U[u] * V[v], W[w])
                        if abs(val - 1.0) > 1e-10:
                            full_err += abs(val - 1.0)
                    for (u, v, w) in all_invalid:
                        val = np.dot(U[u] * V[v], W[w])
                        if abs(val) > 1e-10:
                            full_err += abs(val)
                    if full_err < 1e-10:
                        found = True
                        break
            
            if found:
                best_rank = R
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                break
        
        if found:
            break
    
    # Fallback: Schoolbook decomposition (rank 36)
    if best_U is None:
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_U = U
        best_V = V
        best_W = W
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
