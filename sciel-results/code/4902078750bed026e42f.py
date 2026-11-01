"""Matrix multiplication 2x3x4: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 24
    
    # Schoolbook solution as fallback
    def schoolbook():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # The known minimal rank for 2x3x4 matrix multiplication is 14.
    # We attempt to find an exact decomposition of rank 14.
    # 
    # Strategy: Use the structure that many known decompositions of small matrix
    # multiplications have entries in {0, 1/2, 1, -1/2, -1} or similar small sets.
    # 
    # We'll use a structured search: try to find U, V, W with entries from a small
    # set of candidate values, using constraint satisfaction / local search.
    #
    # The tensor T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term,
    # else 0. The decomposition requires:
    #   sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = T[i*m+j, j*p+k, k*n+i]
    #
    # Let's index:
    #   U has shape (n*m, r) = (6, r)
    #   V has shape (m*p, r) = (12, r)  
    #   W has shape (p*n, r) = (8, r)
    #   T[a, b, c] where a in [0,6), b in [0,12), c in [0,8)
    #   T[a,b,c] = 1 if the triple (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i is valid
    #
    # For 2x3x4: i in {0,1}, j in {0,1,2}, k in {0,1,2,3}
    # a = i*3+j, b = j*4+k, c = k*2+i
    # T is nonzero only for these 24 valid triples.
    
    # Build the target tensor
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1
    
    num_target = len(target)  # 24
    total_entries = n * m * m * p * p * n  # 2*3*3*4*4*2 = 576
    num_zeros = total_entries - num_target  # 552
    
    # Candidate values for entries
    candidates = [0, 1, -1, 0.5, -0.5, 1.5, -1.5, 2, -2]
    
    best_result = schoolbook()
    
    # Try rank 14 first, then 15, 16...
    for rank in range(14, min(24, rank_schoolbook)):
        if time.time() > deadline:
            break
            
        # For each rank, try multiple random restarts with local search
        max_restarts = 20
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                found = True
                break
                
            # Initialize with small random values from candidate set
            import random
            random.seed(restart * 1000 + rank)
            
            U = [[random.choice(candidates) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(candidates) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice(candidates) for _ in range(rank)] for _ in range(p * n)]
            
            # Compute current error
            def compute_error(U, V, W, rank):
                err = 0.0
                # Check target entries
                for (a, b, c) in target:
                    s = 0.0
                    for t in range(rank):
                        s += U[a][t] * V[b][t] * W[c][t]
                    err += (s - 1.0) ** 2
                # Check zero entries (sample some for speed)
                import random as r2
                r2.seed(42)
                samples = r2.sample(range(total_entries), min(200, num_zeros))
                for idx in samples:
                    # Decode index to (a,b,c)
                    c = idx % (p * n)
                    tmp = idx // (p * n)
                    b = tmp % (m * p)
                    a = tmp // (m * p)
                    if (a, b, c) not in target:
                        s = 0.0
                        for t in range(rank):
                            s += U[a][t] * V[b][t] * W[c][t]
                        err += s * s
                return err
            
            def verify(U, V, W, rank):
                """Check if decomposition is exact."""
                # Check all target entries
                for (a, b, c) in target:
                    s = 0.0
                    for t in range(rank):
                        s += U[a][t] * V[b][t] * W[c][t]
                    if abs(s - 1.0) > 1e-9:
                        return False
                # Check all zero entries
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                        # This is target, already checked
                # Check non-target entries
                for a in range(n * m):
                    for b in range(m * p):
                        for c in range(p * n):
                            if (a, b, c) not in target:
                                s = 0.0
                                for t in range(rank):
                                    s += U[a][t] * V[b][t] * W[c][t]
                                if abs(s) > 1e-9:
                                    return False
                return True
            
            cur_err = compute_error(U, V, W, rank)
            
            # Local search: iteratively modify one entry at a time to reduce error
            max_iters = 500
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                if verify(U, V, W, rank):
                    found = True
                    best_result = {"U": [row[:] for row in U], "V": [row[:] for row in V], "W": [row[:] for row in W]}
                    return best_result
                    
                best_delta = 0
                best_change = None
                best_err = cur_err
                
                # Try changing one entry in U
                for a in range(n * m):
                    for t in range(rank):
                        old_val = U[a][t]
                        for new_val in candidates:
                            if new_val == old_val:
                                continue
                            U[a][t] = new_val
                            new_err = compute_error(U, V, W, rank)
                            if new_err < best_err:
                                best_err = new_err
                                best_delta = new_err - cur_err
                                best_change = ('U', a, t, new_val)
                            U[a][t] = old_val
                        U[a][t] = old_val
                    
                # Try changing one entry in V
                for b in range(m * p):
                    for t in range(rank):
                        old_val = V[b][t]
                        for new_val in candidates:
                            if new_val == old_val:
                                continue
                            V[b][t] = new_val
                            new_err = compute_error(U, V, W, rank)
                            if new_err < best_err:
                                best_err = new_err
                                best_delta = new_err - cur_err
                                best_change = ('V', b, t, new_val)
                            V[b][t] = old_val
                        V[b][t] = old_val
                        
                # Try changing one entry in W
                for c in range(p * n):
                    for t in range(rank):
                        old_val = W[c][t]
                        for new_val in candidates:
                            if new_val == old_val:
                                continue
                            W[c][t] = new_val
                            new_err = compute_error(U, V, W, rank)
                            if new_err < best_err:
                                best_err = new_err
                                best_delta = new_err - cur_err
                                best_change = ('W', c, t, new_val)
                            W[c][t] = old_val
                        W[c][t] = old_val
                        
                if best_change is None:
                    # Stuck, try a random perturbation
                    break
                    
                mat, idx1, idx2, new_val = best_change
                if mat == 'U':
                    U[idx1][idx2] = new_val
                elif mat == 'V':
                    V[idx1][idx2] = new_val
                else:
                    W[idx1][idx2] = new_val
                cur_err = best_err
            
            if verify(U, V, W, rank):
                found = True
                best_result = {"U": [row[:] for row in U], "V": [row[:] for row in V], "W": [row[:] for row in W]}
                return best_result
        
        if found:
            return best_result
    
    # If we couldn't find a better rank, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
