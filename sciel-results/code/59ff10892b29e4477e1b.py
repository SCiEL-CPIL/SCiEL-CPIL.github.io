"""Matrix multiplication 2x3 x 3x7 - rank minimization search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]
DEADLINE = 140.0  # safety margin before 150s kill

def _validate(U, V, W, n=2, m=3, p=7):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor T[i,j,k] = delta
    # Check: for all (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # and for all other index combos, the sum == 0
    # Efficient check: build matrices
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # For each (i,j,k), compute sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # This should equal 1 if it's a "valid" multiplication entry
    # Actually, the tensor for matrix mult (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition: (A*B)[i,k] = sum_c (sum_j U[i,j,c]*A[i,j]) * (sum_j V[j,k,c]*B[j,k]) * W[k,i,c]
    # Wait, let me re-read the convention.
    # W[k*n+i, r] - so W is indexed by (k, i) where k is row of B output, i is row of A
    # Standard: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor form: T_{i,j,k} with T_{i,j,k} = 1 for all valid (i,j,k) in the product
    # The rank-r decomposition: T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
    # where U has shape (n*m, rank) indexed as U[i*m+j, c]
    # V has shape (m*p, rank) indexed as V[j*p+k, c]
    # W has shape (p*n, rank) indexed as W[k*n+i, c]
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And that's it - the tensor T_{i,j,k} = 1 for all valid indices.
    # Wait no, that's not right either. The matrix multiplication tensor has T_{i,j,k} = 1
    # only when the indices are "compatible" in the multiplication. Actually for standard
    # matrix multiplication, T_{i,j,k} = 1 for ALL (i,j,k) because (AB)_{i,k} = sum_j A_{i,j}B_{j,k}.
    # The tensor is T[i,j,k] = 1 for all i,j,k. That's the standard schoolbook.
    # Hmm, but that means we need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
    
    # Let me verify with schoolbook: U[i*m+j, c] = 1 when c corresponds to (i,j,k'), V[j*p+k, c] = 1 when c corresponds to (i',j,k), W[k'*n+i', c] = 1 when c corresponds to (i,j,k).
    # For c = (i,j,k): U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1. Product = 1. Sum over all c gives 1 (only one c matches). Correct.
    
    # So the condition is: for ALL i in range(n), j in range(m), k in range(p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    # Efficient computation using einsum or matrix operations
    # For fixed j: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Let's compute for all (i,k) pairs for a fixed j:
    #   result[i,k] = sum_c (U[:,j,:] * W[:,:,i] ... ) hmm this is getting complex.
    
    # Simpler: just iterate since n*m*p = 42 is small
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-10

def _try_rank(rank, n=2, m=3, p=7, max_iters=200, seed=0):
    """Try to find a valid rank-r decomposition via random init + local search."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    U = np_rng.randn(n*m, rank)
    V = np_rng.randn(m*p, rank)
    W = np_rng.randn(p*n, rank)
    
    best_err = float('inf')
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        # Compute errors for all (i,j,k)
        errors = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    errors[i,j,k] = np.sum(U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:]) - 1.0
        
        max_err = np.max(np.abs(errors))
        if max_err < 1e-10:
            return U.tolist(), V.tolist(), W.tolist()
        
        if max_err < best_err:
            best_err = max_err
        
        # Gradient-free local search: perturb one random element
        # Choose which matrix and which element to perturb
        mat = rng.choice(['U', 'V', 'W'])
        if mat == 'U':
            row = rng.randint(0, n*m-1)
        elif mat == 'V':
            row = rng.randint(0, m*p-1)
        else:
            row = rng.randint(0, p*n-1)
        col = rng.randint(0, rank-1)
        
        # Try small perturbations
        best_perturb = None
        best_perturb_err = max_err
        for delta in [0.1, -0.1, 0.01, -0.01, 0.5, -0.5]:
            if mat == 'U':
                orig = U[row, col]
                U[row, col] = orig + delta
            elif mat == 'V':
                orig = V[row, col]
                V[row, col] = orig + delta
            else:
                orig = W[row, col]
                W[row, col] = orig + delta
            
            # Quick error check
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:])
                        e = abs(s - 1.0)
                        if e > err:
                            err = e
                        if err > best_perturb_err:
                            break
                    if err > best_perturb_err:
                        break
                if err > best_perturb_err:
                    break
            
            if mat == 'U':
                U[row, col] = orig
            elif mat == 'V':
                V[row, col] = orig
            else:
                W[row, col] = orig
            
            if err < best_perturb_err:
                best_perturb_err = err
                best_perturb = (mat, row, col, delta)
        
        if best_perturb is not None:
            mat, row, col, delta = best_perturb
            if mat == 'U':
                U[row, col] += delta
            elif mat == 'V':
                V[row, col] += delta
            else:
                W[row, col] += delta
        else:
            # Stuck, try a bigger random jump
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                U += np_rng.randn(n*m, rank) * 0.5
            elif mat == 'V':
                V += np_rng.randn(m*p, rank) * 0.5
            else:
                W += np_rng.randn(p*n, rank) * 0.5
    
    if best_err < 1e-10:
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _try_structured_rank(rank, n=2, m=3, p=7):
    """Try structured decompositions based on known patterns."""
    # Try rank = n*p + m*p - n - m + 1 type constructions
    # For 2x3x7, let's try various structured approaches
    
    # Approach: use the fact that we can share some factors
    # Try: U[i,j,c] depends on i and c in a structured way
    pass
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Start from schoolbook and try to reduce
    # First, try to find any valid decomposition at lower ranks
    
    # Try ranks from just below schoolbook down
    # But first, let's try some known lower bounds
    # For matrix multiplication, the tensor rank of (n x m) * (m x p) is at least max(n*p, n*m, m*p)
    # For 2x3x7: max(14, 6, 21) = 21
    
    # Try random search at various ranks
    # Start with a few tries at rank 21, then 22, etc.
    
    best_result = None
    best_rank = schoolbook_rank
    
    # First, generate schoolbook as fallback
    U_school = [[0]*schoolbook_rank for _ in range(n*m)]
    V_school = [[0]*schoolbook_rank for _ in range(m*p)]
    W_school = [[0]*schoolbook_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Try ranks from 21 upward (theoretical lower bound)
    # Use limited time for each rank attempt
    
    ranks_to_try = list(range(21, min(35, schoolbook_rank)))  # Try 21 to 34
    # Also try some higher ranks if time permits
    ranks_to_try += list(range(35, schoolbook_rank))
    
    for target_rank in ranks_to_try:
        if time.time() > DEADLINE - 10:
            break
        
        # Try a few seeds for this rank
        max_attempts = 3 if target_rank <= 25 else 2
        for attempt in range(max_attempts):
            if time.time() > DEADLINE - 10:
                break
            
            result = _try_rank(target_rank, n, m, p, max_iters=100, seed=attempt*1000+42)
            if result is not None:
                U, V, W = result
                # Verify
                if _validate(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # If we haven't found anything better, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
