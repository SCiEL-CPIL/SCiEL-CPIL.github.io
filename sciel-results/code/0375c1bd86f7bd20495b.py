"""Matrix multiplication 3x5x7 - rank-minimal exact tensor decomposition search."""
import numpy as np
from fractions import Fraction
import time
import random
from itertools import product

DIMENSIONS = [3, 5, 7]
DEADLINE = 140.0  # safety margin before 150s kill

def _time_remaining():
    return DEADLINE - time.time()

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,i')delta(j,j')delta(k,k')"""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 if (i,j,k) corresponds to valid multiplication, 0 otherwise
    # Actually for matrix mult: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The target tensor is: T_target[i,j,k] = 1 if i,i',j,j',k,k' ... 
    # Wait, let me re-read. The tensor for matmul C = A*B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor has indices (i, j, k) and value 1 (it's the structural tensor).
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    # But that's not right either. The tensor T_{i,j,k} = 1 means the structural tensor.
    # Actually the matrix multiplication tensor T has T_{i,j,k} = 1 for all i in [n], j in [m], k in [p].
    # No wait - the tensor is T_{i,j,k} where the multiplication is sum_j A_{i,j} B_{j,k} = C_{i,k}
    # The structural tensor has T_{i,j,k} = 1 for all i,j,k (it's the all-ones tensor in the appropriate space)
    # Hmm, actually no. Let me think more carefully.
    # The tensor for matrix multiplication C = A * B is:
    # T_{i,j,k} = 1 for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}
    # And the decomposition says T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for ALL i,j,k
    # That seems too restrictive. Let me reconsider.
    
    # Actually, I think the convention is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T has entries T_{(i,j), (j,k), (k,i)} or similar.
    # Given the W indexing W[k*n+i, r], the third index of the tensor is (k,i) which maps to k*n+i.
    # So T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And this should equal 1 for all valid (i,j,k) triples? That can't be right for rank minimality.
    
    # Let me re-read: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for n x m times m x p has rank at least max(n*m, m*p, n*p) and at most n*m*p.
    # The tensor is T_{i,j,k} = 1 (structural tensor, all ones).
    # So yes, we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k.
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > 1e-6:
                    return False, val
    return True, 1.0

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if s != 1:
                    return False
    return True

def _to_rational_list(mat, tol=1e-9):
    """Convert float matrix to list of Fractions, rounding to nearest rational with small denominator."""
    n_rows = len(mat)
    n_cols = len(mat[0])
    result = [[Fraction(0) for _ in range(n_cols)] for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i][j]
            # Try to find a simple rational
            if abs(val) < tol:
                result[i][j] = Fraction(0)
            else:
                # Try denominators up to 20
                found = False
                for denom in range(1, 21):
                    num = round(val * denom)
                    if abs(val - num/denom) < tol:
                        result[i][j] = Fraction(num, denom)
                        found = True
                        break
                if not found:
                    # Use a larger denominator
                    result[i][j] = Fraction(val).limit_denominator(1000)
    return result

def _local_search(U, V, W, n, m, p, max_iters=2000, tol=1e-10):
    """Local refinement to minimize max |sum_r U*V*W - 1| over all (i,j,k)."""
    rank = U.shape[1]
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    
    def objective():
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    best_obj = objective()
    lr = 0.01
    for it in range(max_iters):
        if _time_remaining() < 5:
            break
        # Compute gradient-like updates
        # For each (i,j,k), the error is e_{ijk} = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
        # We want to minimize sum e_{ijk}^2
        # Gradient w.r.t. U[i*m+j, r] = sum_{k'} e_{i,j,k'} * V[j*p+k', r] * W[k'*n+i, r]
        # This is expensive, so use a simpler approach: coordinate descent style
        
        # Pick a random (i,j,k)
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        
        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        err = val - 1.0
        if abs(err) < tol:
            continue
        
        # Adjust one of the three vectors
        choice = random.randint(0, 2)
        if choice == 0:
            # Update U[i*m+j, :]
            grad = V[j*p+k, :] * W[k*n+i, :]
            U[i*m+j, :] -= lr * err * grad
        elif choice == 1:
            # Update V[j*p+k, :]
            grad = U[i*m+j, :] * W[k*n+i, :]
            V[j*p+k, :] -= lr * err * grad
        else:
            # Update W[k*n+i, :]
            grad = U[i*m+j, :] * V[j*p+k, :]
            W[k*n+i, :] -= lr * err * grad
        
        # Check improvement
        obj = objective()
        if obj < best_obj:
            best_obj = obj
            # Decay learning rate slowly
            if it % 500 == 0 and it > 0:
                lr *= 0.9
        else:
            # Revert
            if choice == 0:
                U[i*m+j, :] += lr * err * grad
            elif choice == 1:
                V[j*p+k, :] += lr * err * grad
            else:
                W[k*n+i, :] += lr * err * grad
    
    return U, V, W, best_obj

def _random_decomposition(rank, n, m, p, seed=None):
    """Generate a random rank-r decomposition."""
    if seed is not None:
        random.seed(seed)
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    return U, V, W

def _try_rank(rank, n, m, p, max_time=30.0):
    """Try to find a valid decomposition of given rank."""
    start = time.time()
    best_obj = float('inf')
    best_U = best_V = best_W = None
    
    num_attempts = 50
    for attempt in range(num_attempts):
        if time.time() - start > max_time - 2:
            break
        if _time_remaining() < 10:
            break
        
        U, V, W = _random_decomposition(rank, n, m, p, seed=attempt*1000+42)
        
        # Local search
        U, V, W, obj = _local_search(U, V, W, n, m, p, max_iters=3000)
        
        if obj < best_obj:
            best_obj = obj
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Check if we got close enough to try rational recovery
        if obj < 1e-4:
            # Try to recover exact solution
            U_r = _to_rational_list(best_U.tolist())
            V_r = _to_rational_list(best_V.tolist())
            W_r = _to_rational_list(best_W.tolist())
            
            if _verify_exact(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 105
    
    # First, ensure we have a valid fallback (schoolbook)
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    fallback = schoolbook()
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, n*p) = max(15, 35, 21) = 35
    # Also rank >= ceil(n*m*p / max(n,m,p)) or other bounds
    # Let's try ranks from a reasonable lower bound up to schoolbook
    
    # Theoretical lower bound for matmul rank: at least max(n*m, m*p, p*n)
    lower_bound = max(n*m, m*p, p*n)  # 35
    
    # Try a range of ranks
    ranks_to_try = list(range(lower_bound, schoolbook_rank + 1, 5))
    # Make sure we include some key values
    for r in [35, 36, 37, 38, 39, 40, 45, 50, 55, 60, 70, 80, 90, 100, 105]:
        if r not in ranks_to_try and lower_bound <= r <= schoolbook_rank:
            ranks_to_try.append(r)
    ranks_to_try = sorted(set(ranks_to_try))
    
    best_result = fallback
    best_rank = schoolbook_rank
    
    for rank in ranks_to_try:
        if _time_remaining() < 15:
            break
        if rank >= best_rank:
            break
            
        result = _try_rank(rank, n, m, p, max_time=min(20.0, _time_remaining() - 5))
        if result is not None:
            U_r, V_r, W_r = result
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            best_rank = rank
            # If we found a valid decomposition, we can stop (we want minimum rank)
            # But let's continue to see if we can go lower
            if rank == lower_bound:
                break
    
    return best_result
