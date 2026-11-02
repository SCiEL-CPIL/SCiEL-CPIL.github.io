"""Rewritten program for matrix multiplication 2x5 by 5x5 rank minimization."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the full tensor T[i,j,k] which should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                vals = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                s = np.sum(vals)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_rank_reduction(n, m, p, base_rank, max_tries=50, deadline=None):
    """
    Try to find a decomposition of rank less than base_rank by starting from schoolbook
    and attempting to merge/eliminate terms via linear dependencies.
    """
    n, m, p = 2, 5, 5
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    
    # Start with schoolbook
    U = np.zeros((nm, base_rank))
    V = np.zeros((mp, base_rank))
    W = np.zeros((pn, base_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = base_rank
    
    for attempt in range(max_tries):
        if deadline is not None and time.time() > deadline - 5:
            break
            
        # Try to find a column that can be eliminated
        # We look for a column c such that the tensor contribution of column c
        # can be expressed as a linear combination of other columns.
        
        # For a given column c, its contribution to T[i,j,k] is:
        # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # We want to see if for all (i,j,k):
        # U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_{c' != c} alpha_{c'} * U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c']
        
        # This is a linear system in the alphas. Let's build it.
        # Number of equations: n*m*p = 50
        # Number of unknowns: base_rank - 1
        
        # Build matrix A where A[eq, c'] = U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c'] for c' != c
        # And b[eq] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        
        cols = [c_prime for c_prime in range(base_rank) if c_prime != c]
        num_cols = len(cols)
        
        # Build the system
        A = np.zeros((n*m*p, num_cols))
        b = np.zeros(n*m*p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    b[eq_idx] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    for col_idx, c_prime in enumerate(cols):
                        A[eq_idx, col_idx] = U[i*m+j, c_prime] * V[j*p+k, c_prime] * W[k*n+i, c_prime]
                    eq_idx += 1
        
        # Solve least squares
        try:
            alpha, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        # Check if the solution is valid (residuals close to 0)
        if len(residuals) == 0 or residuals[0] > 1e-6:
            continue
        
        # If valid, we can eliminate column c
        # New rank is base_rank - 1
        # But we need to update U, V, W to reflect the elimination
        
        # The new decomposition: for each remaining column c', its coefficient is 1 + alpha_{c'}
        # Wait, that's not quite right. The elimination means:
        # contribution of c = sum alpha_{c'} * contribution of c'
        # So we can absorb c into the others by scaling.
        
        # Actually, we just need to verify the new set of columns (without c) works.
        # The new columns are cols, and we need to check if they span the tensor.
        # Since A * alpha ≈ b, the remaining columns can reproduce the tensor.
        
        # But we need actual U, V, W matrices. The issue is that the "columns" in the
        # decomposition are not independent vectors - they're triplets (u_c, v_c, w_c).
        # The linear combination works on the tensor level, but we need to express
        # the tensor as sum of rank-1 tensors.
        
        # If A*alpha = b, then for each (i,j,k):
        # U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_{c'} alpha_{c'} * U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c']
        
        # This means the rank-1 tensor for column c equals the sum of scaled rank-1 tensors.
        # But a sum of rank-1 tensors is not generally a rank-1 tensor, so we can't just
        # replace column c with a single new column.
        
        # The correct approach: we can remove column c and adjust the other columns.
        # But this only works if the tensor for c is exactly a linear combination of
        # the others, which means we can drop it and the remaining still span.
        
        # Actually, for the decomposition to be valid, we need:
        # sum_c u_c ⊗ v_c ⊗ w_c = T
        # If we drop column c, we need sum_{c' != c} u_{c'} ⊗ v_{c'} ⊗ w_{c'} = T - u_c ⊗ v_c ⊗ w_c
        # This is NOT the same as what we checked. What we checked was whether
        # u_c ⊗ v_c ⊗ w_c is in the span of {u_{c'} ⊗ v_{c'} ⊗ w_{c'} : c' != c}
        # which is a different condition.
        
        # Let me try a different approach: directly search for lower rank decompositions
        # using random initialization and gradient-free methods.
        pass
    
    return best_result, best_rank

def _random_search(n, m, p, target_rank, max_iter=200, deadline=None):
    """
    Random search for a decomposition of the given rank.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
            
        U = np.random.randint(0, 2, size=(nm, target_rank))
        V = np.random.randint(0, 2, size=(mp, target_rank))
        W = np.random.randint(0, 2, size=(pn, target_rank))
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(target_rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 50
    
    # Try random search for lower ranks
    # Known lower bounds: rank >= n*m*p / max(n,m,p) ... actually various bounds exist
    # For 2x5x5, the rank is known to be at most some value. Let's try to find it.
    
    # Try ranks from 50 down
    for rank in range(best_rank - 1, 30, -1):
        if time.time() > deadline - 10:
            break
        res = _random_search(n, m, p, rank, max_iter=100, deadline=deadline)
        if res is not None:
            best_rank = rank
            result = res
            break
    
    return result
