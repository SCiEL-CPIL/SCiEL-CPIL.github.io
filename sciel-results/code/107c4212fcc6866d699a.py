"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import numpy as np
from fractions import Fraction
import random
import time

# EVOLVE-BLOCK-START
def _verify_exact(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly satisfies the tensor identity.
    
    For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    For all i, j, k, i', j', k' where (i,j,k) != (i',j',k'): the sum == 0
    
    Actually, the correct condition is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,i')*delta(j,j')*delta(k,k')
    where the RHS is 1 if (i,j,k) matches the "output" index, 0 otherwise.
    
    More precisely, the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    in the range, and the decomposition must satisfy:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, that's not right either. Let me think about this more carefully.
    
    The matrix multiplication tensor for multiplying an (n x m) matrix by an (m x p) matrix
    to get an (n x p) result. The standard schoolbook has rank n*m*p.
    
    The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    T[i,j,k] = 1 (it's the identity-like tensor for matmul).
    
    A rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    So we need: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    AND for the "cross terms" to vanish... wait, actually the tensor is just all ones?
    No. Let me reconsider.
    
    Actually, the matrix multiplication tensor is defined such that:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i, j, k] represents the coefficient of A[i,j] * B[j,k] in C[i,k], which is 1.
    
    So T[i,j,k] = 1 for all valid i, j, k.
    
    A rank-r decomposition means: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    So we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for ALL i, j, k.
    
    That's n*m*p = 4*4*5 = 80 equations.
    """
    # Check all n*m*p entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _als_step(U, V, W, n, m, p, rank, max_iter=500, tol=1e-10):
    """Alternating least squares to refine the decomposition.
    
    Minimize ||T - U*V*W||_F where the product is defined as:
    (U*V*W)[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Target tensor: all ones
    # We'll use a simplified ALS approach
    
    for iteration in range(max_iter):
        # Update U: for each (i, j) and column c, we want
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
        # This is a system of p equations for each (i,j)
        # For fixed (i,j), define for each c: a_c[k] = V[j*p+k, c] * W[k*n+i, c] for k in [0,p)
        # Then sum_c U[i*m+j, c] * a_c[k] = 1 for all k
        # This is: A_{(i,j)} * u_{(i,j)} = 1_vector
        # where A_{(i,j)}[k, c] = V[j*p+k, c] * W[k*n+i, c]
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build system: for each k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                A = np.zeros((p, rank))
                for k in range(p):
                    for c in range(rank):
                        A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                b = np.ones(p)
                # Solve least squares
                try:
                    sol, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = sol
                except:
                    U_new[i*m+j, :] = U[i*m+j, :]
        
        # Update V: for each (j, k) and column c
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    for c in range(rank):
                        A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                b = np.ones(n)
                try:
                    sol, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = sol
                except:
                    V_new[j*p+k, :] = V[j*p+k, :]
        
        # Update W: for each (k, i) and column c
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                b = np.ones(m)
                try:
                    sol, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = sol
                except:
                    W_new[k*n+i, :] = W[k*n+i, :]
        
        # Check convergence
        diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
        U, V, W = U_new, V_new, W_new
        if diff < tol:
            break
    
    return U, V, W

def _round_to_rational(arr, max_denom=1000):
    """Try to round a float array to simple rationals."""
    result = []
    for val in arr:
        if abs(val) < 1e-12:
            result.append(0)
        elif abs(val - round(val)) < 1e-10:
            result.append(int(round(val)))
        else:
            # Try to find a simple rational
            found = False
            for d in range(2, max_denom+1):
                num = val * d
                if abs(num - round(num)) < 1e-8:
                    n_val = int(round(num))
                    # Reduce fraction
                    from math import gcd
                    g = gcd(abs(n_val), d)
                    n_val //= g
                    d //= g
                    if d == 1:
                        result.append(n_val)
                    else:
                        result.append(n_val / d)
                    found = True
                    break
            if not found:
                result.append(val)
    return result

def _try_rank(rank, n, m, p, max_attempts=20, time_budget=90):
    """Try to find a valid rank-r decomposition."""
    start_time = time.time()
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        rng = random.Random(attempt * 12345 + 67890)
        scale = 1.0 / (rank ** 0.5)
        U = [[rng.uniform(-scale, scale) for _ in range(rank)] for _ in range(n*m)]
        V = [[rng.uniform(-scale, scale) for _ in range(rank)] for _ in range(m*p)]
        W = [[rng.uniform(-scale, scale) for _ in range(rank)] for _ in range(p*n)]
        
        # Run ALS
        U_arr, V_arr, W_arr = _als_step(U, V, W, n, m, p, rank, max_iter=200)
        
        # Convert back to lists
        U_list = U_arr.tolist()
        V_list = V_arr.tolist()
        W_list = W_arr.tolist()
        
        # Check if the solution is close to exact
        # Compute the error
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-8:
            # Try to rationalize
            U_rat = _round_to_rational([x for row in U_list for x in row], max_denom=100)
            V_rat = _round_to_rational([x for row in V_list for x in row], max_denom=100)
            W_rat = _round_to_rational([x for row in W_list for x in row], max_denom=100)
            
            # Reshape
            U_rat = [U_rat[i*rank:(i+1)*rank] for i in range(n*m)]
            V_rat = [V_rat[i*rank:(i+1)*rank] for i in range(m*p)]
            W_rat = [W_rat[i*rank:(i+1)*rank] for i in range(p*n)]
            
            # Verify exactly
            if _verify_exact(U_rat, V_rat, W_rat, n, m, p, rank):
                return U_rat, V_rat, W_rat
        
        # If not close enough, try a different initialization strategy
        # Use structured random initialization
        if attempt % 5 == 4:
            # Try initialization with some 1s and 0s
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            for c in range(rank):
                # Random sparse pattern
                if rng.random() < 0.7:
                    idx = rng.randint(0, n*m-1)
                    U[idx][c] = rng.choice([-1, 1])
                if rng.random() < 0.7:
                    idx = rng.randint(0, m*p-1)
                    V[idx][c] = rng.choice([-1, 1])
                if rng.random() < 0.7:
                    idx = rng.randint(0, p*n-1)
                    W[idx][c] = rng.choice([-1, 1])
            
            U_arr, V_arr, W_arr = _als_step(U, V, W, n, m, p, rank, max_iter=200)
            U_list = U_arr.tolist()
            V_list = V_arr.tolist()
            W_list = W_arr.tolist()
            
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c]
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-8:
                U_rat = _round_to_rational([x for row in U_list for x in row], max_denom=100)
                V_rat = _round_to_rational([x for row in V_list for x in row], max_denom=100)
                W_rat = _round_to_rational([x for row in W_list for x in row], max_denom=100)
                
                U_rat = [U_rat[i*rank:(i+1)*rank] for i in range(n*m)]
                V_rat = [V_rat[i*rank:(i+1)*rank] for i in range(m*p)]
                W_rat = [W_rat[i*rank:(i+1)*rank] for i in range(p*n)]
                
                if _verify_exact(U_rat, V_rat, W_rat, n, m, p, rank):
                    return U_rat, V_rat, W_rat
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Theoretical lower bound
    lower_bound = max(n*p, m*p, n*m)
    
    # Try ranks from lower_bound upward
    # Schoolbook rank is n*m*p = 80
    schoolbook_rank = n * m * p
    
    for rank in range(lower_bound, schoolbook_rank + 1):
        result = _try_rank(rank, n, m, p, max_attempts=15, time_budget=80)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
