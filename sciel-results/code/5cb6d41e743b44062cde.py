"""Matrix multiplication 3x5x7 rank minimization."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 7]
n, m, p = DIMENSIONS

# Deadline guard
START_TIME = time.time()
DEADLINE = 115.0  # 150s limit, 35s safety margin

def _time_left():
    return DEADLINE - time.time()

def _check_decomposition(U, V, W):
    """Check if U, V, W form a valid decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1
    For all other (i,j,k) combinations in the tensor, the product should be 0.
    
    Actually, the matrix multiplication tensor T[i,j,k] is defined as:
    T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    
    The tensor for matrix multiplication has entries:
    T_{i,j,k} = delta (it's 1 for the "standard" multiplication tensor)
    
    Wait, let me think more carefully. The matrix multiplication tensor T is a 3-tensor
    of dimensions (n, m, p) where:
    T[i, j, k] = 1 if the entry corresponds to C[i,k] = sum_j A[i,j] B[j,k]
    
    Actually, the standard matrix multiplication tensor has:
    T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    No wait. The matrix multiplication tensor T is defined such that:
    C[i,k] = sum_{j} A[i,j] * B[j,k]
    
    As a 3-tensor, T has indices (i, j, k) where:
    - i indexes rows of A and C (size n)
    - j indexes columns of A / rows of B (size m)  
    - k indexes columns of B and C (size p)
    
    And T[i,j,k] = 1 for all valid (i,j,k).
    
    A rank-r decomposition means:
    T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    Wait, the problem says W[k*n+i, r] follows the official trace convention.
    
    So the decomposition is:
    For all i in [0,n), j in [0,m), k in [0,p):
    sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    where T[i,j,k] = 1 (the matrix multiplication tensor is all ones).
    
    But wait, that can't be right because then we'd just need U*V*W = 1 for all indices.
    
    Let me reconsider. The matrix multiplication tensor is actually:
    T_{i,j,k} = 1 if (i,j,k) is a valid triple, but the structure is that
    the multiplication is: C = A * B where C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T has entries T[i,j,k] = 1 for all i,j,k in their respective ranges.
    The rank of this tensor is what we're trying to minimize.
    
    Actually no. Let me think again. The bilinear map (A,B) -> C where C = AB
    can be represented as a 3-tensor T with T[i,j,k] = delta_{ik}... no.
    
    OK let me be very precise. We have:
    - A is n x m
    - B is m x p  
    - C = AB is n x p
    - C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    The tensor T is a 3-tensor of shape (n, m, p) where:
    T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    And a rank-r decomposition means:
    T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    Wait, but that would mean for every (i,j,k) triple, the sum equals 1.
    
    Hmm, but that's not quite right either. Let me look at this differently.
    
    The matrix multiplication tensor is typically defined as:
    T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    
    where e_i, e_j, e_k are standard basis vectors. And T[i,j,k] = 1 for all valid indices.
    
    A rank-r decomposition of T means:
    T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    
    where u_c is in R^n, v_c in R^m, w_c in R^p.
    
    In terms of the flattened matrices:
    U has shape (n*m, r) where U[i*m+j, c] = u_c[i] * v_c[j]
    V has shape (m*p, r) where V[j*p+k, c] = v_c[j] * w_c[k]  
    W has shape (p*n, r) where W[k*n+i, c] = w_c[k] * u_c[i]
    
    And the condition is:
    For all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    But wait, that's a very strong condition. Let's verify with the schoolbook:
    U[i*m+j, c] = 1 for c = i*m*p + j*p + k
    V[j*p+k, c] = 1 for c = i*m*p + j*p + k
    W[k*n+i, c] = 1 for c = i*m*p + j*p + k
    
    So for a given (i,j,k), only one term c = i*m*p + j*p + k contributes, and it gives 1*1*1 = 1. ✓
    
    For a different (i',j',k') ≠ (i,j,k), the product U[i'*m+j',c] * V[j'*p+k',c] * W[k'*n+i',c] 
    is nonzero only if c corresponds to (i',j',k'), so it's 0. ✓
    
    Great, so the condition is indeed:
    For all i in [0,n), j in [0,m), k in [0,p):
    sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is a system of n*m*p = 105 equations in 3*r variables (well, U has n*m*r = 15r entries,
    V has m*p*r = 35r entries, W has p*n*r = 21r entries, total 71r unknowns).
    
    But the equations are cubic (products of three variables), not linear!
    
    So this is NOT a linear system. It's a system of polynomial equations of degree 3.
    
    To make it linear, we can fix two of the three matrices and solve for the third.
    For example, if we fix U and V, then for each (i,j,k):
    sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    This is linear in W! So we can:
    1. Fix U and V randomly
    2. Solve the linear system for W
    3. Check if the solution is exact (integer/half-integer)
    
    Or better, we can use a more systematic approach.
    
    Actually, let me reconsider. The problem says "integer/half-integer entries". This suggests
    the solution might have a specific structure.
    
    Let me think about known results. For 3x5x7 matrix multiplication:
    - Schoolbook rank: 3*5*7 = 105
    - Lower bound: max(n, m, p) * min(n,m,p) = 7*3 = 21 (very weak)
    - Better lower bound: n*m*p / (something)
    
    Actually, the tensor rank of the matrix multiplication tensor [n,m,p] is known to be
    at most n*m*p and at least max(n,m,p). For small cases, exact values are known.
    
    For [3,5,7], I don't know the exact minimum rank off the top of my head. Let me try to
    find a good decomposition.
    
    Strategy: Use random initialization and solve the linear system for W given fixed U and V.
    Then iterate: fix W, solve for U; fix U, solve for V; etc.
    
    But the equations are cubic, so fixing two and solving for the third gives linear equations.
    
    Let me implement this:
    1. Initialize U, V randomly with small integer values
    2. Solve for W: For each k*n+i (row of W), we have m equations (one for each j)
       sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
       
       This is a system of m equations in r unknowns (the entries W[k*n+i, c] for c=0..r-1).
       
    3. Check if W has nice values (integer/half-integer)
    
    Let me code this up.
"""

def _solve_linear_system(A, b):
    """Solve A x = b using numpy. Returns None if no solution or underdetermined."""
    try:
        # Check if system is consistent
        A_rank = np.linalg.matrix_rank(A)
        Ab = np.column_stack([A, b]) if b.ndim == 1 else np.hstack([A, b])
        Ab_rank = np.linalg.matrix_rank(Ab)
        if A_rank != Ab_rank:
            return None
        if A_rank < A.shape[1]:
            # Underdetermined - return minimum norm solution
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return x
        else:
            return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

def _is_nice(x, tol=1e-9):
    """Check if x is close to an integer or half-integer."""
    return np.allclose(x, np.round(2*x)/2, atol=tol)

def _to_nice(x, tol=1e-9):
    """Round x to nearest integer or half-integer if close."""
    rounded = np.round(2*x)/2
    if np.allclose(x, rounded, atol=tol):
        return rounded
    return x

def _check_decomposition_full(U, V, W, tol=1e-9):
    """Full check of decomposition."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1) > tol:
                    return False
    return True

def _try_rank(r, max_iter=500, seed=42):
    """Try to find a rank-r decomposition using alternating least squares."""
    rng = np.random.RandomState(seed)
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for attempt in range(20):
        if _time_left() < 5:
            break
            
        # Initialize U, V with small random integers
        U = rng.randint(-2, 3, size=(n*m, r)).astype(float)
        V = rng.randint(-2, 3, size=(m*p, r)).astype(float)
        
        # Ensure some structure - make each row have at least one nonzero
        for i in range(n*m):
            if np.all(U[i] == 0):
                U[i, rng.randint(0, r)] = 1
        for j in range(m*p):
            if np.all(V[j] == 0):
                V[j, rng.randint(0, r)] = 1
        
        for it in range(max_iter):
            if _time_left() < 5:
                break
                
            # Solve for W given U, V
            W = np.zeros((p*n, r))
            solvable = True
            for i in range(n):
                for k in range(p):
                    # For row k*n+i of W, solve:
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for j=0..m-1
                    A_mat = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A_mat[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    b_vec = np.ones(m)
                    x = _solve_linear_system(A_mat, b_vec)
                    if x is None:
                        solvable = False
                        break
                    W[k*n+i] = x
                if not solvable:
                    break
            
            if not solvable:
                continue
            
            # Check if W is nice
            if _is_nice(W):
                W = _to_nice(W)
                # Verify
                if _check_decomposition_full(U, V, W):
                    return U, V, W
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                        err += (val - 1)**2
            err /= (n*m*p)
            
            if err < best_error:
                best_error = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            # Also try solving for U given V, W
            U_new = np.zeros((n*m, r))
            solvable = True
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            A_mat[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    b_vec = np.ones(p)
                    x = _solve_linear_system(A_mat, b_vec)
                    if x is None:
                        solvable = False
                        break
                    U_new[i*m+j] = x
                if not solvable:
                    break
            if solvable:
                if _is_nice(U_new):
                    U_new = _to_nice(U_new)
                    if _check_decomposition_full(U_new, V, W):
                        return U_new, V, W
                U = U_new
            
            # Solve for V given U, W
            V_new = np.zeros((m*p, r))
            solvable = True
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            A_mat[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    b_vec = np.ones(n)
                    x = _solve_linear_system(A_mat, b_vec)
                    if x is None:
                        solvable = False
                        break
                    V_new[j*p+k] = x
                if not solvable:
                    break
            if solvable:
                if _is_nice(V_new):
                    V_new = _to_nice(V_new)
                    if _check_decomposition_full(U, V_new, W):
                        return U, V_new, W
                V = V_new
        
        # After iterations, check best solution
        if best_U is not None:
            if _is_nice(best_U) and _is_nice(best_V) and _is_nice(best_W):
                U_nice = _to_nice(best_U)
                V_nice = _to_nice(best_V)
                W_nice = _to_nice(best_W)
                if _check_decomposition_full(U_nice, V_nice, W_nice):
                    return U_nice, V_nice, W_nice
    
    return None

def _make_schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
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

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    # Start with schoolbook
    schoolbook = _make_schoolbook()
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try ranks from lower bound upward
    # Lower bound: at least max(n, m, p) = 7, but realistically higher
    # For [3,5,7], try ranks from 15 upward
    for r in range(15, best_rank + 1):
        if _time_left() < 10:
            break
        
        result = _try_rank(r, max_iter=200, seed=r)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            # Verify
            if _check_decomposition_full(np.array(U_list), np.array(V_list), np.array(W_list)):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try a few more random attempts with the best rank found so far
    if _time_left() > 20:
        for r in range(15, min(30, best_rank)):
            if _time_left() < 10:
                break
            for seed in range(50):
                if _time_left() < 5:
                    break
                result = _try_rank(r, max_iter=100, seed=seed + 1000)
                if result is not None:
                    U, V, W = result
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    if _check_decomposition_full(np.array(U_list), np.array(V_list), np.array(W_list)):
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result
