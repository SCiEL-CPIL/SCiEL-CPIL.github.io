"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact: sum_c U[i*m+j,c]*V[j*p+k,c] == delta[i,k] for all i,j,k
    Wait, the tensor for matrix multiplication T[i,j,k] = delta[i,k] is not right.
    The tensor is T[i,j,k] such that (AB)[i,k] = sum_j A[i,j] B[j,k].
    In tensor form, C[i,k] = sum_{j,c} A[i,j] B[j,k] W... no.
    The standard decomposition: C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) * (sum_k V[j,k,c] B[j,k])? No.
    
    Correct tensor decomposition of matrix multiplication:
    The multiplication tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    T[i,j,k] = 1 if i==k? No.
    Let's look at the schoolbook:
    U[i*m+j, c] = 1 if c corresponds to (i,j,k)
    V[j*p+k, c] = 1 if c corresponds to (i,j,k)
    W[k*n+i, c] = 1 if c corresponds to (i,j,k)
    
    The contraction is: C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])?
    Actually, the standard form is:
    C = A * B
    C[i,k] = sum_j A[i,j] B[j,k]
    
    In tensor decomposition form, we want:
    T_{i,j,k} = sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)} ? No.
    
    Let's re-read the task description carefully.
    "output factor follows the official trace convention W[k*n+i,r]"
    The matrices are U (n*m, rank), V (m*p, rank), W (p*n, rank).
    The decomposition is:
    T[i,j,k] = sum_{c=0}^{rank-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For matrix multiplication, the tensor T[i,j,k] is 1 if i==k? No.
    The tensor T represents the bilinear map (A, B) -> C.
    C[i,k] = sum_j A[i,j] B[j,k].
    This is a trilinear form? No, it's a bilinear map.
    The tensor T is defined such that C[i,k] = sum_{j,c} A[i,j] B[j,k] * ...
    
    Actually, the standard tensor decomposition for matrix multiplication is:
    T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} is not quite right because the indices don't match up for a simple product.
    
    Let's look at the schoolbook example again:
    U[i*m+j, c] = 1
    V[j*p+k, c] = 1
    W[k*n+i, c] = 1
    for c = i*m*p + j*p + k.
    
    Then sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 if there exists c such that all three are 1.
    For a fixed (i,j,k), only one c has U=1, V=1, W=1. That c is the one corresponding to (i,j,k).
    So the sum is 1 for all (i,j,k).
    But the matrix multiplication tensor should be T[i,j,k] = delta_{i,k}? No.
    
    The matrix multiplication tensor T is defined by T[i,j,k] = 1 if the (i,k)-th entry of the product depends on A[i,j] and B[j,k].
    In the product C = AB, C[i,k] = sum_j A[i,j] B[j,k].
    So the coefficient of A[i,j] B[j,k] in C[i,k] is 1.
    The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all i, j, k? No.
    The tensor T[i,j,k] is the coefficient of A[i,j] B[j,k] in the expression for C.
    But C is an n x p matrix. The entry C[i,k] depends on A[i,j] and B[j,k] for all j.
    So the "tensor" of the bilinear map is actually a 3-tensor with indices (i, j, k) where i is the row of A, j is the column of A / row of B, k is the column of B.
    And T[i,j,k] = 1 for all i,j,k.
    
    Wait, the task says "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix by a 5 by 7 matrix."
    The tensor T has size n x m x p. For matrix multiplication, T[i,j,k] = 1 for all i,j,k.
    Because C[i,k] = sum_j A[i,j] B[j,k], so the term A[i,j] B[j,k] appears with coefficient 1 in C[i,k].
    
    So we need to decompose the all-ones tensor T[i,j,k] = 1 into a sum of rank-1 tensors:
    1 = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    for all i, j, k.
    
    The schoolbook decomposition has rank n*m*p = 3*5*7 = 105.
    Can we do better?
    
    For small matrices, the minimum rank is known.
    For 3x5 * 5x7, the minimum rank is not trivially known, but we can try to find a better decomposition using numerical methods.
    
    Let's try to use a numerical approach to find a lower rank decomposition.
    We can set up the problem as a system of polynomial equations and try to solve it, or use a heuristic search.
    
    Since the entries are required to be integer/half-integer, we can try to search over small integer/half-integer values.
    
    Let's try a simple approach: start with the schoolbook decomposition and try to merge terms or find a lower rank decomposition.
    
    Actually, for matrix multiplication, the minimum rank is at least max(n*m, m*p, p*n) = max(15, 35, 21) = 35.
    And at most n*m*p = 105.
    
    Let's try to find a decomposition of rank 35 or lower.
    
    We can use a randomized approach: initialize U, V, W with random integer/half-integer values and try to minimize the error using gradient descent or a local search.
    
    However, the problem is non-convex and the search space is discrete.
    
    Let's try a different approach: use the fact that the tensor is all-ones, and try to find a low-rank decomposition.
    
    We can use a simple heuristic: try to find a decomposition of rank r by solving a system of equations.
    
    For a given rank r, we have:
    U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i, j, k, c.
    
    This is a system of n*m*p = 105 equations in r*(n*m + m*p + p*n) = r*(15+35+21) = 71r unknowns.
    
    For r = 35, we have 71*35 = 2485 unknowns and 105 equations. This is an underdetermined system, so we can try to find a solution.
    
    Let's try to find a solution using a simple iterative method.
    
    We can fix one of the matrices, say W, and try to solve for U and V.
    
    Alternatively, we can use a randomized approach: initialize U, V, W with random values and try to minimize the error using a local search.
    
    Let's implement a simple local search:
    1. Start with the schoolbook decomposition.
    2. Try to remove one term (set U, V, W for one column to zero) and see if the error increases.
    3. If the error decreases, keep the change.
    
    However, the schoolbook decomposition is exact, so removing a term will increase the error.
    
    Instead, let's try to find a new decomposition from scratch.
    
    We can use a numerical solver to find a low-rank decomposition.
    
    Let's try to use a simple approach: initialize U, V, W with random integer/half-integer values and try to minimize the error using a local search.
    
    We'll use a deadline guard to ensure we don't exceed the time limit.
    
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    n, m, p = DIMENSIONS
    rank = U_np.shape[1]
    
    # Compute the error
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                error += (val - 1.0)**2
    return error

def _random_init(n, m, p, rank, scale=1.0):
    """Initialize U, V, W with random integer/half-integer values."""
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    for c in range(rank):
        U[:, c] = np.random.randint(-scale, scale+1, size=n*m) * 0.5
        V[:, c] = np.random.randint(-scale, scale+1, size=m*p) * 0.5
        W[:, c] = np.random.randint(-scale, scale+1, size=p*n) * 0.5
    return U, V, W

def _local_search(U, V, W, n, m, p, max_iters=1000):
    """Perform a simple local search to minimize the error."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    best_error = _verify(U, V, W, n, m, p)
    
    for _ in range(max_iters):
        # Pick a random column and a random entry to modify
        c = np.random.randint(rank)
        mat = np.random.randint(3)
        if mat == 0:
            idx = np.random.randint(n*m)
            old_val = U[idx, c]
            new_val = old_val + np.random.choice([-0.5, 0.5])
            U[idx, c] = new_val
        elif mat == 1:
            idx = np.random.randint(m*p)
            old_val = V[idx, c]
            new_val = old_val + np.random.choice([-0.5, 0.5])
            V[idx, c] = new_val
        else:
            idx = np.random.randint(p*n)
            old_val = W[idx, c]
            new_val = old_val + np.random.choice([-0.5, 0.5])
            W[idx, c] = new_val
        
        new_error = _verify(U, V, W, n, m, p)
        if new_error < best_error:
            best_error = new_error
        else:
            # Revert
            if mat == 0:
                U[idx, c] = old_val
            elif mat == 1:
                V[idx, c] = old_val
            else:
                W[idx, c] = old_val
    
    return U, V, W, best_error

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s margin
    
    # Start with the schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1
                V[j*p+k, column] = 1
                W[k*n+i, column] = 1
                column += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to find a lower rank decomposition
    for r in range(35, rank+1):
        if time.time() > deadline:
            break
        # Try a few random initializations
        for _ in range(10):
            if time.time() > deadline:
                break
            U_r, V_r, W_r = _random_init(n, m, p, r, scale=2)
            U_r, V_r, W_r, err = _local_search(U_r, V_r, W_r, n, m, p, max_iters=100)
            if err < 1e-6:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = r
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
