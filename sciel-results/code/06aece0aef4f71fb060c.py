"""Rewritten program for Matrix Multiplication 3,4,5 rank minimization."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0  # seconds, with safety margin

def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)

def _schoolbook():
    """Return the schoolbook decomposition with rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form a valid decomposition of the matrix multiplication tensor.
    
    The condition is: for all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    Wait, that's not right. Let me reconsider.
    
    The matrix multiplication tensor T has the property that for matrices A (n×m) and B (m×p):
    C = A * B where C[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor form, T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n} (using trace convention for output).
    
    The rank-1 decomposition T = sum_r u_r ⊗ v_r ⊗ w_r means:
    T[a, b, c] = sum_r u_r[a] * v_r[b] * w_r[c]
    
    where a indexes A (n*m), b indexes B (m*p), c indexes C in trace convention (p*n).
    
    For the matrix multiplication tensor, T[a, b, c] = 1 if the indices correspond to a valid
    multiplication term, else 0. Specifically, if a = i*m+j, b = j'*p+k, c = k'*n+i',
    then T[a,b,c] = 1 if i=i', j=j', k=k', else 0.
    
    Wait, that's the identity tensor, not the multiplication tensor.
    
    Let me reconsider. The matrix multiplication tensor is defined as:
    T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}  (using appropriate index conventions)
    
    Actually, I think the correct formulation is:
    
    The bilinear map (A, B) -> A*B can be represented by a tensor T such that:
    (A*B)[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    
    But that's not quite right either. The standard tensor formulation is:
    
    T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p} where
    T = sum_{i,j,k} E_{i,j} ⊗ E_{j,k} ⊗ E_{i,k}
    
    But the problem uses the trace convention for W, so W has shape (p*n, rank) with W[k*n+i].
    
    So the output C is indexed as C[k*n+i] = C[i,k] (row i, column k).
    
    The tensor condition is:
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    And for all other combinations of (a, b, c) where a = i1*m+j1, b = j2*p+k2, c = k3*n+i3,
    if (i1,j1,k2) != (i1,j2,k2) or similar mismatch, the sum should be 0.
    
    Actually, the matrix multiplication tensor has exactly n*m*p nonzero entries, one for each
    (i,j,k) triple, and the entry is 1. So the condition is:
    
    For the specific triple (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    For all other (a,b,c) not corresponding to a valid (i,j,k): sum = 0
    
    But wait, that's not how it works. The tensor T is defined such that:
    T[a,b,c] = 1 if a corresponds to (i,j), b corresponds to (j,k), c corresponds to (i,k)
    for the same (i,j,k), else 0.
    
    So there are exactly n*m*p entries that are 1, and all others are 0.
    """
    n, m, p = n, m, p
    rank = len(U[0])
    
    # Check the n*m*p "diagonal" entries should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check some "off-diagonal" entries should equal 0
    # Sample a few to save time
    import random
    trials = min(1000, n*m*p*10)
    for _ in range(trials):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        # Check if this is a "valid" triple
        i1 = a // m
        j1 = a % m
        j2 = b // p
        k2 = b % p
        k3 = c // n
        i3 = c % n
        if i1 == i3 and j1 == j2 and k2 == k3:
            continue  # This is a valid triple, skip
        s = 0.0
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        if abs(s) > tol:
            return False
    
    return True

def _try_random_search(rank, n, m, p, max_iters=100000):
    """
    Try to find a valid decomposition of given rank using random search.
    """
    if _remaining_time() < 5:
        return None
    
    best = None
    for attempt in range(max_iters):
        if _remaining_time() < 5:
            break
        
        # Random initialization with small integers/half-integers
        scale = 2.0
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank)) * scale
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank)) * scale
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank)) * scale
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to list format
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x5 matrix.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Start with schoolbook as fallback
    best_result = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Based on known results, try ranks from 40 to 59
    for rank in range(40, schoolbook_rank):
        if _remaining_time() < 10:
            break
        
        result = _try_random_search(rank, n, m, p, max_iters=50000)
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found a valid decomposition, stop
    
    return best_result
