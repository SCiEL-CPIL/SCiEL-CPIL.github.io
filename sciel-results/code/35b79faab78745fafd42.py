import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def _verify_full(U, V, W):
    """Full verification of the tensor decomposition for 2x3x6."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _find_rank7_2x2x3(max_restarts=20, deadline=None):
    """Try to find a rank-7 decomposition of 2x2x3 matrix multiplication."""
    n, m, p = 2, 2, 3
    rank = 7
    dim_U = n * m  # 4
    dim_V = m * p  # 6
    dim_W = p * n  # 6
    total_dim = dim_U * rank + dim_V * rank + dim_W * rank # 126
    
    def make_objective(x):
        U = x[:dim_U*rank].reshape(dim_U, rank)
        V = x[dim_U*rank:(dim_U+dim_V)*rank].reshape(dim_V, rank)
        W = x[(dim_U+dim_V)*rank:].reshape(dim_W, rank)
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    loss += (total - 1.0)**2
        return loss
    
    for attempt in range(max_restarts):
        if deadline and time.time() > deadline:
            break
        x0 = np.random.randn(total_dim) * 0.5
        # Use L-BFGS-B or BFGS
        result = minimize(make_objective, x0, method='BFGS', 
                         options={'maxiter': 5000, 'gtol': 1e-14})
        if result.fun < 1e-16:
            U = result.x[:dim_U*rank].reshape(dim_U, rank)
            V = result.x[dim_U*rank:(dim_U+dim_V)*rank].reshape(dim_V, rank)
            W = result.x[(dim_U+dim_V)*rank:].reshape(dim_W, rank)
            return U, V, W
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 110.0
    try:
        # Try to find rank-7 for 2x2x3
        U7, V7, W7 = _find_rank7_2x2x3(max_restarts=20, deadline=deadline)
        if U7 is not None and time.time() < deadline:
            # Construct rank-21 for 2x3x6
            n, m, p = DIMENSIONS # 2, 3, 6
            rank = 21
            U = [[0.0] * rank for _ in range(n * m)] # 6x21
            V = [[0.0] * rank for _ in range(m * p)] # 18x21
            W = [[0.0] * rank for _ in range(p * n)] # 12x21
            
            # We split p=6 into 3 blocks of size 2.
            # Block t (t=0,1,2) corresponds to k in [2t, 2t+1].
            # The sub-problem is 2x3x2 (A is 2x3, B_block is 3x2, C_block is 2x2).
            # Wait, my search was for 2x2x3.
            # 2x2x3: A(2x2), B(2x3).
            # 2x3x2: A(2x3), B(3x2).
            # These are isomorphic.
            # In 2x2x3 solution:
            # U7 indices: i in 0..1, j in 0..1. (4 rows)
            # V7 indices: j in 0..1, k in 0..2. (6 rows)
            # W7 indices: k in 0..2, i in 0..1. (6 rows)
            
            # In 2x3x2 usage:
            # We need A(2x3) and B(3x2).
            # This is NOT directly 2x2x3.
            # 2x2x3 has A with 2 cols. 2x3x2 has A with 3 cols.
            # So I cannot directly use 2x2x3 for 2x3x2.
            
            # I need to find rank for 2x3x2 directly.
            # Or split 2x3x6 into 2x2x6 and 1x3x6?
            
            # Let's restart the logic.
            # Is M(2,3,2) known?
            # M(2,3,2) = M(2,2,3) by symmetry of n and p?
            # C = AB. C^T = B^T A^T.
            # If A is 2x3, B is 3x2.
            # B^T is 2x3, A^T is 3x2.
            # So M(2,3,2) = M(2,2,3)? No.
            # M(n,m,p) is rank of tensor T_{i,j,k}.
            # T_{i,j,k} for 2x3x2: i=0,1; j=0,1,2; k=0,1.
            # T_{i,j,k} for 2x2x3: i=0,1; j=0,1; k=0,1,2.
            # These are different tensors.
            # However, M(2,3,2) <= M(2,2,2) + M(1,3,2)?
            
            # Let's just search for rank-10 or 12 for 2x3x2.
            # If I find rank 10, total rank 30.
            # If I find rank 9, total rank 27.
            
            # I will modify the search to target 2x3x2 rank 10.
            pass
    except Exception:
        pass
    return _schoolbook()
