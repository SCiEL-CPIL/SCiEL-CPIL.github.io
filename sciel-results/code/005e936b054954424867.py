"""Matrix multiplication 3,3,8 — rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Return the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-10):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    The constraint is:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1  for all i,j,k
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_structured_decomposition(n, m, p, deadline):
    """
    Try a structured decomposition based on grouping the p dimension.
    
    Key insight: For 3x3 x 3x8, we can group the 8 columns into two groups of 4.
    For each group, we compute A * B_group where B_group is 3x4.
    
    A 3x3 x 3x4 multiplication can potentially be done with less than 36 (schoolbook) rank.
    
    Actually, let me think differently. The tensor is:
    T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,8)
    
    We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    Let me try a rank-r decomposition where r < 72.
    
    Ansatz: Use a Kronecker-like structure.
    Since 8 = 2^3, maybe we can use a tensor product structure.
    
    Alternative: Use the fact that 3x3x3 has rank 23 (Landsberg).
    If we can decompose 3x3x8 as a sum of 3x3x3 and 3x3x5, etc.
    
    Let me try a direct numerical search with a restricted ansatz.
    Assume U, V, W have entries in {0, 1, 1/2, -1/2, -1}.
    
    For small rank r, we can try to solve the system.
    """
    start_time = time.time()
    best_rank = n * m * p
    best_solution = None
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But we need to be more careful. For fixed i, the equations are:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j,k
    # This is a system of m*p equations in r variables (for fixed i).
    # Similarly for fixed j and k.
    
    # Let me try a greedy approach: start with schoolbook and try to merge columns.
    U, V, W = _build_schoolbook(n, m, p)
    current_rank = n * m * p
    
    # Precompute signatures for each column
    def get_signature(U_mat, V_mat, W_mat, col, n, m, p):
        sig = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_mat[i*m+j, col] * V_mat[j*p+k, col] * W_mat[k*n+i, col]
                    if abs(prod) > 1e-10:
                        sig.add((i, j, k, prod))
        return sig
    
    # Try to find columns that can be merged
    # Two columns c1 and c2 can be merged if there exists a new column c_new
    # such that for all (i,j,k):
    #   U[i*m+j,c1]*V[j*p+k,c1]*W[k*n+i,c1] + U[i*m+j,c2]*V[j*p+k,c2]*W[k*n+i,c2]
    #   = U_new[i*m+j]*V_new[j*p+k]*W_new[k*n+i]
    
    # This is a hard problem. Let me try a different approach.
    
    # Use a randomized local search:
    # 1. Start with a random rank-r decomposition
    # 2. Iteratively adjust entries to satisfy the constraints
    # 3. Round to nearest half-integer and verify
    
    for r in range(40, min(72, 72)):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with entries in {0, 1}
            U_rand = np.random.randint(0, 2, size=(n*m, r)).astype(float)
            V_rand = np.random.randint(0, 2, size=(m*p, r)).astype(float)
            W_rand = np.random.randint(0, 2, size=(p*n, r)).astype(float)
            
            # Check if this random initialization satisfies the constraints
            # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_rand[k*n+i, c]
                        if abs(s - 1.0) > 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = r
                best_solution = (U_rand, V_rand, W_rand)
                break
        
        if best_solution is not None:
            break
    
    if best_solution is not None:
        return best_solution
    
    return None


def _try_kronecker_decomposition(n, m, p, deadline):
    """
    Try a Kronecker product based decomposition.
    
    If we can write 3x3x8 as a sum of Kronecker products, we might get lower rank.
    
    For example, if 8 = 4*2, we could try to decompose as:
    (3x3x4) ⊗ (1x1x2) in some sense.
    """
    start_time = time.time()
    
    # Try to find a decomposition based on the structure of 8 = 2^3
    # Use a tensor product approach
    
    # For now, return None
    return None


def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p  # 72
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = rank_sb
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try structured decomposition
    result = _try_structured_decomposition(n, m, p, deadline)
    if result is not None:
        U_new, V_new, W_new = result
        if _verify(U_new, V_new, W_new, n, m, p):
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # Try Kronecker decomposition
    result = _try_kronecker_decomposition(n, m, p, deadline)
    if result is not None:
        U_new, V_new, W_new = result
        if _verify(U_new, V_new, W_new, n, m, p):
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # Return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
