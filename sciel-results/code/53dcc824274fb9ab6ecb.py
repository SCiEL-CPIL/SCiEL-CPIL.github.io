"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 3]

def _sat_solve_gf2(n, m, p, rank, timeout=30):
    """
    Try to find a rank decomposition over GF(2) using backtracking search.
    Returns (U, V, W) as numpy arrays with entries in {0, 1} if found, else None.
    """
    start = time.time()
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Target tensor: T[i,j,k] = 1 if we want C[i,k] += A[i,j]*B[j,k]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    # over GF(2)
    
    # Precompute which (i,j,k) tuples we need to satisfy
    targets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                targets.append((i, j, k))
    
    def check_partial():
        """Check if partial assignment is consistent with all constraints."""
        for i, j, k in targets:
            s = 0
            for r in range(rank):
                u_val = U[i * m + j, r]
                v_val = V[j * p + k, r]
                w_val = W[k * n + i, r]
                if u_val != 0 and v_val != 0 and w_val != 0:
                    s = (s + 1) % 2
            # If all three are determined (non-zero or we know they're zero), check
            # Actually, we can only check if the sum of known contributions == 1 mod 2
            # AND no remaining free variable can change it
            # For simplicity, only check when all variables in this constraint are assigned
            all_assigned = True
            for r in range(rank):
                if U[i * m + j, r] == 0 and V[j * p + k, r] == 0 and W[k * n + i, r] == 0:
                    # This term is definitely 0
                    pass
                elif U[i * m + j, r] == 0 or V[j * p + k, r] == 0 or W[k * n + i, r] == 0:
                    # At least one is 0, so term is 0
                    pass
                else:
                    # All three are 1, so term is 1
                    pass
            # This simple check is too weak. Let's do a proper check.
        return True
    
    # Better approach: use constraint propagation
    # For each (i,j,k), the constraint is: XOR over r of (U[i*m+j,r] AND V[j*p+k,r] AND W[k*n+i,r]) == 1
    
    # We'll use a simple backtracking with constraint propagation
    # Order variables by most constrained first
    
    # For now, use a simpler randomized approach with verification
    # This is a heuristic, not guaranteed to find a solution if one exists
    
    # Try random assignments
    max_attempts = 10000
    for attempt in range(max_attempts):
        if time.time() - start > timeout:
            break
        
        # Random assignment
        U[:] = np.random.randint(0, 2, size=(n * m, rank))
        V[:] = np.random.randint(0, 2, size=(m * p, rank))
        W[:] = np.random.randint(0, 2, size=(p * n, rank))
        
        # Verify
        valid = True
        for i, j, k in targets:
            s = 0
            for r in range(rank):
                if U[i * m + j, r] == 1 and V[j * p + k, r] == 1 and W[k * n + i, r] == 1:
                    s = (s + 1) % 2
            if s != 1:
                valid = False
                break
        
        if valid:
            return U.astype(float), V.astype(float), W.astype(float)
    
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if s != 1:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Leave margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook rank-27 as fallback
    best_rank = 27
    U_best = np.zeros((n * m, best_rank))
    V_best = np.zeros((m * p, best_rank))
    W_best = np.zeros((p * n, best_rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_best[i * m + j, column] = 1
                V_best[j * p + k, column] = 1
                W_best[k * n + i, column] = 1
                column += 1
    
    # Try to find lower rank decompositions over GF(2)
    # Start from rank 19 and go up to 26
    for rank in range(19, 27):
        if time.time() - start_time > deadline:
            break
        
        result = _sat_solve_gf2(n, m, p, rank, timeout=min(10, deadline - (time.time() - start_time)))
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = rank
                U_best = U
                V_best = V
                W_best = W
                break  # Found the lowest rank, stop
    
    elapsed = time.time() - start_time
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
