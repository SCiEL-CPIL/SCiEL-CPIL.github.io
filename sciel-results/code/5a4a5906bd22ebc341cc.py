"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

import time
import random
import numpy as np
from fractions import Fraction

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _verify(U, V, W, n, m, p):
    """Check if decomposition is valid: sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k,c."""
    rank = len(U[0])
    # Build tensors as numpy for speed
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # The multiplication tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c]
    # must equal W[k*n+i,c] summed over c... wait, no.
    # Actually the tensor decomposition means:
    # T_{i,j,k} = sum_{c=0}^{rank-1} U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    # But that's a 3-way tensor. Let me re-read the task.
    
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # The tensor for matrix mult (n x m) * (m x p) -> (n x p) is:
    # T_{i,j,k} where i in [0,n), j in [0,m), k in [0,p)
    # T_{i,j,k} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    # Wait, that doesn't make sense dimensionally. Let me think again.
    
    # Standard tensor decomposition for matmul:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # Tensor form: T_{i,j,k} = sum_c u_{i,j,c} * v_{j,k,c} * w_{i,k,c}
    # where u has shape (n*m, rank), v has shape (m*p, rank), w has shape (n*p, rank)
    # But the task says W has shape (p*n, rank) and uses convention W[k*n+i, r]
    # So W[k*n+i, c] corresponds to w_{i,k,c} with index k*n+i.
    
    # So: T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T_{i,j,k} should be 1 if this is the "identity" tensor... 
    # Actually no. The matmul tensor is defined such that:
    # C_{i,k} = sum_j sum_c A_{i,j} * B_{j,k} * (coefficient)
    # The tensor T_{i,j,k} = 1 for all i,j,k in the standard decomposition.
    # Wait, I think the convention is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # = sum_j sum_c A_{i,j} B_{j,k} * delta_{j,j}  (trivial)
    # In tensor form: C_{i,k} = sum_{j,c} A_{i,j} B_{j,k} T_{i,j,k}^c ... 
    
    # Let me just check: for the schoolbook, U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # sum_c U*V*W = 1*1*1 = 1 for each (i,j,k) triple.
    # So the tensor T_{i,j,k} = 1 for all i,j,k.
    
    # Verification: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1 for all i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(n, m, p, target_rank, seed=0, time_limit=10):
    """Try to find a valid decomposition of given rank using random search."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    start = time.time()
    
    # We need U (n*m x R), V (m*p x R), W (p*n x R) such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Strategy: fix U and V randomly, solve for W
    # For each (i,j,k), we need: W[k*n+i, c] = 1 / sum_c' U[i*m+j,c']*V[j*p+k,c'] ... 
    # That's not linear in W. 
    
    # Alternative: This is a 3-way tensor completion problem.
    # Let's use a different approach: parameterize with structure.
    
    # For 3x3x3, the minimal rank is known to be 23 (Strassen-like).
    # For 3x3x6, we can think of it as 2 copies of 3x3x3 plus cross terms.
    # Actually 3x3 * 3x6 = 3x3 * [3x3, 3x3] so it's like two 3x3x3 matmuls sharing A.
    # The tensor is: T_{i,j,k} = 1 for i in [0,3), j in [0,3), k in [0,6)
    # This is the same as two 3x3x3 tensors stacked in the k dimension.
    
    # A decomposition of 3x3x3 with rank R1 gives a decomposition of 3x3x6 with rank 2*R1
    # by duplicating. But we might do better by sharing some terms.
    
    # Let's try: use known low-rank decompositions of 3x3x3.
    # The best known for 3x3x3 is rank 23 (but that's complex).
    # Simple: rank 27 (schoolbook). 
    # Strassen gives 7 for 2x2x2, can be extended.
    
    # For a practical approach within time limits, let's try:
    # 1. Use schoolbook for 3x3x3 (rank 27), duplicate for 3x3x6 (rank 54) - baseline
    # 2. Try to find rank < 54 using structured search
    
    # Structured approach: exploit that k dimension has 6 elements = 2*3
    # Group k into two groups of 3. The tensor is block-diagonal in a sense.
    
    # Let's try random search with the constraint that entries are small integers/half-integers.
    
    best_result = None
    
    for attempt in range(100):
        if time.time() - start > time_limit:
            break
        
        # Random initialization with small values
        U_init = np_rng.integers(-2, 3, size=(n*m, target_rank)).astype(float)
        V_init = np_rng.integers(-2, 3, size=(m*p, target_rank)).astype(float)
        
        # Solve for W: for each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is nonlinear in W. But if we fix U and V, for each index (k*n+i) in W,
        # it appears in equations for all j. So:
        # For fixed (k,i), the equations are: for all j, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Let a_j(c) = U[i*m+j,c]*V[j*p+k,c]. Then sum_c a_j(c)*W[k*n+i,c] = 1 for all j.
        # This is a linear system in W[k*n+i,:]!
        
        W = np.zeros((p*n, target_rank))
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Build linear system: A * w = 1, where A[j,c] = U[i*m+j,c]*V[j*p+k,c]
                A_mat = np.zeros((m, target_rank))
                for j in range(m):
                    A_mat[j, :] = U_init[i*m+j, :] * V_init[j*p+k, :]
                
                # Solve A_mat @ w = ones(m)
                # Least squares
                try:
                    w, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                except:
                    valid = False
                    break
                
                # Check if it's an exact solution
                if np.linalg.norm(A_mat @ w - np.ones(m)) > 1e-8:
                    valid = False
                    break
                
                W[k*n+i, :] = w
        
        if not valid:
            continue
        
        # Round to nearest half-integer and verify
        U_rounded = np.round(U_init * 2) / 2
        V_rounded = np.round(V_init * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                    if abs(s - 1.0) > 1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            # Convert to lists
            U_list = [[int(x*2)/2 for x in U_rounded[row]] for row in range(n*m)]
            V_list = [[int(x*2)/2 for x in V_rounded[row]] for row in range(m*p)]
            W_list = [[int(x*2)/2 for x in W_rounded[row]] for row in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to improve from lower ranks
    # Known: 3x3x3 has rank at most 23, so 3x3x6 should be at most 46
    # Try ranks from 40 up to 53
    for target_rank in range(40, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            result = _try_rank(n, m, p, target_rank, seed=seed, time_limit=8)
            if result is not None:
                best = result
                best_rank = target_rank
                break  # Found for this rank, try lower
    
    return best
# EVOLVE-BLOCK-END
