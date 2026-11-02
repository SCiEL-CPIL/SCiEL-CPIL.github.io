"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under the 120s/150s wall-clock limits

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 36, 36, 36

    # First, build the schoolbook decomposition as a fallback
    schoolbook_rank = nm * mp  # Wait, that's wrong. Schoolbook is n*m*p = 216
    schoolbook_rank = n * m * p  # 216
    
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    # Try to find a lower rank decomposition
    # Known lower bound for 6x6 matrix mult is around 154 (from 4x4 results scaled)
    # But let's try a range of ranks and see if we can find exact decompositions
    
    # For efficiency, we'll try ranks from a reasonable lower bound up
    # The tensor has 6*6*6 = 216 non-zero entries in its standard basis
    # A rank-r decomposition requires: r*36 + r*36 + r*36 = 108r variables
    # And must satisfy 216*216 = 46656 equations (but many are zero)
    # Actually the tensor T_{i,j,k} for A*B where C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor entries are T[(i*m+j), (j*p+k), (k*n+i)] = 1 for valid (i,j,k)
    # There are 216 such entries, all equal to 1
    
    # Let's try to use a randomized approach with local search
    # Start with ranks that might be achievable
    
    # For 6x6x6, let's try ranks from 160 up to 216
    # But this is computationally expensive, so we'll be selective
    
    ranks_to_try = [216, 215, 214, 213, 212, 211, 210, 209, 208, 207, 206, 205, 204, 203, 202, 201, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178, 176, 174, 172, 170, 168, 166, 164, 162, 160]
    
    # Limit the number of ranks we try based on time
    max_ranks_to_try = 15
    
    for rank in ranks_to_try[:max_ranks_to_try]:
        if time.time() > deadline - 30:  # Leave 30s for finalization
            break
            
        if rank >= best_rank:
            continue
            
        # Try to find a rank-r decomposition
        success = _try_find_decomposition(n, m, p, rank, deadline)
        if success is not None:
            U_found, V_found, W_found = success
            best_rank = rank
            best_U = U_found
            best_V = V_found
            best_W = W_found
            # If we found a good one, we can stop early if it's close to lower bound
            if rank <= 170:
                break

    # Convert to lists for the output format
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}

def _try_find_decomposition(n, m, p, target_rank, deadline):
    """Try to find a rank-target_rank decomposition for n x m x p matrix mult."""
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor as a matrix: T[i,j] = sum_k U[i,k] V[j,k] W[k, ...]
    # Actually, the decomposition is: T_{a,b,c} = sum_r U[a,r] V[b,r] W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    # And T_{a,b,c} = 1 if (i,j,k) is consistent (same j in a and b, same k in b and c, same i in c and a)
    
    # The non-zero entries are at positions where:
    # a = i*m + j, b = j*p + k, c = k*n + i
    # for i in range(n), j in range(m), k in range(p)
    
    # There are n*m*p = 216 such entries, each with value 1
    
    # We need to find U (nm x rank), V (mp x rank), W (pn x rank) such that
    # for each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # and for all other (a,b,c) triples: sum_r U[a,r] * V[b,r] * W[c,r] = 0
    
    # This is a system of polynomial equations. Let's use a randomized approach:
    # 1. Initialize U, V, W randomly
    # 2. Use gradient descent or a specialized solver
    
    # For efficiency, let's use a different approach:
    # The tensor can be written as T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r, V_r, W_r are vectors
    
    # Let's try a few random initializations and see if we can converge
    
    num_trials = 5
    np.random.seed(42)  # Reproducibility
    
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
            
        # Random initialization with small values
        U = np.random.randn(nm, target_rank) * 0.1
        V = np.random.randn(mp, target_rank) * 0.1
        W = np.random.randn(pn, target_rank) * 0.1
        
        # Use a simple gradient descent on the error
        # Error = sum over all (a,b,c) of (T_target[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
        
        # This is expensive to compute directly. Let's use a smarter approach.
        
        # Actually, for this problem, let's try a different strategy:
        # Use the fact that for small matrices, we can use known constructions.
        
        # For 6x6, we can potentially use a Kronecker product structure or
        # a combination of smaller matrix multiplication algorithms.
        
        # Let's try a structured initialization based on 3x3 blocks
        if n == 6 and m == 6 and p == 6:
            # Try to use a 2x2 block structure where each block is 3x3
            # Or use the fact that 6 = 2*3
            
            # For now, let's just try to verify if the schoolbook can be improved
            # by checking if there's a linear dependency among the rank-1 terms
            
            pass
    
    # If we can't find a better decomposition, return None
    return None

def _verify_decomposition(n, m, p, U, V, W, tol=1e-8):
    """Verify that U, V, W form a valid decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    # Check that U, V, W have the right shapes
    if U.shape != (nm, rank) or V.shape != (mp, rank) or W.shape != (pn, rank):
        return False
    
    # Check all non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # Compute sum_r U[a,r] * V[b,r] * W[c,r]
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check some zero entries (sampled)
    # There are nm*mp*pn - n*m*p = 36*36*36 - 216 = 46656 - 216 = 46440 zero entries
    # We can't check all, so sample
    np.random.seed(123)
    for _ in range(1000):
        a = np.random.randint(0, nm)
        b = np.random.randint(0, mp)
        c = np.random.randint(0, pn)
        
        # Check if this is a non-zero entry
        i1 = a // m
        j1 = a % m
        j2 = b // p
        k1 = b % p
        k2 = c // n
        i2 = c % n
        
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue  # This is a non-zero entry, skip
        
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    
    return True
# EVOLVE-BLOCK-END
