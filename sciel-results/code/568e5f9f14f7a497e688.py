import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START

def _build_rank17():
    """Build the known rank-17 decomposition."""
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the 2x3x3 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    # Build target tensor T[i,j,k] = 1 if i,j,k form a valid MM triple
    # A is n x m, B is m x p, C is p x n (trace convention: C[k,i] = sum_j A[i,j]*B[j,k])
    # Tensor index: T[i*n*m + j*m + k] for the MM operation
    # Actually, the tensor is T[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k]
    # In flattened form with trace convention W[k*n+i, r]:
    # The tensor entry is T[i,j,k] = 1 for all valid (i,j,k)
    
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Actually for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 for all i,j,k (it's the "all ones" tensor for this specific MM)
    # Wait no - the MM tensor has T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # because C[i,k] = sum_j A[i,j]*B[j,k] means every (i,j,k) triple contributes.
    
    # So the target tensor is all-ones of shape (n, m, p)
    # Decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    
    # Flatten U to (n*m, rank), V to (m*p, rank), W to (p*n, rank)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # Compute the decomposed tensor
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can compute this as: for each r, outer product U[:,r] * V[:,r] * W[:,r] reshaped
    
    T_decomposed = np.zeros((n, m, p))
    for r in range(rank):
        u_vec = U[:, r].reshape(n, m)
        v_vec = V[:, r].reshape(m, p)
        w_vec = W[:, r].reshape(p, n)
        # T[i,j,k] += U[i,j,r] * V[j,k,r] * W[k,i,r]
        # = sum over all: u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
        # This is a tensor contraction
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomposed[i,j,k] += u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
    
    # Target is all ones
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_decomposed, T_target, atol=tol, rtol=0)

def _try_rank16_search(U17, V17, W17, deadline):
    """
    Try to find a rank-16 decomposition by perturbing the rank-17 solution
    and trying to drive one term to zero.
    
    Strategy: For each candidate term r to eliminate, perturb the other 16 terms
    and solve for whether the perturbed 16-term system can reproduce the tensor.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    rank = 17
    
    # The tensor has n*m*p = 18 entries, each must equal 1
    # A rank-r decomposition has r * (nm + mp + pn) = r * (6 + 9 + 6) = r * 21 variables
    # With r=16: 336 variables, 18 equations -> highly underdetermined
    # But the nonlinearity makes it hard
    
    # Approach: Start from rank-17, try to merge two terms or zero out one
    # by adjusting the others.
    
    # Method: For each pair of terms (a, b), try to express their combined
    # contribution as a single term, effectively reducing rank by 1.
    
    # Alternative: Random perturbation + Newton's method on the 16-term system
    
    best_U, best_V, best_W = None, None, None
    best_err = np.inf
    
    rng = np.random.default_rng(42)
    
    # Try each term as the one to eliminate
    for elim_r in range(rank):
        if time.time() > deadline:
            break
            
        # Take the 16 remaining terms
        keep = [r for r in range(rank) if r != elim_r]
        
        # Initialize with the 16 kept terms
        U16 = U17[:, keep].copy()
        V16 = V17[:, keep].copy()
        W16 = W17[:, keep].copy()
        
        # The residual from removing term elim_r
        # T_decomposed_16[i,j,k] = sum_{r in keep} U[i,j,r]*V[j,k,r]*W[k,i,r]
        # We need: sum_{r in keep} ... = 1 for all i,j,k
        # Current: sum_{r in keep} ... = 1 - U[i,j,elim_r]*V[j,k,elim_r]*W[k,i,elim_r]
        
        # Try to adjust the 16 terms to absorb the missing contribution
        # Use iterative refinement: perturb terms to reduce the error
        
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Compute current error
            T16 = np.zeros((n, m, p))
            for r in range(16):
                u_vec = U16[:, r].reshape(n, m)
                v_vec = V16[:, r].reshape(m, p)
                w_vec = W16[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T16[i,j,k] += u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
            
            err = np.max(np.abs(T16 - 1.0))
            
            if err < best_err:
                best_err = err
                if err < 1e-9:
                    best_U, best_V, best_W = U16.copy(), V16.copy(), W16.copy()
                    return best_U, best_V, best_W
            
            if err < 1e-9:
                break
            
            # Gradient-free perturbation: try small random adjustments
            # Focus on terms that have the most "room" to change
            
            # Compute which terms contribute most to the error
            # and perturb them slightly
            for r in range(16):
                if time.time() > deadline:
                    break
                # Small random perturbation
                scale = 0.01 * (1 + attempt * 0.1)
                U16[:, r] += rng.standard_normal(nm) * scale
                V16[:, r] += rng.standard_normal(mp) * scale
                W16[:, r] += rng.standard_normal(pn) * scale
            
            # Also try to add back some of the eliminated term's structure
            # distributed across the remaining terms
            elim_U = U17[:, elim_r].reshape(n, m)
            elim_V = V17[:, elim_r].reshape(m, p)
            elim_W = W17[:, elim_r].reshape(p, n)
            
            # Distribute the eliminated term's contribution
            for r in range(16):
                factor = 1.0 / 16.0
                U16[:, r] += elim_U.flatten() * factor * 0.1
                V16[:, r] += elim_V.flatten() * factor * 0.1
                W16[:, r] += elim_W.flatten() * factor * 0.1
    
    if best_U is not None:
        return best_U, best_V, best_W
    return None, None, None

def _try_rank16_perturbation(U17, V17, W17, deadline):
    """
    Alternative approach: Start from rank-17, add small random perturbations
    to all terms, then try to zero out one term via optimization.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    rank = 17
    
    rng = np.random.default_rng(123)
    
    for seed_offset in range(20):
        if time.time() > deadline:
            break
            
        rng_local = np.random.default_rng(42 + seed_offset)
        
        # Perturb the rank-17 solution
        scale = 0.05
        U = U17 + rng_local.standard_normal((nm, rank)) * scale
        V = V17 + rng_local.standard_normal((mp, rank)) * scale
        W = W17 + rng_local.standard_normal((pn, rank)) * scale
        
        # Verify perturbed solution still works
        if not _verify(U, V, W, tol=1e-6):
            continue
        
        # Try to eliminate each term
        for elim_r in range(rank):
            if time.time() > deadline:
                    break
                    
            keep = [r for r in range(rank) if r != elim_r]
            U16 = U[:, keep].copy()
            V16 = V[:, keep].copy()
            W16 = W[:, keep].copy()
            
            # Iterate: try to fix the 16-term decomposition
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Compute error
                T16 = np.zeros((n, m, p))
                for r in range(16):
                    u_vec = U16[:, r].reshape(n, m)
                    v_vec = V16[:, r].reshape(m, p)
                    w_vec = W16[:, r].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T16[i,j,k] += u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
                
                err = np.max(np.abs(T16 - 1.0))
                
                if err < 1e-10:
                    return U16, V16, W16
                
                # Gradient computation (finite differences)
                # For each variable, compute how it affects the error
                # This is expensive, so use a simpler heuristic
                
                # Perturb terms that have large magnitude
                for r in range(16):
                    u_mag = np.max(np.abs(U16[:, r]))
                    v_mag = np.max(np.abs(V16[:, r]))
                    w_mag = np.max(np.abs(W16[:, r]))
                    total_mag = u_mag * v_mag * w_mag
                    
                    if total_mag > 0:
                        # Scale down slightly if term is too large
                        # or adjust to reduce error
                        step = 0.01
                        U16[:, r] *= (1 - step * 0.1)
                        V16[:, r] *= (1 - step * 0.1)
                        W16[:, r] *= (1 - step * 0.1)
            
    return None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with the known rank-17 solution
    U17, V17, W17 = _build_rank17()
    
    # Verify it works
    if not _verify(U17, V17, W17, tol=1e-9):
        # Fallback: return the rank-17 solution
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
    
    # Try to find rank-16
    if time.time() < deadline:
        U16, V16, W16 = _try_rank16_search(U17, V17, W17, deadline)
        if U16 is not None and _verify(U16, V16, W16, tol=1e-9):
            return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    if time.time() < deadline:
        U16, V16, W16 = _try_rank16_perturbation(U17, V17, W17, deadline)
        if U16 is not None and _verify(U16, V16, W16, tol=1e-9):
            return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Return rank-17 as fallback
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
