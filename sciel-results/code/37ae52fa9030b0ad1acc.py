import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W form a valid decomposition.
    
    For each (i,j,k), we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a valid entry
    Actually, the condition is:
    For the matrix multiplication tensor T[i,j,k] = delta_{i,j,k} (which is 1 for all valid indices),
    we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    """
    # Build the tensor T where T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The decomposition gives: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should equal 1 for all (i,j,k)
    
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False, rank
    
    # Check dimensions
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False, rank
    
    # Verify: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # We can vectorize this check
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # For each (i,j,k): sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # We can compute this as a tensor contraction
    
    # Create a 4D tensor: A[i,j,k,r] = U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Then sum over r
    
    # More efficient: for each r, compute the outer product structure
    # Actually, let's just check a subset or use numpy efficiently
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # = sum_r (U3[i,j,r] * W3[k,i,r]) * V3[j,k,r]
    
    # Let's build it step by step
    # For efficiency, compute in chunks or use einsum
    
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Use einsum: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # We want T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Reshape to 3D: U3 (n,m,rank), V3 (m,p,rank), W3 (p,n,rank)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    
    # This is a 3-way tensor contraction. Let's compute it.
    # For small dimensions, we can just do it directly.
    
    T = np.zeros((n, m, p))
    for r in range(rank):
        # U3[:, :, r] is (n, m), V3[:, :, r] is (m, p), W3[:, :, r] is (p, n)
        # We need: T[i,j,k] += U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # = (U3[:,:,r] outer with W3[:,:,r] transposed?) No...
        # U3[i,j,r] * W3[k,i,r] gives a (k, i, j) structure... 
        # Let's think: for fixed r, we have three vectors/matrix slices.
        # U3[:,:,r] is n×m, V3[:,:,r] is m×p, W3[:,:,r] is p×n
        # We want T_r[i,j,k] = U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # This is the elementwise product of three "diagonal" tensors.
        
        # Create a 3D array: for each (i,j,k), multiply U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # We can do: 
        # temp1[i,j,k] = U3[i,j,r] * W3[k,i,r]  -> this is (n,m,p) where temp1[i,j,k] = U3[i,j,r]*W3[k,i,r]
        # Then T_r[i,j,k] = temp1[i,j,k] * V3[j,k,r]
        
        # U3[:,:,r] is (n,m), W3[:,:,r] is (p,n)
        # We want temp1[i,j,k] = U3[i,j,r] * W3[k,i,r]
        # = U3[:,:,r] (n,m) broadcast with W3[:,:,r].T (n,p) -> but we need (n,m,p)
        # temp1[i,j,k] = U3[i,j,r] * W3[k,i,r]
        # This is: for each i, U3[i,:,r] (m,) * W3[:,i,r] (p,) -> outer product gives (m,p)
        # So temp1[i] = np.outer(U3[i,:,r], W3[:,i,r]) which is (m,p)
        # Then T_r[i,j,k] = temp1[i,j,k] * V3[j,k,r]
        
        for i in range(n):
            # u_i = U3[i,:,r] shape (m,)
            # w_i = W3[:,i,r] shape (p,)
            # v = V3[:,:,r] shape (m,p)
            # temp1[i] = outer(u_i, w_i) shape (m,p)
            # T_r[i] = temp1[i] * v shape (m,p)
            u_i = U3[i, :, r]  # (m,)
            w_i = W3[:, i, r]  # (p,)
            v = V3[:, :, r]    # (m,p)
            # T[i, :, :] += outer(u_i, w_i) * v
            T[i, :, :] += np.outer(u_i, w_i) * v
    
    # Check if T is all ones
    if not np.allclose(T, 1.0, atol=tol, rtol=tol):
        return False, rank
    return True, rank

def _try_randomized_search(n, m, p, max_rank, time_limit, rng):
    """Try to find a decomposition of rank <= max_rank using randomized methods.
    
    Uses the idea that we can randomly initialize U, V and try to solve for W,
    or use algebraic methods.
    
    For now, we'll try a structured approach: assume some sparsity patterns.
    """
    start = time.time()
    best_rank = None
    best_decomp = None
    
    # Strategy 1: Try ranks from a lower bound up to max_rank
    # Lower bound for matrix multiplication rank: max(n*m, m*p, p*n) is not quite right
    # Actually, the rank is at least max(n*m, m*p, p*n) / something... 
    # The trivial lower bound is max(n*m, m*p, p*n) but that's not tight.
    # A better lower bound: rank >= max(n, m, p) * something
    # For now, let's try from a reasonable lower bound
    
    # The schoolbook rank is n*m*p
    schoolbook = n * m * p
    
    # Try to find lower rank decompositions
    # For 3x5x6, schoolbook = 90
    # Let's try various ranks
    
    # We'll use a heuristic: try ranks that are multiples of gcd or have special structure
    candidate_ranks = []
    for r in range(max(n*m, m*p, p*n), min(max_rank + 1, schoolbook + 1)):
        candidate_ranks.append(r)
    
    # Also try some specific values
    for r in [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]:
        if max(n*m, m*p, p*n) <= r <= max_rank:
            if r not in candidate_ranks:
                candidate_ranks.append(r)
    
    candidate_ranks = sorted(set(candidate_ranks))
    
    for target_rank in candidate_ranks:
        if time.time() - start > time_limit:
            break
        if best_rank is not None and target_rank >= best_rank:
            break
        
        # Try to find a decomposition of rank target_rank
        # Method: Random initialization + least squares for W given U, V
        
        n_trials = 20  # number of random trials per rank
        for trial in range(n_trials):
            if time.time() - start > time_limit:
                break
            
            # Random U, V with entries from {-1, 0, 1}
            U = rng.integers(-1, 2, size=(n * m, target_rank)).astype(float)
            V = rng.integers(-1, 2, size=(m * p, target_rank)).astype(float)
            
            # We need to find W such that:
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k)
            
            # This is a linear system in W!
            # For each (i,j,k): sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
            # Let A[i,j,k,r] = U[i*m+j,r] * V[j*p+k,r]
            # Then sum_r A[i,j,k,r] * W[k*n+i,r] = 1
            # But W is indexed by (k*n+i, r), so this couples i and k in the row index of W.
            
            # Let's reorganize. Let w_{a,r} where a = k*n + i.
            # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * w_{k*n+i, r} = 1
            
            # This is a system of n*m*p equations in p*n*rank unknowns.
            # But the equations are not independent in a simple way because the same w appears in multiple equations.
            
            # Let's set up the linear system properly.
            # Unknowns: w_{a,r} for a in [0, p*n), r in [0, rank)
            # Total unknowns: p*n*rank
            
            # Equations: for each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * w_{k*n+i,r} = 1
            # Total equations: n*m*p
            
            # This is a linear system: M * w = b where b is all ones
            # M has shape (n*m*p, p*n*rank)
            
            # For efficiency, let's build M sparsely or use structure.
            
            # Actually, for each (i,j,k), the equation only involves w_{k*n+i, r} for all r.
            # So the row for (i,j,k) only has non-zero entries in columns corresponding to a = k*n+i.
            
            # Let's build M as a dense matrix for small problems.
            n_eq = n * m * p
            n_unk = p * n * target_rank
            
            if n_unk > 10000 or n_eq > 10000:
                continue  # Too large for dense solve
            
            M = np.zeros((n_eq, n_unk))
            b = np.ones(n_eq)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = k * n + i
                        for r in range(target_rank):
                            M[eq_idx, a * target_rank + r] = U[i * m + j, r] * V[j * p + k, r]
                        eq_idx += 1
            
            # Solve M * w = b using least squares
            try:
                w, residuals, rank_m, sv = np.linalg.lstsq(M, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if the solution is good
            if residuals.size > 0:
                if residuals[0] > 1e-8:
                    continue
            
            # Check if the solution actually works
            W = w.reshape(p * n, target_rank)
            
            # Verify
            valid, _ = _verify_decomposition(U, V, W, n, m, p, tol=1e-8)
            if valid:
                # Check if entries are close to integers or half-integers
                all_close = True
                for mat in [U, V, W]:
                    # Check if entries are close to integers or half-integers
                    for val in mat.flatten():
                        # Check if close to integer
                        if abs(val - round(val)) > 1e-6:
                            # Check if close to half-integer
                            if abs(val * 2 - round(val * 2)) > 1e-5:
                                all_close = False
                                break
                    if not all_close:
                        break
                if all_close:
                    # Round to nearest integer or half-integer
                    def round_to_half(x):
                        return np.round(x * 2) / 2.0
                    U = round_to_half(U)
                    V = round_to_half(V)
                    W = round_to_half(W)
                    
                    # Re-verify after rounding
                    valid2, _ = _verify_decomposition(U, V, W, n, m, p, tol=1e-6)
                    if valid2:
                        best_rank = target_rank
                        best_decomp = (U, V, W)
                        break  # Found a valid decomposition for this rank
        
        if best_rank is not None and best_rank <= target_rank:
            break
    
    return best_decomp

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Leave margin before 150s kill
    
    # First, get the schoolbook decomposition as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Try to find a better decomposition
    rng = np.random.default_rng(42)
    
    # Time budget for search: use most of the available time
    time_for_search = min(100.0, deadline - (time.time() - start_time) - 10.0)
    
    if time_for_search > 5.0:
        # Try to find lower rank decomposition
        # Start with a reasonable upper bound
        schoolbook_rank = n * m * p
        # Try to find something better
        result = _try_randomized_search(n, m, p, schoolbook_rank - 1, time_for_search, rng)
        
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U_list = U_school.tolist()
    V_list = V_school.tolist()
    W_list = W_school.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
