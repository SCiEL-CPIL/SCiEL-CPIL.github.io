"""Find rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication."""

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]
START_TIME = time.time()
TIME_LIMIT = 140  # seconds, safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[k*p*n + i*p + j] for output W[k*n+i, r] convention.
    
    For A (n x m), B (m x p), C = A @ B (n x p).
    The task says output factor follows W[k*n+i, r] convention.
    Wait, let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    
    Actually, let me think about this more carefully.
    
    Standard matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    where i in [0,n), j in [0,m), k in [0,p).
    
    The tensor T has indices (i, j, k) with T[i,j,k] = 1 if this term contributes to C[i,k].
    
    The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Wait, the problem says:
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    So U is indexed by (i,j) flattened to i*m+j
    V is indexed by (j,k) flattened to j*p+k
    W is indexed by (k,i) flattened to k*n+i
    
    And the tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This is the standard matrix multiplication tensor.
    """
    # T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Actually the matrix multiplication tensor is:
    # T[i,j,k] = 1 (it's just the identity in the sense that each term A[i,j]*B[j,k] contributes to C[i,k])
    
    # So we need to find U, V, W such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a rank-3 tensor decomposition where the tensor T has all ones entries
    # in the appropriate 3D array of shape (n*m, m*p, p*n) but with the constraint
    # that only "compatible" triples (i,j,k) have value 1.
    
    # Wait, ALL triples (i,j,k) have value 1 because for any i,j,k, the term A[i,j]*B[j,k] 
    # does contribute to C[i,k]. So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    
    # But not all triples (a,b,c) where a in [0,n*m), b in [0,m*p), c in [0,p*n) 
    # correspond to valid (i,j,k). Let's be more careful.
    
    # a = i*m + j for i in [0,n), j in [0,m)
    # b = j*p + k for j in [0,m), k in [0,p)  
    # c = k*n + i for k in [0,p), i in [0,n)
    
    # So the tensor T of shape (n*m, m*p, p*n) has T[a,b,c] = 1 iff:
    #   a = i*m+j, b = j'*p+k, c = k'*n+i' 
    #   for some i,j,k in ranges AND j=j' AND k=k' AND i=i'
    #   i.e., the j from a equals the j from b, the k from b equals the k from c, and the i from c equals the i from a.
    
    # So T[a,b,c] = 1 iff:
    #   a % m == b // p  (j matches)
    #   b % p == c // n  (k matches)
    #   a // m == c % n  (i matches)
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check shapes
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    # Reconstruct tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    # Check against target
    T_target = _build_tensor(n, m, p)
    
    # Check exactness (allow tiny floating point error)
    if not np.allclose(T, T_target, atol=1e-10):
        return False
    
    return True

def _is_half_integer(x):
    """Check if x is a half-integer (integer or integer + 0.5)."""
    return abs(2*x - round(2*x)) < 1e-10

def _check_half_integers(U, V, W):
    """Check if all entries are half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if not _is_half_integer(val):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook rank = n*m*p = 60
    rank_schoolbook = n * m * p
    
    # First, build the schoolbook solution as fallback
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Now try to find a lower rank decomposition
    # The known rank of 3x4*4x5 matrix multiplication:
    # R(3,4,5) - this is a specific case. Let's think about what's achievable.
    # 
    # For 2x2x2, rank is 7 (Strassen).
    # For 3x3x3, the rank is at most 23 (and at least 19).
    # 
    # For 3x4x5, we can potentially use Kronecker products or other constructions.
    # But let's try a direct search approach.
    
    # Build the target tensor
    T_target = _build_tensor(n, m, p)
    
    # Try ranks from lower to higher
    # Lower bound: rank >= max(n*m, m*p, p*n) is not quite right
    # Actually rank >= max(n*p, m*n, m*p) for the "standard" lower bounds
    # More precisely, for matrix multiplication R(n,m,p), we have R >= max(n*m, m*p, n*p)
    # Actually that's not right either. The trivial lower bound is R >= n*p (since C has n*p entries
    # and each rank-1 term contributes at most 1 to each entry... no that's not right).
    
    # Let's just try a range of ranks. Start from a reasonable lower bound.
    # For 3x4x5: n*p=15, m*n=12, m*p=20. So rank >= 20 is a loose lower bound.
    # Actually the rank must be at least the max of the "support" sizes in some sense.
    
    # Let's try ranks from 20 up to 60 (schoolbook)
    # We'll use a greedy/iterative approach to try to find decompositions
    
    # For each candidate rank r, we try to find U, V, W such that the decomposition holds.
    # This is a non-convex problem, so we use random initialization + iterative refinement.
    
    def try_find_decomposition(target_rank, n, m, p, T_target, max_iter=500, seed=42):
        """Try to find a rank decomposition of the given rank."""
        np.random.seed(seed)
        
        # Initialize with random values
        U_init = np.random.randn(n*m, target_rank)
        V_init = np.random.randn(m*p, target_rank)
        W_init = np.random.randn(p*n, target_rank)
        
        # Use iterative methods to refine
        # We want: sum_r U[:,r] outer V[:,r] outer W[:,r] = T_target
        
        # One approach: alternate minimization
        # Fix V and W, solve for U (linear)
        # Fix U and W, solve for V (linear)  
        # Fix U and V, solve for W (linear)
        
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for iteration in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT:
                return None
            
            # Update U: for each a, U[a,r] should satisfy
            # sum_r U[a,r] * V[b,r] * W[c,r] = T_target[a,b,c]
            # This is: U[a,:] * (V[b,:] elementwise W[c,:]) summed over r = T[a,b,c]
            # For each a, we have a system: U[a,:] @ (V*W)_ac = T[a,c] where (V*W)_ac[r] = V[?,r]*W[c,r]
            # Actually this is more complex. Let me think differently.
            
            # The tensor equation is: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # 
            # If we fix V and W, then for each (b,c), we need:
            # sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
            # This is a linear system in U[a,:] for each a, but the RHS depends on (b,c).
            # Actually, for fixed (b,c), the equation is:
            # U[:, r] dot (V[b,r] * W[c,r]) = T[:, b, c] for all a
            # So: U @ diag(V[b,:] * W[c,:]) = T[:, b, c]
            # This gives us a linear system for U (all rows at once) for each (b,c).
            # But we have multiple (b,c) pairs, so we can combine them.
            
            # Actually, let's flatten: the equation is
            # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
            # 
            # If we fix V and W, this is linear in U. We can set up a big linear system:
            # For each (a, b, c): sum_r U[a,r] * (V[b,r]*W[c,r]) = T[a,b,c]
            # 
            # This is: for each r, U[:,r] is a column vector of length n*m.
            # The coefficient of U[a,r] in equation (a,b,c) is V[b,r]*W[c,r].
            # 
            # So we have (n*m * m*p * p*n) equations and (n*m * rank) unknowns.
            # This is an overdetermined system. We can use least squares.
            
            # But this might be expensive. Let's try a different approach.
            
            # Alternative: use the fact that this is a tensor rank decomposition.
            # We can use an alternating least squares approach where we update one factor at a time.
            
            # Update U: fix V, W. For each a, we want to find U[a,:] such that
            # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all b,c.
            # This is: U[a,:] @ (V*W)_b,c = T[a,b,c] where (V*W)_b,c[r] = V[b,r]*W[c,r]
            # Flatten: for fixed a, we have a system with variables U[a,0],...,U[a,rank-1]
            # and equations for each (b,c): sum_r U[a,r] * V[b,r]*W[c,r] = T[a,b,c]
            # This is a linear system: A_a @ U[a,:] = t_a where A_a has rows indexed by (b,c)
            # and A_a[(b,c), r] = V[b,r]*W[c,r], t_a[(b,c)] = T[a,b,c]
            
            for a in range(n*m):
                # Build system for U[a,:]
                A_a = np.zeros((m*p*p*n, target_rank))
                t_a = np.zeros(m*p*p*n)
                idx = 0
                for b in range(m*p):
                    for c in range(p*n):
                        A_a[idx] = V[b, :] * W[c, :]
                        t_a[idx] = T_target[a, b, c]
                        idx += 1
                # Solve least squares
                try:
                    U[a, :], _, _, _ = np.linalg.lstsq(A_a, t_a, rcond=None)
                except:
                    pass
            
            # Update V similarly
            for b in range(m*p):
                A_b = np.zeros((n*m*p*n, target_rank))
                t_b = np.zeros(n*m*p*n)
                idx = 0
                for a in range(n*m):
                    for c in range(p*n):
                        A_b[idx] = U[a, :] * W[c, :]
                        t_b[idx] = T_target[a, b, c]
                        idx += 1
                try:
                    V[b, :], _, _, _ = np.linalg.lstsq(A_b, t_b, rcond=None)
                except:
                    pass
            
            # Update W similarly
            for c in range(p*n):
                A_c = np.zeros((n*m*m*p, target_rank))
                t_c = np.zeros(n*m*m*p)
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        A_c[idx] = U[a, :] * V[b, :]
                        t_c[idx] = T_target[a, b, c]
                        idx += 1
                try:
                    W[c, :], _, _, _ = np.linalg.lstsq(A_c, t_c, rcond=None)
                except:
                    pass
            
            # Check convergence
            T_recon = np.zeros((n*m, m*p, p*n))
            for r in range(target_rank):
                T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(T_recon - T_target) / np.linalg.norm(T_target)
            
            if err < 1e-10:
                # Convert to lists with half-integer rounding if possible
                U_list = [[round(val * 2) / 2 for val in row] for row in U]
                V_list = [[round(val * 2) / 2 for val in row] for row in V]
                W_list = [[round(val * 2) / 2 for val in row] for row in W]
                
                # Verify with rounded values
                if _validate(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list, "rank": target_rank}
                
                # Try without rounding
                U_list = [[val for val in row] for row in U]
                V_list = [[val for val in row] for row in V]
                W_list = [[val for val in row] for row in W]
                
                if _validate(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list, "rank": target_rank}
        
        return None
    
    # Try ranks from lower to upper
    # Lower bound: at least n*p = 15, but likely higher
    # Let's try from 20 to 59
    for r in range(20, min(60, rank_schoolbook)):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Try multiple random initializations
        for seed in range(5):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            result = try_find_decomposition(r, n, m, p, T_target, max_iter=200, seed=seed)
            if result is not None:
                # Found a valid decomposition with rank r
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}
                    # Can stop searching this rank
                    break
        
        if r < best_rank:
            # Found something better, continue to see if we can go lower
            continue
    
    return best_result
